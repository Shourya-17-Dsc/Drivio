"""FastAPI app: live state, sessions, analytics, reports and ML experiments. Everything runs locally."""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from . import reports
from .config import DATA_DIR, IMAGE_DATASET_DIR, settings
from .auth import auth_dependency, hash_password, issue_token, revoke_token, verify_password
from .database import AuditLog, AlertRecord, DriveSession, DriverProfile, ModelRun, Sample, SessionLocal, UserAccount, init_db
from .health import evaluate_health
from .ml import experiments
from .ml.image_dataset import audit_image_dataset
from .pipeline import MonitoringPipeline

FATIGUE_STATES = ("fatigued", "drowsy")
MAX_DATASET_UPLOAD_BYTES = 10 * 1024 * 1024


def _save_sample(session_id: int, snap: dict) -> None:
    m = snap["metrics"]
    with SessionLocal() as db:
        db.add(Sample(session_id=session_id, score=snap["score"], state=snap["state"], ear=m["ear"], mar=m["mar"],
                      perclos=m["perclos"], closure_s=m["closure_s"], blink_rate=m["blink_rate_pm"],
                      yawns_recent=m["yawns_recent"], yaw=m["yaw"], pitch=m["pitch"],
                      head_turned_s=m["head_turned_s"], safety_score=snap.get("safety", {}).get("score", 100),
                      ground_truth=snap.get("ground_truth")))
        db.commit()


def _save_event(session_id: int, ev: dict) -> None:
    with SessionLocal() as db:
        db.add(AlertRecord(session_id=session_id, level=ev["level"], kind=ev["kind"], state=ev["state"],
                           message=ev["message"]))
        db.commit()


def _save_calibration(driver_id: int, calibration: dict) -> None:
    with SessionLocal() as db:
        profile = db.get(DriverProfile, driver_id)
        if profile:
            profile.calibration_json = json.dumps(calibration)
            db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.pipeline = MonitoringPipeline(settings, on_sample=_save_sample, on_event=_save_event,
                                            on_calibrated=_save_calibration)
    app.state.pipeline.start()
    yield
    app.state.pipeline.stop()


app = FastAPI(title="DRIVIO", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_methods=["*"], allow_headers=["*"])

_image_training_jobs: dict[str, dict] = {}
_image_training_lock = threading.Lock()
_temporal_training_jobs: dict[str, dict] = {}
_temporal_training_lock = threading.Lock()


def get_db():
    with SessionLocal() as db:
        yield db


def get_current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> UserAccount:
    return auth_dependency(authorization, db)


def require_admin(user: UserAccount = Depends(get_current_user)) -> UserAccount:
    if user.role != "admin":
        raise HTTPException(403, "Administrator access required")
    return user


class AuthBody(BaseModel):
    username: str
    password: str


def _user_response(user: UserAccount) -> dict:
    return {"id": user.id, "username": user.username, "role": user.role}


def _audit(db: Session, actor: str, action: str, target: str = "", detail: str | None = None) -> None:
    db.add(AuditLog(actor=actor, action=action, target=target, detail=detail))


@app.post("/api/auth/register")
def register(body: AuthBody, db: Session = Depends(get_db)):
    username = body.username.strip().lower()
    if len(username) < 3 or len(username) > 80:
        raise HTTPException(400, "Username must contain 3-80 characters")
    if db.scalars(select(UserAccount).limit(1)).first() is not None:
        raise HTTPException(409, "The first administrator has already been created")
    try:
        password_hash = hash_password(body.password)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    user = UserAccount(username=username, password_hash=password_hash, role="admin")
    db.add(user)
    db.commit()
    db.refresh(user)
    _audit(db, user.username, "admin_registered", "user", "First administrator created")
    db.commit()
    return {"user": _user_response(user), "token": issue_token(db, user)}


@app.post("/api/auth/login")
def login(body: AuthBody, db: Session = Depends(get_db)):
    user = db.scalar(select(UserAccount).where(UserAccount.username == body.username.strip().lower()))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    return {"user": _user_response(user), "token": issue_token(db, user)}


@app.get("/api/auth/me")
def me(user: UserAccount = Depends(get_current_user)):
    return _user_response(user)


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None), db: Session = Depends(get_db),
           user: UserAccount = Depends(get_current_user)):
    if authorization and authorization.lower().startswith("bearer "):
        revoke_token(db, authorization.split(" ", 1)[1].strip())
    return {"ok": True}


def _summary(db: Session, s: DriveSession) -> dict:
    n, avg, low = db.execute(select(func.count(Sample.id), func.avg(Sample.score), func.min(Sample.score))
                             .where(Sample.session_id == s.id)).one()
    fatigued = db.scalar(select(func.count(Sample.id)).where(Sample.session_id == s.id,
                                                             Sample.state.in_(FATIGUE_STATES))) or 0
    alerts = db.scalar(select(func.count(AlertRecord.id)).where(AlertRecord.session_id == s.id,
                                                                AlertRecord.kind == "escalation")) or 0
    min_safety = db.scalar(select(func.min(Sample.safety_score)).where(Sample.session_id == s.id))
    end = s.ended_at or datetime.utcnow()
    driver = db.get(DriverProfile, s.driver_id) if s.driver_id else None
    return {"id": s.id, "label": s.label, "mode": s.mode, "driver_id": s.driver_id,
            "driver_name": driver.name if driver else None, "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            "duration_s": int((end - s.started_at).total_seconds()), "samples": n,
            "avg_score": round(avg) if avg is not None else None, "min_score": low,
            "fatigue_pct": round(fatigued / n * 100) if n else 0, "alerts": alerts,
            "min_safety_score": min_safety}


# ---- live ----------------------------------------------------------------
@app.get("/api/health")
def health():
    snap = app.state.pipeline.snapshot()
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False
    return evaluate_health(snap, database_ok, settings.store_video)


@app.get("/api/state")
def state():
    return app.state.pipeline.snapshot()


# ---- driver profiles -------------------------------------------------------
class DriverBody(BaseModel):
    name: str


class CalibrateBody(BaseModel):
    driver_id: int | None = None


class GroundTruthBody(BaseModel):
    label: str | None = None


def _driver_response(profile: DriverProfile) -> dict:
    return {"id": profile.id, "name": profile.name, "calibration_available": bool(profile.calibration_json),
            "created_at": profile.created_at.isoformat(), "updated_at": profile.updated_at.isoformat()}


@app.get("/api/drivers")
def drivers(db: Session = Depends(get_db)):
    return [_driver_response(profile) for profile in db.scalars(select(DriverProfile).order_by(DriverProfile.name)).all()]


@app.post("/api/drivers")
def create_driver(body: DriverBody, db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Driver name cannot be empty")
    if db.scalar(select(DriverProfile).where(func.lower(DriverProfile.name) == name.lower())):
        raise HTTPException(409, "A driver with that name already exists")
    profile = DriverProfile(name=name)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return _driver_response(profile)


def _select_driver(profile: DriverProfile, pipe) -> dict:
    calibration = json.loads(profile.calibration_json) if profile.calibration_json else None
    pipe.set_driver(profile.id, profile.name, calibration)
    return {"ok": True, "driver": _driver_response(profile), "calibration_loaded": calibration is not None}


@app.post("/api/drivers/{driver_id}/select")
def select_driver(driver_id: int, db: Session = Depends(get_db)):
    if app.state.pipeline.session_id is not None:
        raise HTTPException(409, "Stop the active session before switching drivers")
    profile = db.get(DriverProfile, driver_id)
    if not profile:
        raise HTTPException(404, "Driver not found")
    return _select_driver(profile, app.state.pipeline)


@app.post("/api/drivers/clear")
def clear_driver():
    if app.state.pipeline.session_id is not None:
        raise HTTPException(409, "Stop the active session before switching drivers")
    app.state.pipeline.set_driver(None)
    return {"ok": True, "driver": None}


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(app.state.pipeline.snapshot()))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/video")
async def video():
    async def frames():
        while True:
            jpg = app.state.pipeline.latest_jpeg()
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/calibrate")
def calibrate(body: CalibrateBody | None = None, db: Session = Depends(get_db)):
    if body and body.driver_id is not None:
        if app.state.pipeline.session_id is not None and body.driver_id != app.state.pipeline.driver_id:
            raise HTTPException(409, "Stop the active session before calibrating another driver")
        profile = db.get(DriverProfile, body.driver_id)
        if not profile:
            raise HTTPException(404, "Driver not found")
        _select_driver(profile, app.state.pipeline)
    app.state.pipeline.recalibrate()
    return {"ok": True, "driver_id": app.state.pipeline.driver_id}


@app.post("/api/session/label")
def label_session(body: GroundTruthBody):
    if app.state.pipeline.session_id is None:
        raise HTTPException(409, "Start a session before assigning labels")
    label = body.label.strip().lower() if body.label else None
    allowed = {None, "alert", "drowsy", "fatigued", "distracted", "non_drowsy"}
    if label not in allowed:
        raise HTTPException(400, "Label must be alert, non_drowsy, fatigued, drowsy or distracted")
    app.state.pipeline.set_ground_truth(label)
    return {"ok": True, "ground_truth": label}


# ---- sessions --------------------------------------------------------------
class StartBody(BaseModel):
    label: str = Field(default="Drive", max_length=120)
    driver_id: int | None = None


@app.post("/api/session/start")
def start_session(body: StartBody, db: Session = Depends(get_db)):
    pipe = app.state.pipeline
    if pipe.session_id is not None:
        raise HTTPException(409, "A session is already recording.")
    if body.driver_id is not None:
        profile = db.get(DriverProfile, body.driver_id)
        if not profile:
            raise HTTPException(404, "Driver not found")
        _select_driver(profile, pipe)
    s = DriveSession(label=body.label.strip() or "Drive", mode=pipe.snapshot().get("mode", "camera"),
                     driver_id=pipe.driver_id)
    db.add(s)
    db.commit()
    pipe.set_session(s.id)
    return _summary(db, s)


@app.post("/api/session/stop")
def stop_session(db: Session = Depends(get_db)):
    pipe = app.state.pipeline
    if pipe.session_id is None:
        raise HTTPException(409, "No session is recording.")
    s = db.get(DriveSession, pipe.session_id)
    pipe.set_session(None)
    s.ended_at = datetime.utcnow()
    db.commit()
    return _summary(db, s)


@app.post("/api/session/break")
def take_break():
    if app.state.pipeline.session_id is None:
        raise HTTPException(409, "No session is recording.")
    app.state.pipeline.take_break()
    return {"ok": True, "safety": app.state.pipeline.snapshot().get("safety", {})}


@app.get("/api/sessions")
def sessions(db: Session = Depends(get_db)):
    rows = db.scalars(select(DriveSession).order_by(DriveSession.id.desc()).limit(100)).all()
    return [_summary(db, s) for s in rows]


def _require(db: Session, session_id: int) -> DriveSession:
    s = db.get(DriveSession, session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: int, db: Session = Depends(get_db)):
    s = _require(db, session_id)
    samples = db.scalars(select(Sample).where(Sample.session_id == session_id).order_by(Sample.t)).all()
    events = db.scalars(select(AlertRecord).where(AlertRecord.session_id == session_id).order_by(AlertRecord.t)).all()
    t0 = s.started_at
    return {
        "summary": _summary(db, s),
        "samples": [{"s": int((x.t - t0).total_seconds()), "score": x.score, "state": x.state, "perclos": x.perclos,
                     "ear": x.ear, "safety_score": x.safety_score, "ground_truth": x.ground_truth} for x in samples],
        "events": [{"s": int((e.t - t0).total_seconds()), "time": e.t.isoformat(), "level": e.level, "kind": e.kind,
                    "state": e.state, "message": e.message} for e in events],
    }


@app.get("/api/sessions/{session_id}/report.csv")
def report_csv(session_id: int, db: Session = Depends(get_db)):
    _require(db, session_id)
    return Response(reports.session_csv(db, session_id), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=drivio-session-{session_id}.csv"})


@app.get("/api/sessions/{session_id}/report.pdf")
def report_pdf(session_id: int, db: Session = Depends(get_db)):
    _require(db, session_id)
    return Response(reports.session_pdf(db, session_id), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=drivio-session-{session_id}.pdf"})


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    s = _require(db, session_id)
    if app.state.pipeline.session_id == session_id:
        raise HTTPException(409, "Stop the session before deleting it.")
    for model in (Sample, AlertRecord):
        for row in db.scalars(select(model).where(model.session_id == session_id)).all():
            db.delete(row)
    db.delete(s)
    db.commit()
    return {"ok": True}


# ---- analytics -------------------------------------------------------------
@app.get("/api/analytics/trends")
def trends(days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.scalars(select(DriveSession).where(DriveSession.started_at >= since)
                      .order_by(DriveSession.started_at)).all()
    per_session = [_summary(db, s) for s in rows]
    dist = dict(db.execute(select(Sample.state, func.count(Sample.id)).join(DriveSession)
                           .where(DriveSession.started_at >= since).group_by(Sample.state)).all())
    return {"sessions": per_session, "state_distribution": dist}


# ---- fleet overview --------------------------------------------------------
@app.get("/api/fleet/overview")
def fleet_overview(days: int = 30, db: Session = Depends(get_db), user: UserAccount = Depends(require_admin)):
    days = max(1, min(days, 365))
    since = datetime.utcnow() - timedelta(days=days)
    profiles = db.scalars(select(DriverProfile).order_by(DriverProfile.name)).all()
    driver_rows = []
    for profile in profiles:
        sessions = db.scalars(select(DriveSession).where(DriveSession.driver_id == profile.id,
                                                          DriveSession.started_at >= since)
                              .order_by(DriveSession.started_at.desc())).all()
        summaries = [_summary(db, session) for session in sessions]
        average_safety = db.scalar(select(func.avg(Sample.safety_score)).join(DriveSession)
                                   .where(DriveSession.driver_id == profile.id,
                                          DriveSession.started_at >= since))
        driver_rows.append({
            "id": profile.id,
            "name": profile.name,
            "sessions": len(summaries),
            "average_safety": round(float(average_safety)) if average_safety is not None else None,
            "lowest_safety": min((s["min_safety_score"] for s in summaries if s["min_safety_score"] is not None), default=None),
            "fatigue_pct": round(sum(s["fatigue_pct"] * s["samples"] for s in summaries) /
                                  max(sum(s["samples"] for s in summaries), 1)),
            "alerts": sum(s["alerts"] for s in summaries),
            "last_session": summaries[0] if summaries else None,
        })
    all_sessions = db.scalars(select(DriveSession).where(DriveSession.started_at >= since)).all()
    return {
        "days": days,
        "totals": {"drivers": len(profiles), "sessions": len(all_sessions),
                   "alerts": sum(_summary(db, session)["alerts"] for session in all_sessions)},
        "active": {"driver_id": app.state.pipeline.driver_id, "driver_name": app.state.pipeline.driver_name,
                   "recording": app.state.pipeline.session_id is not None,
                   "state": app.state.pipeline.snapshot().get("state"),
                   "safety": app.state.pipeline.snapshot().get("safety")},
        "drivers": driver_rows,
    }


@app.get("/api/privacy/status")
def privacy_status(db: Session = Depends(get_db), user: UserAccount = Depends(require_admin)):
    return {"store_video": settings.store_video,
            "sessions": db.scalar(select(func.count(DriveSession.id))) or 0,
            "samples": db.scalar(select(func.count(Sample.id))) or 0,
            "alerts": db.scalar(select(func.count(AlertRecord.id))) or 0}


@app.delete("/api/privacy/recorded-data")
def delete_recorded_data(db: Session = Depends(get_db), user: UserAccount = Depends(require_admin)):
    if app.state.pipeline.session_id is not None:
        raise HTTPException(409, "Stop the active session before deleting recorded data")
    counts = {
        "alerts": db.scalar(select(func.count(AlertRecord.id))) or 0,
        "samples": db.scalar(select(func.count(Sample.id))) or 0,
        "sessions": db.scalar(select(func.count(DriveSession.id))) or 0,
    }
    for model in (AlertRecord, Sample, DriveSession):
        for row in db.scalars(select(model)).all():
            db.delete(row)
    _audit(db, user.username, "recorded_data_deleted", "recorded_data", json.dumps(counts))
    db.commit()
    return {"ok": True, "deleted": counts}


# ---- ML experiments --------------------------------------------------------
class TrainBody(BaseModel):
    dataset: str
    algorithm: str = "random_forest"


class NameBody(BaseModel):
    name: str = "synthetic_demo"


def _ml(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/ml/datasets")
def ml_datasets():
    return experiments.list_datasets()


@app.post("/api/ml/datasets/upload")
async def ml_upload(file: UploadFile = File(...)):
    content = await file.read(MAX_DATASET_UPLOAD_BYTES + 1)
    if len(content) > MAX_DATASET_UPLOAD_BYTES:
        raise HTTPException(413, "Dataset CSV must be 10 MB or smaller")
    return _ml(experiments.save_upload, file.filename or "upload", content)


@app.post("/api/ml/datasets/synthetic")
def ml_synthetic(body: NameBody):
    return _ml(experiments.make_synthetic_dataset, body.name)


@app.post("/api/ml/datasets/from-sessions")
def ml_from_sessions(body: NameBody, db: Session = Depends(get_db)):
    return _ml(experiments.export_sessions, db, body.name)


@app.post("/api/ml/datasets/from-labeled-sessions")
def ml_from_labeled_sessions(body: NameBody, db: Session = Depends(get_db)):
    return _ml(experiments.export_labeled_sessions, db, body.name)


@app.get("/api/ml/temporal-dataset/status")
def temporal_dataset_status(db: Session = Depends(get_db)):
    from .ml.temporal_training import labelled_dataset_status

    rows = db.scalars(select(Sample).where(Sample.ground_truth.is_not(None))).all()
    return labelled_dataset_status(rows)


@app.get("/api/ml/algorithms")
def ml_algorithms():
    return [*experiments.ALGORITHMS.keys(), "mlp_torch"]


@app.get("/api/ml/image-dataset/audit")
def image_dataset_audit():
    """Validate the local Drowsy/Non Drowsy image dataset before training."""
    return audit_image_dataset(IMAGE_DATASET_DIR)


class ImageTrainBody(BaseModel):
    epochs: int = Field(default=3, ge=1, le=30)
    batch_size: int = Field(default=32, ge=1, le=512)
    image_size: Literal[32, 48, 64, 96, 128] = 64
    max_images_per_class: int | None = Field(default=None, ge=1)
    focus: Literal["full", "eyes"] = "eyes"


def _run_image_training(job_id: str, body: ImageTrainBody) -> None:
    from .ml.image_training import train_image_model

    def progress(update: dict) -> None:
        with _image_training_lock:
            _image_training_jobs[job_id].update(update)

    try:
        result = train_image_model(
            IMAGE_DATASET_DIR,
            DATA_DIR / "models",
            epochs=body.epochs,
            batch_size=body.batch_size,
            image_size=body.image_size,
            max_images_per_class=body.max_images_per_class,
            focus=body.focus,
            progress=progress,
        )
        with SessionLocal() as db:
            run = ModelRun(
                algorithm=result["algorithm"],
                dataset="image_drowsiness",
                metrics_json=json.dumps(result),
                model_path=result["artifact"],
            )
            db.add(run)
            db.commit()
            result["run_id"] = run.id
        with _image_training_lock:
            _image_training_jobs[job_id].update({"status": "complete", "result": result})
    except Exception as exc:  # surfaced through the job endpoint and UI
        with _image_training_lock:
            _image_training_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/api/ml/image-model/train")
def image_model_train(body: ImageTrainBody):
    with _image_training_lock:
        if any(job["status"] in {"queued", "split", "training"} for job in _image_training_jobs.values()):
            raise HTTPException(409, "An image-model training job is already running.")
        job_id = uuid.uuid4().hex
        _image_training_jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued", "error": None}
    threading.Thread(target=_run_image_training, args=(job_id, body), daemon=True, name="drivio-image-training").start()
    return _image_training_jobs[job_id]


@app.get("/api/ml/image-model/jobs/{job_id}")
def image_model_job(job_id: str):
    with _image_training_lock:
        job = _image_training_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Image-model training job not found")
        return dict(job)


class TemporalTrainBody(BaseModel):
    window_size: int = Field(default=10, ge=3, le=120)
    stride: int = Field(default=5, ge=1)
    epochs: int = Field(default=10, ge=1, le=50)
    batch_size: int = Field(default=32, ge=1, le=512)


def _run_temporal_training(job_id: str, body: TemporalTrainBody) -> None:
    from .ml.temporal_training import build_sequences, train_temporal_model

    def progress(update: dict) -> None:
        with _temporal_training_lock:
            _temporal_training_jobs[job_id].update(update)

    try:
        with SessionLocal() as db:
            rows = db.scalars(select(Sample).where(Sample.ground_truth.is_not(None)).order_by(Sample.session_id, Sample.t)).all()
            records = build_sequences(rows, body.window_size, body.stride)
        output = DATA_DIR / "models" / "temporal_gru.pt"
        result = train_temporal_model(records, str(output), epochs=body.epochs, batch_size=body.batch_size, progress=progress)
        with SessionLocal() as db:
            run = ModelRun(algorithm=result["algorithm"], dataset="labelled_sessions",
                           metrics_json=json.dumps(result), model_path=result["artifact"])
            db.add(run)
            db.commit()
            result["run_id"] = run.id
        with _temporal_training_lock:
            _temporal_training_jobs[job_id].update({"status": "complete", "result": result})
    except Exception as exc:
        with _temporal_training_lock:
            _temporal_training_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/api/ml/temporal-model/train")
def temporal_model_train(body: TemporalTrainBody):
    with _temporal_training_lock:
        if any(job["status"] in {"queued", "training"} for job in _temporal_training_jobs.values()):
            raise HTTPException(409, "A temporal-model training job is already running.")
        job_id = uuid.uuid4().hex
        _temporal_training_jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued", "error": None}
    threading.Thread(target=_run_temporal_training, args=(job_id, body), daemon=True, name="drivio-temporal-training").start()
    return _temporal_training_jobs[job_id]


@app.get("/api/ml/temporal-model/jobs/{job_id}")
def temporal_model_job(job_id: str):
    with _temporal_training_lock:
        job = _temporal_training_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Temporal-model training job not found")
        return dict(job)


@app.post("/api/ml/train")
def ml_train(body: TrainBody, db: Session = Depends(get_db)):
    return _ml(experiments.train, db, body.dataset, body.algorithm)


@app.get("/api/ml/runs")
def ml_runs(db: Session = Depends(get_db)):
    return experiments.list_runs(db)


@app.get("/api/audit")
def audit_logs(db: Session = Depends(get_db), user: UserAccount = Depends(require_admin)):
    rows = db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(50)).all()
    return [{"id": row.id, "actor": row.actor, "action": row.action, "target": row.target,
             "detail": row.detail, "created_at": row.created_at.isoformat()} for row in rows]


@app.post("/api/ml/runs/{run_id}/approve")
def approve_model(run_id: int, db: Session = Depends(get_db), user: UserAccount = Depends(require_admin)):
    run = db.get(ModelRun, run_id)
    if not run:
        raise HTTPException(404, "Model run not found")
    from .ml.governance import deployment_decision

    metrics = json.loads(run.metrics_json)
    allowed, note = deployment_decision(metrics, bool(run.model_path) and Path(run.model_path).exists())
    if not allowed:
        raise HTTPException(409, note)
    for other in db.scalars(select(ModelRun).where(ModelRun.deployment_status == "approved")).all():
        other.deployment_status = "retired"
    run.deployment_status = "approved"
    run.approved_at = datetime.utcnow()
    run.approval_note = f"Approved by {user.username}: {note}"
    _audit(db, user.username, "model_approved", f"model_run:{run.id}", note)
    db.commit()
    return experiments.run_to_dict(run)


@app.get("/api/ml/compare")
def ml_compare(dataset: str = "", db: Session = Depends(get_db)):
    return experiments.compare(db, dataset or None)
