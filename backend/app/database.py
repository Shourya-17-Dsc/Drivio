"""SQLite persistence via SQLAlchemy. Stores metrics and events, never video."""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATA_DIR, settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class UserAccount(Base):
    __tablename__ = "user_accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuthToken(Base):
    __tablename__ = "auth_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(80))
    target: Mapped[str] = mapped_column(String(120), default="")
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DriverProfile(Base):
    __tablename__ = "driver_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    calibration_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DriveSession(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(120), default="Drive")
    mode: Mapped[str] = mapped_column(String(20), default="camera")
    driver_id: Mapped[Optional[int]] = mapped_column(ForeignKey("driver_profiles.id"), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Sample(Base):
    __tablename__ = "samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    t: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    score: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(20))
    ear: Mapped[float] = mapped_column(Float, default=0.0)
    mar: Mapped[float] = mapped_column(Float, default=0.0)
    perclos: Mapped[float] = mapped_column(Float, default=0.0)
    closure_s: Mapped[float] = mapped_column(Float, default=0.0)
    blink_rate: Mapped[float] = mapped_column(Float, default=0.0)
    yawns_recent: Mapped[int] = mapped_column(Integer, default=0)
    yaw: Mapped[float] = mapped_column(Float, default=0.0)
    pitch: Mapped[float] = mapped_column(Float, default=0.0)
    head_turned_s: Mapped[float] = mapped_column(Float, default=0.0)
    safety_score: Mapped[int] = mapped_column(Integer, default=100)
    ground_truth: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)


class AlertRecord(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    t: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    level: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))
    state: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    algorithm: Mapped[str] = mapped_column(String(40))
    dataset: Mapped[str] = mapped_column(String(120))
    metrics_json: Mapped[str] = mapped_column(Text)
    model_path: Mapped[str] = mapped_column(String(255))
    deployment_status: Mapped[str] = mapped_column(String(20), default="candidate")
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    approval_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    # create_all does not alter an existing SQLite table. Keep existing
    # installations usable when the driver-profile feature is introduced.
    columns = {column["name"] for column in inspect(engine).get_columns("sessions")}
    if "driver_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE sessions ADD COLUMN driver_id INTEGER"))
    sample_columns = {column["name"] for column in inspect(engine).get_columns("samples")}
    if "safety_score" not in sample_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE samples ADD COLUMN safety_score INTEGER DEFAULT 100"))
    if "ground_truth" not in sample_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE samples ADD COLUMN ground_truth VARCHAR(20)"))
    model_columns = {column["name"] for column in inspect(engine).get_columns("model_runs")}
    if "deployment_status" not in model_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE model_runs ADD COLUMN deployment_status VARCHAR(20) DEFAULT 'candidate'"))
    if "approved_at" not in model_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE model_runs ADD COLUMN approved_at DATETIME"))
    if "approval_note" not in model_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE model_runs ADD COLUMN approval_note TEXT"))
