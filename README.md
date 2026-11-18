# DRIVIO

Real-time driver monitoring that detects fatigue, drowsiness and distraction from a camera.
All processing happens on your machine. Video is never written to disk.

```
Camera -> Face tracking -> Feature extraction -> Fatigue analysis -> Alertness score
       -> Driver classification -> Alerts and dashboard
```

## Run it

Needs Python 3.10-3.12 (MediaPipe has no wheels for newer versions) and Node 20.19+ or 22.12+.

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                                             # http://localhost:5173
```

No camera yet? Start the backend with `DRIVIO_SIMULATE=1` and a scripted driver replays
a 60 second cycle: alert, drowsy, micro-sleep, looking away, recovery. If the camera
cannot be opened, DRIVIO falls back to the simulator and says so on screen.

When an alert begins, the browser speaks **“Please be cautious!”**. Keep the browser
and system audio unmuted. Backend sound alerts are enabled by default; setting
`DRIVIO_AUDIO=0` disables those tones.

### Camera troubleshooting on Windows

Run the real-camera mode with `DRIVIO_SIMULATE=0`. Use Python 3.10-3.12 and install
`backend/requirements.txt`; if OpenCV is unavailable, Drivio intentionally uses the simulator.
Enable **Camera access** and **Allow desktop apps to access your camera** in Windows Privacy
settings, close Teams/Zoom/Camera and other webcam users, and try `DRIVIO_CAMERA=1` if the webcam
is not device index 0. Drivio tries the default, DirectShow and Media Foundation backends and
reports the selected backend or the startup error in the Live system-health panel.

| Variable | Default | Purpose |
|---|---|---|
| `DRIVIO_CAMERA` | `0` | Webcam index |
| `DRIVIO_SIMULATE` | `0` | Use the synthetic driver |
| `DRIVIO_AUDIO` | `1` | Play backend audio tones (Pygame or Windows beeper) |
| `DRIVIO_PHONE` | `0` | Phone detection with YOLO (`pip install ultralytics`) |
| `DRIVIO_DATA_DIR` | `backend/data` | Database, datasets and trained models |
| `DRIVIO_IMAGE_DATASET_DIR` | project root | Folder containing `Drowsy` and `Non Drowsy` image folders |
| `DRIVIO_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated frontend origins allowed by the API |

## How it decides

| Signal | Method | Where |
|---|---|---|
| Eye closure | Eye aspect ratio, calibrated per driver in the first 3 s | `vision/features.py`, `analysis/fatigue.py` |
| Blinks and long closures | Closure duration: 0.04-0.4 s is a blink, 1 s or more is prolonged, 2 s is a micro-sleep | `analysis/fatigue.py` |
| PERCLOS | Share of the last 60 s with eyes closed | `analysis/fatigue.py` |
| Yawns | Mouth aspect ratio held open for 1.2 s | `analysis/fatigue.py` |
| Head pose | `solvePnP` on six landmarks, relative to the calibrated baseline | `vision/features.py` |
| Phone | YOLO class "cell phone", optional | `vision/phone.py` |

**Alertness score.** Starts at 100. Each risk subtracts a penalty, and the dashboard lists every
penalty with the measurement behind it, so the score is always explainable (`analysis/scoring.py`).

**States.** `alert`, `fatigued`, `drowsy`, `distracted`, `no_face`, plus a confidence value that drops
when the driver is uncalibrated, recently appeared, or turned away from the camera.

**Alerts.** Caution, warning and critical. A level is raised only after its condition holds for a
moment (no flicker), and lowered only after 5 s of improvement, which logs a recovery event
(`analysis/alerts.py`). Sounds are synthesised, so there are no audio files to ship.

All thresholds are in `backend/app/config.py`.

## Analytics and reports

Press **Start recording** on the Live tab. Once per second the score, state and measurements are saved
to SQLite, along with every alert. The Analytics tab shows per-session trends, time in each state,
a score timeline with alert markers, and CSV and PDF downloads.

The Fleet tab aggregates driver profiles, recorded sessions, safety scores, fatigue exposure and
escalated alerts for the selected reporting period. It also shows the currently monitored driver
when a session is active. This is a local overview; authentication and role-based access are still
required before exposing fleet data beyond the local machine.

The Collection tab provides a safe annotation workflow for derived measurements. Start a stationary
or simulator session, assign labels as the participant demonstrates each condition, and export only
explicitly labelled feature rows for ML experiments. Raw video is still not stored.

Fleet access now has a local administrator login. On a fresh database, use **Create the first
administrator** once; later access requires the username and password. Passwords are stored as
PBKDF2 hashes and the browser stores only an expiring bearer token. The current authentication
slice protects the Fleet overview; Live and Analytics remain local-monitoring surfaces.

Fleet administrators can inspect privacy status and delete recorded sessions, measurements and
alerts from the Privacy controls panel. The destructive action is blocked while a session is active
and requires an explicit confirmation in the UI.
Administrative registration, model approvals and recorded-data deletion are written to a local
audit log visible in the Fleet overview.

`GET /api/health` reports both liveness and readiness: database connectivity, pipeline startup,
video-storage policy and current pipeline health are returned separately.
Training endpoints validate hyperparameter ranges at the request boundary, so invalid jobs are
rejected before a background worker is created.

## ML experiments

The Experiments tab manages datasets (generate a demo set, export your recorded sessions, or upload a
CSV), trains random forest, gradient boosting, logistic regression or a PyTorch network, and compares
runs by accuracy and macro F1 with a confusion matrix. For tree and logistic models it also shows
normalized global feature importance. Models are saved under `backend/data/models`.
The rule-based classifier is what drives alerts today. A trained model is a candidate to replace it
once you have labelled data from real drives.

### Image dataset checkpoint

The Experiments tab also audits a local image dataset stored as:

```
<dataset root>/Drowsy/*.png
<dataset root>/Non Drowsy/*.png
```

The audit counts all image files, samples up to 200 images per class for decoding and dimensions,
and reports filename groups that can be used to create a leakage-resistant train/test split. Set
`DRIVIO_IMAGE_DATASET_DIR` when the dataset is outside the project root. The audit is intentionally
read-only; it never copies or stores the source images.

When the audit is ready, **Train image model** starts a background PyTorch CNN job. The job keeps
filename groups together across train, validation and test sets, saves `drowsiness_cnn.pt` under
`backend/data/models`, and records accuracy, macro precision/recall/F1 and a confusion matrix in
the existing model-run history. This model is evaluated separately from the current rule-based
live monitor until the hybrid inference stage is implemented.

The training result includes a deployment-quality gate based on unseen-group macro F1 and drowsy
recall. A model that fails the gate remains available for analysis but is not eligible to drive
live alerts.

Every trained model is tracked as a `candidate` in the Experiments page. An administrator may
approve a candidate only when its quality gate passes and its saved artifact exists; approving a
new candidate retires the previous approved candidate. Approval is a governance record for future
inference work and does not replace the current rule-based live monitor automatically.

## Tests

```bash
cd backend && pytest                       # geometry, fatigue logic, scoring, alerts, API
```

The two HTTP integration tests are retained for the supported Python 3.10-3.12
runtime. On Windows with Python 3.14 they are skipped because the local
AnyIO/asyncio harness can hang before FastAPI receives a request; the focused
feature tests still run and must remain green.

```bash
cd e2e && npm install && npx playwright install chromium && npm test
```

## Calibration and safety notes

- EAR, MAR and head angles vary with camera position and face shape. Sit normally for the first few seconds
  after starting (or press **Recalibrate**), and check the thresholds in `config.py` against your own setup.
- Glasses, sunglasses and strong backlight reduce landmark quality. Confidence drops accordingly.
- DRIVIO is a research and learning project. It is not a certified safety device and should not be
  relied on to keep a vehicle's driver awake.
