"""CSV and PDF session reports."""
from __future__ import annotations

import csv
import io
from collections import Counter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import AlertRecord, DriveSession, DriverProfile, Sample

CSV_COLUMNS = ["t", "score", "safety_score", "state", "ear", "mar", "perclos", "closure_s", "blink_rate",
               "yawns_recent", "yaw", "pitch", "head_turned_s"]


def session_csv(db: Session, session_id: int) -> str:
    rows = db.scalars(select(Sample).where(Sample.session_id == session_id).order_by(Sample.t)).all()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    for r in rows:
        writer.writerow([r.t.isoformat(), r.score, r.safety_score, r.state, r.ear, r.mar, r.perclos, r.closure_s,
                         r.blink_rate, r.yawns_recent, r.yaw, r.pitch, r.head_turned_s])
    return out.getvalue()


def session_pdf(db: Session, session_id: int) -> bytes:
    sess = db.get(DriveSession, session_id)
    driver = db.get(DriverProfile, sess.driver_id) if sess and sess.driver_id else None
    samples = db.scalars(select(Sample).where(Sample.session_id == session_id).order_by(Sample.t)).all()
    alerts = db.scalars(select(AlertRecord).where(AlertRecord.session_id == session_id).order_by(AlertRecord.t)).all()
    styles = getSampleStyleSheet()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"DRIVIO session {session_id}")
    driver_text = f" - driver {driver.name}" if driver else " - driver unassigned"
    story = [Paragraph("DRIVIO session report", styles["Title"]),
             Paragraph(f"{sess.label} - started {sess.started_at:%Y-%m-%d %H:%M} UTC ({sess.mode}){driver_text}", styles["Normal"]),
             Spacer(1, 12)]

    scores = [s.score for s in samples]
    states = Counter(s.state for s in samples)
    summary = [["Samples", len(samples)],
               ["Average alertness", f"{sum(scores) / len(scores):.0f}" if scores else "-"],
               ["Lowest alertness", min(scores) if scores else "-"],
               ["Lowest trip safety", min((s.safety_score for s in samples), default="-")],
               ["Alerts raised", sum(1 for a in alerts if a.kind == "escalation")]]
    summary += [[f"Time {k}", f"{v / len(samples) * 100:.0f}%"] for k, v in states.most_common()]
    story += [Table(summary, colWidths=[200, 120], style=TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)])),
              Spacer(1, 16), Paragraph("Event log", styles["Heading2"])]

    rows = [["Time", "Level", "Type", "Message"]] + [
        [a.t.strftime("%H:%M:%S"), a.level, a.kind, Paragraph(a.message, styles["BodyText"])] for a in alerts[:200]]
    if len(rows) == 1:
        rows.append(["-", "-", "-", "No alerts in this session"])
    story.append(Table(rows, colWidths=[60, 40, 70, 300], repeatRows=1, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP")])))
    doc.build(story)
    return buf.getvalue()
