export type DriverState = "alert" | "fatigued" | "drowsy" | "distracted" | "no_face";

export interface Factor { name: string; penalty: number; detail: string; kind: string }
export interface AlertEvent { ts: number; level: number; level_name: string; kind: "escalation" | "recovery"; state: DriverState; message: string }

export interface Metrics {
  ear: number; ear_threshold: number; mar: number; yaw: number; pitch: number;
  eyes_closed: boolean; closure_s: number; blinks_total: number; blink_rate_pm: number;
  perclos: number; yawning: boolean; yawns_recent: number; head_turned_s: number;
  face_present: boolean; calibrated: boolean; observed_s: number;
  gaze: number; gaze_off_s: number;
}

export interface Health {
  camera_ok: boolean; vision_ok: boolean; audio_ok: boolean; phone_detection: boolean;
  fps: number; latency_ms: number; uptime_s: number; error: string | null; camera_backend?: string;
}

export interface LiveState {
  ts: number; mode: "camera" | "simulation"; recording: boolean; session_id: number | null;
  driver_id: number | null; driver_name?: string | null;
  ground_truth: string | null;
  score: number; state: DriverState; confidence: number; factors: Factor[]; metrics: Metrics;
  level: number; level_name: string; events: AlertEvent[]; health: Health; safety: SafetyState;
}
export interface SafetyState {
  active: boolean; score: number; drive_duration_s: number; fatigue_duration_s: number;
  distraction_duration_s: number; critical_alerts: number; breaks_taken: number;
  break_due: boolean; recommendation: string;
}

export interface SessionSummary {
  id: number; label: string; mode: string; started_at: string; ended_at: string | null;
  driver_id: number | null; driver_name: string | null;
  duration_s: number; samples: number; avg_score: number | null; min_score: number | null; min_safety_score: number | null;
  fatigue_pct: number; alerts: number;
}
export interface DriverProfile {
  id: number; name: string; calibration_available: boolean; created_at: string; updated_at: string;
}
export interface AuthUser { id: number; username: string; role: string }
export interface AuthResponse { user: AuthUser; token: string }
export interface SessionDetail {
  summary: SessionSummary;
  samples: { s: number; score: number; safety_score: number; state: DriverState; perclos: number; ear: number; ground_truth: string | null }[];
  events: { s: number; time: string; level: number; kind: string; state: DriverState; message: string }[];
}

export interface Dataset { name: string; rows: number; classes: Record<string, number> }
export interface ModelRun {
  id: number; created_at: string; algorithm: string; dataset: string; accuracy: number; f1_macro: number;
  labels: string[]; confusion_matrix: number[][]; train_rows: number; test_rows: number;
  deployment_status: "candidate" | "approved" | "retired"; deployment_ready: boolean; quality_note: string;
  feature_importance?: { feature: string; importance: number }[];
}

export interface ImageDatasetClassAudit {
  files: number; readable: number; corrupt: number; ignored: number;
  dimensions: Record<string, number>; groups: Record<string, number>;
}
export interface ImageDatasetAudit {
  root: string; available: boolean; ready_for_training: boolean;
  classes: Record<string, ImageDatasetClassAudit>; total_files: number;
  total_readable: number; total_corrupt: number; class_balance_ratio: number;
  dimensions: Record<string, number>; group_count: number;
  group_prefixes: Record<string, number>; issues: string[];
}
export interface ImageTrainingResult {
  run_id?: number; algorithm: string; artifact: string; accuracy: number;
  precision_macro: number; recall_macro: number; f1_macro: number;
  precision_by_class: number[]; recall_by_class: number[]; f1_by_class: number[]; support_by_class: number[];
  labels: string[]; confusion_matrix: number[][]; train_rows: number;
  validation_rows: number; test_rows: number; groups: number; epochs: number;
  image_size: number; device: string;
  focus: "full" | "eyes";
  generalization_gap: number;
  test_group_metrics: Record<string, { rows: number; accuracy: number; f1_macro: number; drowsy_recall: number }>;
  drowsy_recall: number; deployment_ready: boolean; quality_note: string;
}
export interface ImageTrainingJob {
  id: string; status: "queued" | "split" | "training" | "complete" | "failed";
  stage: string; epoch?: number; loss?: number; accuracy?: number;
  error: string | null; train_rows?: number; validation_rows?: number;
  test_rows?: number; result?: ImageTrainingResult;
}
export interface TemporalTrainingResult {
  algorithm: string; artifact: string; accuracy: number; precision_macro: number;
  recall_macro: number; f1_macro: number; labels: string[]; confusion_matrix: number[][];
  precision_by_class: number[]; recall_by_class: number[]; f1_by_class: number[]; support_by_class: number[];
  train_rows: number; validation_rows: number; test_rows: number; groups: number;
  deployment_ready: boolean; quality_note: string; best_validation_f1: number;
}
export interface TemporalTrainingJob {
  id: string; status: "queued" | "training" | "complete" | "failed"; stage: string;
  epoch?: number; accuracy?: number; error: string | null; result?: TemporalTrainingResult;
}
export interface TemporalDatasetStatus {
  labelled_samples: number; labelled_sessions: number; labels: Record<string, number>;
  sessions_by_label: Record<string, number>; minimum_sessions_required: number;
  ready_for_temporal_training: boolean; reason: string;
}
