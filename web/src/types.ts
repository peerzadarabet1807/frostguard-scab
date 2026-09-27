// Mirrors src/api/schemas.py. Keep in sync when the API contract changes.

export type RiskLevel = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
export type FungicideAction = "none" | "monitor" | "protectant" | "curative";
export type WeatherSource = "open-meteo" | "mock" | "uploaded" | "replay";
export type VisionBackend = "trained" | "fallback";

export interface Health {
  status: "ok";
  version: string;
  vision_backend: VisionBackend;
  model: string;
  time: string;
}

export interface Zone {
  key: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation_m: number;
}

export interface Scenario {
  key: string;
  label: string;
  description: string;
}

export interface ReplayDataset {
  key: string;
  zone: string;
  title: string;
  source: string;
  start: string;
  end: string;
}

export interface TimelinePoint {
  time: string;
  temperature_c: number | null;
  relative_humidity: number | null;
  precipitation_mm: number | null;
  leaf_wet: boolean;
  wet_hours: number;
  risk_ratio: number;
  infection: boolean;
  frost_risk: boolean;
}

export interface FungicideWindow {
  action: FungicideAction;
  window_hours: number | null;
  deadline: string | null;
  message: string;
}

export interface FrostRisk {
  alert: boolean;
  frost_hours: number;
  min_temperature_c: number | null;
  first_frost_time: string | null;
  in_bud_break: boolean;
  threshold_c: number;
}

export interface InfectionEvent {
  event_id: number;
  start: string;
  end: string;
  wet_hours: number;
  mean_temperature_c: number;
  required_hours: number | null;
  risk_ratio: number;
  infected: boolean;
  infection_time: string | null;
}

export interface RiskResponse {
  location: { name: string | null; latitude: number | null; longitude: number | null };
  weather_source: WeatherSource;
  as_of: string;
  horizon_hours: number;
  infection_probability: number;
  current_risk: number;
  peak_risk: number;
  risk_level: RiskLevel;
  fungicide: FungicideWindow;
  frost: FrostRisk;
  alerts: string[];
  infection_events: InfectionEvent[];
  timeline: TimelinePoint[];
  warnings: string[];
}

export interface WeatherRecord {
  time: string;
  temperature_2m: number;
  relative_humidity_2m: number | null;
  precipitation: number | null;
  dew_point_2m: number | null;
}

export interface RiskRequest {
  weather?: WeatherRecord[];
  zone?: string;
  latitude?: number;
  longitude?: number;
  horizon_hours?: number;
  past_hours?: number;
  as_of?: string;
  bud_break?: boolean | null;
  use_mock?: boolean;
  mock_scenario?: string;
  replay?: string;
}

export interface LesionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  label: string;
}

export interface DetectionResponse {
  filename: string | null;
  scab_detected: boolean;
  lesion_count: number;
  max_confidence: number | null;
  detections: LesionBox[];
  image_width: number;
  image_height: number;
  inference_ms: number;
  backend: VisionBackend;
  model: string;
  warnings: string[];
}

export interface ModelCard {
  name: string;
  architecture: string;
  classes: string[];
  input_size: number;
  parameters: number | null;
  gflops: number | null;
  dataset: {
    name: string;
    url: string;
    license: string;
    origin: string;
    train_images: number | null;
    val_images: number | null;
  };
  training: { epochs: number; epochs_completed: number; best_epoch: number; hardware: string };
  metrics: { split: string; mAP50: number; mAP50_95: number; precision: number; recall: number };
  export: { format: string; file: string; size_bytes: number; sha256: string; release_tag: string; url: string };
  limitations: string[];
  license: string;
}

export interface ModelInfo {
  backend: VisionBackend;
  model: string;
  classes: string[];
  input_size: number;
  exported_by: string | null;
  exported_at: string | null;
  card: ModelCard | null;
}
