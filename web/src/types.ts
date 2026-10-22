// Mirrors the FastAPI OpenAPI schema (api/main.py). Update by hand if the API changes.
export type Grade = 0 | 1 | 2 | 3 | 4;

export interface Health {
  status: string;
  device: string;
  assets_dir: string;
  assets: Record<string, boolean>;
  bank: number;
  loaded: { G: boolean; classifiers: boolean };
}

export interface GenerateRequest {
  grade: Grade;
  seed: number;
  psi: number;
  nearest?: boolean;
}

export interface NearestReal {
  image_id: string;
  distance: number;
  thumb_png_b64: string | null;
}

export interface GenerateResponse {
  grade: Grade;
  seed: number;
  psi: number;
  synthetic: true;
  image_png_b64: string;
  nearest_real: NearestReal | null;
  seconds: number;
}

export type ModelKey = "baseline" | "gan";

export interface ClassifyResponse {
  probs: Record<ModelKey, number[]>;
  pred: Record<ModelKey, Grade>;
  gradcam_png_b64: Record<ModelKey, string>;
  preprocessed_png_b64: string;
  grade_names: string[];
  seconds: number;
}

export interface BankResponse {
  grade: Grade | null;
  files: string[];
  base_url: string;
}

export interface ResultsTables {
  generated_at: string;
  n_rows: number;
  n_runs: number;
  metrics: string[];
  condition_labels: Record<string, string>;
  summary_long: Record<string, number | string>[];
  main: Record<string, Record<string, Record<string, Record<string, string>>>>;
  delta_6_vs_5: Record<string, Record<string, number>[]>;
  ratio: Record<string, Record<string, number | string>[]>;
  gan_metrics?: Record<string, number | string>[];
}
