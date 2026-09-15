// 后端 API 类型(与 xpu_platform/api/schemas.py 对应)
export interface User {
  id: string;
  username: string;
  email: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at?: string | null;
}

export interface Model {
  id: string;
  project_id: string;
  name: string;
  filename: string;
  framework: string;
  model_type: string;
  status: string;
  sha256: string;
  size: number;
  created_at?: string | null;
}

export type JobStatus =
  | "CREATED"
  | "QUEUED"
  | "RUNNING"
  | "SUCCESS"
  | "FAILED"
  | "CANCELLED";

export type StageStatus =
  | "PENDING"
  | "RUNNING"
  | "SUCCESS"
  | "FAILED"
  | "SKIPPED"
  | "CANCELLED";

export interface Stage {
  name: string;
  status: StageStatus;
  progress: number;
  error_message: string;
}

export interface Job {
  id: string;
  project_id: string;
  source_model_id: string;
  status: JobStatus;
  pipeline_version: string;
  progress: number;
  worker_id: string;
  error_message: string;
  created_at?: string | null;
  queued_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  stages: Stage[];
}

export interface Artifact {
  id: string;
  job_id: string;
  stage: string;
  artifact_type: string;
  filename: string;
  storage_key: string;
  size: number;
  sha256: string;
  mime_type: string;
  created_at?: string | null;
}

export interface Worker {
  worker_id: string;
  hostname: string;
  device_type: string;
  chip: string;
  device_id: string;
  sdk_version: string;
  driver_version: string;
  status: string;
  last_heartbeat?: string | null;
}

export interface JobEvent {
  job_id: string;
  sequence: number;
  event: string;
  stage: string;
  message: string;
  progress: number | null;
  timestamp?: string | null;
  data: Record<string, unknown>;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  request_id: string;
}
