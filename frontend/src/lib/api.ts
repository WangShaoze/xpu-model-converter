// API 客户端: 统一 BASE URL / Token 注入 / 错误归一化(§42 错误格式)。
import type { ApiErrorBody, TokenResponse, User } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const TOKEN_KEY = "xpu_token";
const USER_KEY = "xpu_user";

export class ApiError extends Error {
  code: string;
  status: number;
  requestId: string;

  constructor(status: number, body: Partial<ApiErrorBody>) {
    super(body.message ?? "请求失败");
    this.name = "ApiError";
    this.code = body.code ?? "REQUEST_ERROR";
    this.status = status;
    this.requestId = body.request_id ?? "";
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getUser(): User | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as User;
  } catch {
    return null;
  }
}

export function setAuth(token: string, user: User): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  token?: string | null;
  signal?: AbortSignal;
}

async function parseError(res: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> = {};
  try {
    body = (await res.json()) as Partial<ApiErrorBody>;
  } catch {
    body = {};
  }
  return new ApiError(res.status, body);
}

/** 通用 JSON 请求; 401 时清理登录态并跳转 /login。 */
export async function api<T>(
  path: string,
  opts: RequestOptions = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  const token = opts.token !== undefined ? opts.token : getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let body: BodyInit | undefined;
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  const res = await fetch(`${API_BASE}${path}`, {
    method: opts.method ?? "GET",
    headers,
    body,
    signal: opts.signal,
  });
  if (!res.ok) {
    const err = await parseError(res);
    if (err.status === 401 && !path.startsWith("/auth/")) {
      clearAuth();
      if (typeof window !== "undefined") window.location.href = "/login";
    }
    throw err;
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** 带进度的 multipart 上传(§37 上传进度), 使用 XHR 获取 progress 事件。 */
export function uploadModel(
  projectId: string,
  file: File,
  onProgress: (percent: number) => void,
  token?: string | null,
): Promise<{ id: string; name: string; sha256: string; size: number; status: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", `${API_BASE}/api/v1/projects/${projectId}/models`);
    const tk = token !== undefined ? token : getToken();
    if (tk) xhr.setRequestHeader("Authorization", `Bearer ${tk}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let body: Partial<ApiErrorBody> = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          body = {};
        }
        reject(new ApiError(xhr.status, body));
      }
    };
    xhr.onerror = () => reject(new Error("网络错误, 上传失败"));
    xhr.send(form);
  });
}

/** 带鉴权下载 Artifact(后端 download 需 Bearer), 触发浏览器保存。 */
export async function downloadArtifact(
  artifactId: string,
  filename: string,
  token?: string | null,
): Promise<void> {
  const tk = token !== undefined ? token : getToken();
  const headers: Record<string, string> = {};
  if (tk) headers.Authorization = `Bearer ${tk}`;
  const res = await fetch(`${API_BASE}/api/v1/artifacts/${artifactId}/download`, {
    headers,
  });
  if (!res.ok) {
    const err = await parseError(res);
    throw err;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** 注册/登录共用封装。 */
export async function authenticate(
  path: "/auth/login" | "/auth/register",
  payload: Record<string, string>,
): Promise<TokenResponse> {
  const data = await api<TokenResponse>(path, { method: "POST", body: payload, token: null });
  setAuth(data.access_token, data.user);
  return data;
}
