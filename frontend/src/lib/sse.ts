// SSE 客户端: 用 fetch 流式读取(可带 Authorization 头), 支持 Last-Event-ID 断线恢复。
import { API_BASE } from "./api";
import type { JobEvent } from "./types";

export interface SseHandlers {
  onEvent: (ev: JobEvent) => void;
  onStatus?: (connected: boolean) => void;
  onError?: (err: Error) => void;
}

export class JobEventStream {
  private controller: AbortController | null = null;
  private closed = false;
  private lastEventId = 0;

  constructor(
    private jobId: string,
    private token: string | null,
    private handlers: SseHandlers,
  ) {}

  start(): void {
    void this.connect();
  }

  stop(): void {
    this.closed = true;
    this.controller?.abort();
    this.controller = null;
  }

  private async connect(): Promise<void> {
    if (this.closed) return;
    this.controller = new AbortController();
    this.handlers.onStatus?.(true);
    const headers: Record<string, string> = { Accept: "text/event-stream" };
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/jobs/${this.jobId}/events?last_event_id=${this.lastEventId}`,
        { headers, signal: this.controller.signal },
      );
      if (!res.ok || !res.body) {
        throw new Error(`SSE 连接失败: HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const block = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          this.parseBlock(block);
        }
      }
    } catch (err) {
      if (this.closed) return;
      this.handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
    } finally {
      this.controller = null;
      this.handlers.onStatus?.(false);
      // 流意外中断(非主动关闭)时自动重连, 携带 Last-Event-ID 恢复
      if (!this.closed) {
        setTimeout(() => this.connect(), 1500);
      }
    }
  }

  private parseBlock(block: string): void {
    let eventName = "message";
    let id: number | null = null;
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("id:")) id = Number.parseInt(line.slice(3).trim(), 10);
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) return;
    try {
      const parsed = JSON.parse(data) as Partial<JobEvent> & Record<string, unknown>;
      const ev: JobEvent = {
        job_id: String(parsed.job_id ?? this.jobId),
        sequence: id ?? Number(parsed.sequence ?? 0),
        event: parsed.event ?? eventName,
        stage: String(parsed.stage ?? ""),
        message: String(parsed.message ?? ""),
        progress: parsed.progress ?? null,
        data: (parsed.data as Record<string, unknown>) ?? {},
      };
      if (ev.sequence > this.lastEventId) this.lastEventId = ev.sequence;
      this.handlers.onEvent(ev);
    } catch {
      // 忽略无法解析的事件帧
    }
  }
}
