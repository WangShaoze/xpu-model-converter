"use client";

// §38 任务详情: 基本信息 / Stage 时间线 / SSE 实时事件 / Artifacts 下载。
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, downloadArtifact } from "@/lib/api";
import { useAuth } from "@/context/auth-context";
import type { Artifact, Job } from "@/lib/types";
import { Badge, toneForStatus } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Spinner, Table, TBody, Td, Th, THead } from "@/components/ui/table";
import { StageTimeline } from "@/components/job/stage-timeline";
import { LiveEvents } from "@/components/job/live-events";
import { formatBytes, formatTime } from "@/lib/utils";

const TERMINAL = new Set(["SUCCESS", "FAILED", "CANCELLED"]);

export default function JobDetailPage() {
  const params = useParams<{ id: string }>();
  const jobId = params.id;
  const { token } = useAuth();
  const router = useRouter();

  const [job, setJob] = useState<Job | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async () => {
    try {
      const [j, arts] = await Promise.all([
        api<Job>(`/api/v1/jobs/${jobId}`),
        api<Artifact[]>(`/api/v1/artifacts/job/${jobId}`).catch(() => []),
      ]);
      setJob(j);
      setArtifacts(arts);
      setError("");
      if (TERMINAL.has(j.status) && timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, [jobId]);

  useEffect(() => {
    void load();
    timerRef.current = setInterval(() => void load(), 3000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [load]);

  async function onRetry() {
    setBusy("retry");
    try {
      const j = await api<Job>(`/api/v1/jobs/${jobId}/retry`, { method: "POST" });
      setJob(j);
    } catch (err) {
      setError(err instanceof Error ? err.message : "重试失败");
    } finally {
      setBusy("");
    }
  }

  async function onCancel() {
    setBusy("cancel");
    try {
      const j = await api<Job>(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" });
      setJob(j);
    } catch (err) {
      setError(err instanceof Error ? err.message : "取消失败");
    } finally {
      setBusy("");
    }
  }

  async function onDelete() {
    if (!window.confirm("确认删除该任务及其记录?")) return;
    setBusy("delete");
    try {
      await api<void>(`/api/v1/jobs/${jobId}`, { method: "DELETE" });
      router.replace("/jobs");
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    } finally {
      setBusy("");
    }
  }

  async function onDownload(a: Artifact) {
    try {
      await downloadArtifact(a.id, a.filename, token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "下载失败");
    }
  }

  if (!job) {
    if (error) return <p className="text-sm text-red-600">加载失败: {error}</p>;
    return (
      <div className="flex justify-center py-16">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }

  const canCancel = job.status === "QUEUED" || job.status === "CREATED" || job.status === "RUNNING";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-neutral-900">
            任务 {job.id.slice(0, 12)}
          </h1>
          <p className="mt-1 text-xs text-neutral-500">
            项目 {job.project_id.slice(0, 8)} · 模型 {job.source_model_id.slice(0, 8)} · pipeline {job.pipeline_version}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {job.status === "FAILED" && (
            <Button size="sm" variant="outline" onClick={() => void onRetry()} disabled={busy === "retry"}>
              {busy === "retry" ? "重试中…" : "重试"}
            </Button>
          )}
          {canCancel && (
            <Button size="sm" variant="outline" onClick={() => void onCancel()} disabled={busy === "cancel"}>
              {busy === "cancel" ? "取消中…" : "取消"}
            </Button>
          )}
          <Button size="sm" variant="destructive" onClick={() => void onDelete()} disabled={busy === "delete"}>
            删除
          </Button>
        </div>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Stage 时间线"
            description={`整体进度 ${job.progress}%`}
            action={<Badge tone={toneForStatus(job.status)}>{job.status}</Badge>}
          />
          <CardContent>
            {job.error_message && (
              <p className="mb-3 rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">
                {job.error_message}
              </p>
            )}
            <StageTimeline stages={job.stages} />
            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-neutral-100 pt-4 text-xs text-neutral-500 sm:grid-cols-3">
              <div>
                <dt>创建时间</dt>
                <dd className="text-neutral-700">{formatTime(job.created_at ?? null)}</dd>
              </div>
              <div>
                <dt>排队时间</dt>
                <dd className="text-neutral-700">{formatTime(job.queued_at ?? null)}</dd>
              </div>
              <div>
                <dt>开始时间</dt>
                <dd className="text-neutral-700">{formatTime(job.started_at ?? null)}</dd>
              </div>
              <div>
                <dt>完成时间</dt>
                <dd className="text-neutral-700">{formatTime(job.finished_at ?? null)}</dd>
              </div>
              <div>
                <dt>Worker</dt>
                <dd className="text-neutral-700">{job.worker_id || "-"}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader title="实时事件" description="SSE 流(断线自动重连)" />
          <CardContent>
            <LiveEvents jobId={jobId} token={token} active />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader title="Artifacts" description={`共 ${artifacts.length} 个产物`} />
        <CardContent className="p-0">
          <Table>
            <THead>
              <Th>阶段</Th>
              <Th>类型</Th>
              <Th>文件名</Th>
              <Th>大小</Th>
              <Th>SHA256</Th>
              <Th>时间</Th>
              <Th className="text-right">操作</Th>
            </THead>
            <TBody>
              {artifacts.length === 0 && (
                <tr>
                  <Td colSpan={7} className="text-neutral-400">
                    暂无产物
                  </Td>
                </tr>
              )}
              {artifacts.map((a) => (
                <tr key={a.id}>
                  <Td className="text-neutral-500">{a.stage}</Td>
                  <Td>{a.artifact_type}</Td>
                  <Td className="max-w-52 truncate font-medium text-neutral-900">
                    {a.filename}
                  </Td>
                  <Td>{formatBytes(a.size)}</Td>
                  <Td className="font-mono text-xs text-neutral-500">
                    {a.sha256.slice(0, 10)}
                  </Td>
                  <Td>{formatTime(a.created_at ?? null)}</Td>
                  <Td className="text-right">
                    <Button size="sm" variant="outline" onClick={() => void onDownload(a)}>
                      下载
                    </Button>
                  </Td>
                </tr>
              ))}
            </TBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
