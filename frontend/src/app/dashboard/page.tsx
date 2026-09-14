"use client";

// §36 Dashboard: 全局统计(Projects/Models/Jobs/Workers) + 最近任务。
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Job, Model, Project, Worker } from "@/lib/types";
import { Badge, toneForStatus } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Spinner, Table, TBody, Td, Th, THead } from "@/components/ui/table";
import { formatTime } from "@/lib/utils";

interface Stats {
  projects: number;
  models: number;
  queued: number;
  running: number;
  success: number;
  failed: number;
  workers: number;
}

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: number | string;
  hint?: string;
}) {
  return (
    <Card>
      <CardContent className="px-5 py-4">
        <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">
          {label}
        </p>
        <p className="mt-2 text-2xl font-semibold text-neutral-900">{value}</p>
        {hint && <p className="mt-1 text-xs text-neutral-400">{hint}</p>}
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [recent, setRecent] = useState<Job[]>([]);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const [projects, jobs, workers] = await Promise.all([
        api<Project[]>("/api/v1/projects"),
        api<Job[]>("/api/v1/jobs"),
        api<Worker[]>("/api/v1/workers"),
      ]);
      let models = 0;
      for (const p of projects) {
        const ms = await api<Model[]>(`/api/v1/projects/${p.id}/models`);
        models += ms.length;
      }
      const s: Stats = {
        projects: projects.length,
        models,
        queued: 0,
        running: 0,
        success: 0,
        failed: 0,
        workers: workers.length,
      };
      for (const j of jobs) {
        if (j.status === "QUEUED" || j.status === "CREATED") s.queued++;
        else if (j.status === "RUNNING") s.running++;
        else if (j.status === "SUCCESS") s.success++;
        else if (j.status === "FAILED") s.failed++;
      }
      setStats(s);
      setRecent(jobs.slice(0, 8));
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (error) {
    return <p className="text-sm text-red-600">加载失败: {error}</p>;
  }
  if (!stats) {
    return (
      <div className="flex justify-center py-16">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-neutral-900">Dashboard</h1>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Projects" value={stats.projects} />
        <StatCard label="Models" value={stats.models} />
        <StatCard label="Running" value={stats.running} />
        <StatCard label="Queued" value={stats.queued} />
        <StatCard label="Success" value={stats.success} />
        <StatCard label="Failed" value={stats.failed} />
      </div>

      <Card>
        <CardHeader
          title="XPU Workers"
          description="当前声明的可用 XPU Worker"
        />
        <CardContent className="flex items-center gap-2 text-sm text-neutral-700">
          <span className="text-2xl font-semibold text-neutral-900">
            {stats.workers}
          </span>
          台在线
          <Link href="/jobs" className="ml-auto text-xs text-neutral-500 hover:underline">
            查看任务 →
          </Link>
        </CardContent>
      </Card>

      <Card>
        <CardHeader
          title="最近任务"
          action={
            <Link
              href="/jobs"
              className="text-xs font-medium text-neutral-500 hover:underline"
            >
              全部 →
            </Link>
          }
        />
        <CardContent>
          <Table>
            <THead>
              <Th>任务</Th>
              <Th>状态</Th>
              <Th>进度</Th>
              <Th>创建时间</Th>
            </THead>
            <TBody>
              {recent.length === 0 && (
                <tr>
                  <Td colSpan={4} className="text-neutral-400">
                    暂无任务
                  </Td>
                </tr>
              )}
              {recent.map((j) => (
                <tr key={j.id}>
                  <Td>
                    <Link
                      href={`/jobs/${j.id}`}
                      className="font-medium text-neutral-900 hover:underline"
                    >
                      {j.id.slice(0, 12)}
                    </Link>
                  </Td>
                  <Td>
                    <Badge tone={toneForStatus(j.status)}>{j.status}</Badge>
                  </Td>
                  <Td>{j.progress}%</Td>
                  <Td>{formatTime(j.created_at ?? null)}</Td>
                </tr>
              ))}
            </TBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
