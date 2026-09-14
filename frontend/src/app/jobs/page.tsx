"use client";

// §38 任务列表: 展示当前用户所有 Conversion Job(可按项目过滤)。
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Job, Project } from "@/lib/types";
import { Badge, toneForStatus } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Spinner, Table, TBody, Td, Th, THead } from "@/components/ui/table";
import { formatTime } from "@/lib/utils";

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const [js, ps] = await Promise.all([
        api<Job[]>("/api/v1/jobs"),
        api<Project[]>("/api/v1/projects"),
      ]);
      setJobs(js);
      setProjects(ps);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const shown = filter ? jobs.filter((j) => j.project_id === filter) : jobs;
  const projectName = (id: string) =>
    projects.find((p) => p.id === id)?.name ?? id.slice(0, 8);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-neutral-900">Jobs</h1>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="h-10 rounded-md border border-neutral-300 bg-white px-3 text-sm"
        >
          <option value="">全部项目</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <Card>
        <CardHeader
          title="转换任务"
          description={`共 ${shown.length} 条`}
        />
        <CardContent className="p-0">
          {loading ? (
            <div className="flex justify-center py-12">
              <Spinner className="h-6 w-6" />
            </div>
          ) : (
            <Table>
              <THead>
                <Th>任务 ID</Th>
                <Th>项目</Th>
                <Th>状态</Th>
                <Th>进度</Th>
                <Th>Worker</Th>
                <Th>创建时间</Th>
                <Th>完成时间</Th>
              </THead>
              <TBody>
                {shown.length === 0 && (
                  <tr>
                    <Td colSpan={7} className="text-neutral-400">
                      暂无任务
                    </Td>
                  </tr>
                )}
                {shown.map((j) => (
                  <tr key={j.id} className="cursor-pointer hover:bg-neutral-50">
                    <Td>
                      <Link
                        href={`/jobs/${j.id}`}
                        className="font-medium text-neutral-900 hover:underline"
                      >
                        {j.id.slice(0, 12)}
                      </Link>
                    </Td>
                    <Td className="text-neutral-500">{projectName(j.project_id)}</Td>
                    <Td>
                      <Badge tone={toneForStatus(j.status)}>{j.status}</Badge>
                    </Td>
                    <Td className="w-36">
                      <Progress value={j.progress} className="h-1.5" />
                      <span className="text-xs text-neutral-400">{j.progress}%</span>
                    </Td>
                    <Td className="text-neutral-500">
                      {j.worker_id ? j.worker_id.slice(0, 8) : "-"}
                    </Td>
                    <Td>{formatTime(j.created_at ?? null)}</Td>
                    <Td>{formatTime(j.finished_at ?? null)}</Td>
                  </tr>
                ))}
              </TBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
