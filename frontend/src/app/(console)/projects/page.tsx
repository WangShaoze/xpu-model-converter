"use client";

// §37 项目列表: 展示 + 创建 Project。
import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Project } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Spinner, Table, TBody, Td, Th, THead } from "@/components/ui/table";
import { formatTime } from "@/lib/utils";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setProjects(await api<Project[]>("/api/v1/projects"));
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

  async function onCreate(e: { preventDefault: () => void }) {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api<Project>("/api/v1/projects", {
        method: "POST",
        body: { name: name.trim(), description: description.trim() },
      });
      setName("");
      setDescription("");
      setCreating(false);
      void load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-neutral-900">项目列表</h1>
        <Button size="sm" onClick={() => setCreating((v) => !v)}>
          {creating ? "取消" : "新建项目"}
        </Button>
      </div>

      {creating && (
        <Card>
          <CardHeader title="新建项目" />
          <CardContent>
            <form onSubmit={onCreate} className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="flex-1">
                <label className="mb-1 block text-xs text-neutral-500">名称</label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="例如: yolov10 检测项目"
                  required
                />
              </div>
              <div className="flex-[2]">
                <label className="mb-1 block text-xs text-neutral-500">描述(可选)</label>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="模型用途 / 备注"
                />
              </div>
              <Button type="submit" disabled={busy || !name.trim()}>
                {busy ? "创建中…" : "创建"}
              </Button>
            </form>
          </CardContent>
        </Card>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}

      <Card>
        <CardContent className="p-0">
          {loading ? (
            <div className="flex justify-center py-12">
              <Spinner className="h-6 w-6" />
            </div>
          ) : (
            <Table>
              <THead>
                <Th>项目</Th>
                <Th>创建时间</Th>
              </THead>
              <TBody>
                {projects.length === 0 && (
                  <tr>
                    <Td colSpan={2} className="py-8 text-center text-neutral-400">
                      暂无项目，点击右上角「新建项目」开始
                    </Td>
                  </tr>
                )}
                {projects.map((p) => (
                  <Link
                    key={p.id}
                    href={`/projects/${p.id}`}
                    className="block border-b border-neutral-100 px-5 py-3.5 transition-colors hover:bg-neutral-50"
                  >
                    <div className="flex items-center justify-between">
                      <div className="min-w-0">
                        <p className="font-medium text-neutral-900">{p.name}</p>
                        <p className="mt-0.5 max-w-64 truncate text-xs text-neutral-400">
                          {p.description || "无描述"}
                        </p>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-neutral-400">
                        <span>{formatTime(p.created_at)}</span>
                        <svg className="h-4 w-4 text-neutral-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                  </Link>
                ))}
              </TBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
