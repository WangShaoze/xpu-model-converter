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
                <Th>名称</Th>
                <Th>描述</Th>
                <Th>创建时间</Th>
                <Th>更新时间</Th>
              </THead>
              <TBody>
                {projects.length === 0 && (
                  <tr>
                    <Td colSpan={4} className="text-neutral-400">
                      暂无项目, 点击右上角创建
                    </Td>
                  </tr>
                )}
                {projects.map((p) => (
                  <tr key={p.id} className="cursor-pointer hover:bg-neutral-50">
                    <Td>
                      <Link
                        href={`/projects/${p.id}`}
                        className="font-medium text-neutral-900 hover:underline"
                      >
                        {p.name}
                      </Link>
                    </Td>
                    <Td className="max-w-64 truncate text-neutral-500">
                      {p.description || "-"}
                    </Td>
                    <Td>{formatTime(p.created_at)}</Td>
                    <Td>{formatTime(p.updated_at ?? null)}</Td>
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
