"use client";

// §37 项目详情: 模型上传(带进度) / 模型列表 / 创建 Conversion Job。
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, uploadModel } from "@/lib/api";
import type { Job, Model, Project } from "@/lib/types";
import { Badge, toneForStatus } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Spinner, Table, TBody, Td, Th, THead } from "@/components/ui/table";
import { formatBytes, formatTime } from "@/lib/utils";

const TASKS = ["detection", "seg", "pose", "cls", "obb", "depth", "sem"];
const PRECISIONS = ["fp16", "fp32", "int8"];

export default function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;

  const [project, setProject] = useState<Project | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // 上传状态
  const [uploading, setUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  // 创建 job 配置
  const [jobFor, setJobFor] = useState<string | null>(null);
  const [task, setTask] = useState("detection");
  const [precision, setPrecision] = useState("fp16");
  const [inputSize, setInputSize] = useState("640,640");
  const [jobBusy, setJobBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [p, ms] = await Promise.all([
        api<Project>(`/api/v1/projects/${projectId}`),
        api<Model[]>(`/api/v1/projects/${projectId}/models`),
      ]);
      setProject(p);
      setModels(ms);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  function onPickFile() {
    fileRef.current?.click();
  }

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploading(true);
    setUploadPercent(0);
    try {
      await uploadModel(projectId, file, setUploadPercent);
      void load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  }

  async function onCreateJob(modelId: string) {
    setJobBusy(true);
    try {
      const [w, h] = inputSize.split(",").map((s) => Number(s.trim()));
      const job = await api<Job>("/api/v1/jobs", {
        method: "POST",
        body: {
          project_id: projectId,
          model_id: modelId,
          config: {
            task,
            precision,
            input_size: [w || 640, h || 640],
            model_type: models.find((m) => m.id === modelId)?.model_type ?? "yolov10",
          },
        },
      });
      setJobFor(null);
      window.location.href = `/jobs/${job.id}`;
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建任务失败");
    } finally {
      setJobBusy(false);
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }
  if (!project) {
    return <p className="text-sm text-red-600">项目不存在: {error}</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <Link href="/projects" className="text-xs text-neutral-500 hover:underline">
          ← 返回项目列表
        </Link>
        <div className="mt-1 flex items-center gap-3">
          <h1 className="text-xl font-semibold text-neutral-900">{project.name}</h1>
          <Badge>{project.id.slice(0, 8)}</Badge>
        </div>
        {project.description && (
          <p className="mt-1 text-sm text-neutral-500">{project.description}</p>
        )}
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <Card>
        <CardHeader
          title="模型上传"
          description="支持 .pt / .pth / .onnx 权重文件, 上传完成后在 Worker 中隔离处理"
          action={
            <Button size="sm" onClick={onPickFile} disabled={uploading}>
              {uploading ? `上传中 ${uploadPercent}%` : "选择文件"}
            </Button>
          }
        />
        <CardContent>
          <input ref={fileRef} type="file" className="hidden" onChange={onUpload} />
          {uploading && <Progress value={uploadPercent} className="h-2" />}
          {!uploading && (
            <p className="text-xs text-neutral-400">
              大文件上传使用分块进度显示; 后端将计算 SHA256 并校验完整性。
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="模型列表" description={`共 ${models.length} 个模型`} />
        <CardContent className="p-0">
          <Table>
            <THead>
              <Th>名称</Th>
              <Th>框架</Th>
              <Th>类型</Th>
              <Th>大小</Th>
              <Th>状态</Th>
              <Th>上传时间</Th>
              <Th className="text-right">操作</Th>
            </THead>
            <TBody>
              {models.length === 0 && (
                <tr>
                  <Td colSpan={7} className="text-neutral-400">
                    暂无模型, 请上传
                  </Td>
                </tr>
              )}
              {models.map((m) => (
                <tr key={m.id}>
                  <Td className="font-medium text-neutral-900">{m.name}</Td>
                  <Td>{m.framework}</Td>
                  <Td>{m.model_type}</Td>
                  <Td>{formatBytes(m.size)}</Td>
                  <Td>
                    <Badge tone={toneForStatus(m.status)}>{m.status}</Badge>
                  </Td>
                  <Td>{formatTime(m.created_at ?? null)}</Td>
                  <Td className="text-right">
                    {m.status === "READY" ? (
                      <Button size="sm" variant="outline" onClick={() => setJobFor(m.id)}>
                        创建转换任务
                      </Button>
                    ) : (
                      <span className="text-xs text-neutral-400">未就绪</span>
                    )}
                  </Td>
                </tr>
              ))}
            </TBody>
          </Table>

          {jobFor && (
            <div className="border-t border-neutral-100 bg-neutral-50 px-5 py-4">
              <p className="mb-3 text-sm font-medium text-neutral-800">
                配置转换任务 (模型: {models.find((m) => m.id === jobFor)?.name})
              </p>
              <div className="flex flex-wrap items-end gap-3">
                <div>
                  <label className="mb-1 block text-xs text-neutral-500">任务类型</label>
                  <select
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    className="h-10 rounded-md border border-neutral-300 bg-white px-3 text-sm"
                  >
                    {TASKS.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-neutral-500">精度</label>
                  <select
                    value={precision}
                    onChange={(e) => setPrecision(e.target.value)}
                    className="h-10 rounded-md border border-neutral-300 bg-white px-3 text-sm"
                  >
                    {PRECISIONS.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-neutral-500">输入尺寸 W,H</label>
                  <Input
                    value={inputSize}
                    onChange={(e) => setInputSize(e.target.value)}
                    className="w-28"
                  />
                </div>
                <Button onClick={() => void onCreateJob(jobFor)} disabled={jobBusy}>
                  {jobBusy ? "提交中…" : "提交任务"}
                </Button>
                <Button variant="ghost" size="md" onClick={() => setJobFor(null)}>
                  取消
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
