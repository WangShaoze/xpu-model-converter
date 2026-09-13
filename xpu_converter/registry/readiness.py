# -*- coding: utf-8 -*-
"""模型支持状态自动检查(ChatGPT 修改意见 P0-7)。

"代码里有 adapter" 不等于 "真支持"。这里把"人工声明生命周期状态"与"实际是否
就绪"自动对账: 对每个模型逐项检测 Adapter / YAML / OutputContract / Runtime,
并给出该声明的**自动建议状态**, 防止把缺 yaml/缺 runtime 覆盖/缺契约的模型
误写成 stable 对外宣称(P0-7 与 §49/§51)。

判定原则:
* ``assert_support_claims`` 是**门禁**: 声明为 ``stable`` 但存在结构性缺失
  (adapter/yaml/contract/runtime 任一不满足)即报错, 拒绝"假 stable";
* 自动判定**不会主动升到 stable**: stable 仍需人工确认 + 真机 conformance 证据,
  readiness 只负责把"应降级/未就绪"暴露出来。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from xpu_converter.errors import ConfigError
from xpu_converter.paths import model_config_path
from xpu_converter.registry import model_registry as _reg

#: 覆盖检测依赖用的 README/目录根, 便于测试向临时目录注入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

#: 通用 Runtime 解码器真正可消费的任务(依据 runtime/detection/nwai_decoders.py)。
#: 检测这些任务会被通用 Runtime 覆盖, 缺失即视为"运行时无法解码该任务"。
RUNTIME_DECODE_TASKS = frozenset(
    {"detection", "segment", "pose", "obb", "cls", "depth", "sem"}
)

#: OutputContract 从 task 确定性推断输出的任务(依据 xpu_converter/contract/output.py)。
CONTRACT_TASKS = frozenset(
    {"detection", "segment", "pose", "obb", "cls", "depth", "sem"}
)

STATUS_STABLE = _reg.STATUS_STABLE
STATUS_EXPERIMENTAL = _reg.STATUS_EXPERIMENTAL
STATUS_PLANNED = _reg.STATUS_PLANNED


@dataclass
class ReadinessItem:
    """一项结构性组件检测结果。"""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class ModelReadiness:
    """一个模型的自动就绪检测结果。"""

    model_type: str
    task: str = "detection"
    declared_status: str = STATUS_EXPERIMENTAL
    items: List[ReadinessItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def structural_ok(self) -> bool:
        """结构性(adapter/yaml/contract/runtime)是否全部就绪。"""
        return bool(self.items) and all(item.ok for item in self.items)

    def missing(self) -> List[str]:
        return [item.name for item in self.items if not item.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "task": self.task,
            "declared_status": self.declared_status,
            "structural_ok": self.structural_ok,
            "items": [{"name": i.name, "ok": i.ok, "detail": i.detail} for i in self.items],
            "missing": self.missing(),
            "warnings": list(self.warnings),
        }


def effective_status(readiness: ModelReadiness) -> str:
    """根据自动检测对声明状态做晋升/降级建议(stable 只降不升)。

    * 声明 ``stable`` 但结构缺失 → 降为 ``experimental``(暴露假 stable);
    * 其余情况维持声明(experimental 不会仅因代码就绪而自动晋升 stable,
      后者要求人工确认 + 真机 conformance 证据)。
    """
    if readiness.declared_status == STATUS_STABLE and not readiness.structural_ok:
        return STATUS_EXPERIMENTAL
    return readiness.declared_status


def check_model_readiness(
    model_type: str,
    loaded_adapters: Optional[Sequence[str]] = None,
) -> ModelReadiness:
    """自动检测单个模型的结构就绪度。``loaded_adapters`` 缺省时按需加载适配器。"""
    key = str(model_type or "").strip().lower()
    support = _reg.model_support(key)
    task = support.task or "detection"
    items: List[ReadinessItem] = []

    if loaded_adapters is not None:
        adapters = set(loaded_adapters)
    else:
        try:
            adapters = set(_reg.available_model_types())
        except Exception:  # 依赖缺失时按"未知"处理, 不因检测而崩溃
            adapters = set()

    items.append(ReadinessItem(
        "adapter", key in adapters,
        "model_type 已注册适配器" if key in adapters else "未找到对应适配器",
    ))
    yaml_path = model_config_path(key) if key else ""
    has_yaml = bool(yaml_path) and Path(yaml_path).is_file()
    items.append(ReadinessItem(
        "yaml", has_yaml,
        "配置文件存在" if has_yaml else "缺 configs/models/{}.yaml".format(key),
    ))
    items.append(ReadinessItem(
        "contract", task in CONTRACT_TASKS,
        "任务 {} 可确定性推断输出布局".format(task) if task in CONTRACT_TASKS
        else "任务 {} 无对应输出契约".format(task),
    ))
    runtime_ok = task in RUNTIME_DECODE_TASKS
    items.append(ReadinessItem(
        "runtime", runtime_ok,
        "通用 Runtime 可解码任务 {}".format(task) if runtime_ok
        else "无 Runtime 解码器覆盖任务 {}".format(task),
    ))

    readiness = ModelReadiness(
        model_type=key, task=task, declared_status=support.status, items=items,
    )
    _warn_conformance(readiness)
    return readiness


def _warn_conformance(readiness: ModelReadiness) -> None:
    """提示: 有 stable/experimental 声明但缺少可复现的逐级回归证据(不阻断)。"""
    golden_dir = Path(_PROJECT_ROOT) / "tests" / "golden"
    covered = False
    if golden_dir.is_dir():
        try:
            for child in golden_dir.iterdir():
                if readiness.model_type in child.name.lower():
                    covered = True
                    break
        except OSError:
            pass
    if readiness.declared_status == STATUS_STABLE and not covered:
        readiness.warnings.append(
            "已声明 stable 但未见 tests/golden 回归用例, 建议补充真机 conformance 证据"
        )


def assert_support_claims(
    loaded_adapters: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """门禁: 校验所有声明为 ``stable`` 的模型结构就绪(ChatGPT P0-7)。

    对存在结构性缺失的 stable 声明抛 :class:`ConfigError`, 防止"假完成";
    返回各模型的就绪摘要供审计(ChatGPT 修改意见 §"真假完成判断" 用)。
    """
    summary: List[Dict[str, Any]] = []
    for support in _reg.supported_models(STATUS_STABLE):
        readiness = check_model_readiness(support.model_type, loaded_adapters)
        summary.append(readiness.to_dict())
        if not readiness.structural_ok:
            raise ConfigError(
                "模型 {} 声明为 stable 但结构未就绪, 缺失: {}({}); "
                "请补齐或改为 experimental".format(
                    readiness.model_type,
                    ", ".join(readiness.missing()),
                    "; ".join(item.detail for item in readiness.items if not item.ok),
                )
            )
    return summary