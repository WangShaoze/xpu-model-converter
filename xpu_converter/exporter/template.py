# -*- coding: utf-8 -*-
"""交付层模板渲染。

Dockerfile / build.sh / install.conf / readme.txt / start.sh 全部由模板渲染生成,
文件名与内容来自同一份 metadata, 从根本上消灭
``manifest ≠ Dockerfile ≠ README ≠ 实际文件`` 的不一致(建设目标 §15)。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.errors import PackageError
from xpu_converter.paths import docker_template_dir

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:  # pragma: no cover - Jinja2 为必装依赖
    Environment = None
    FileSystemLoader = None
    StrictUndefined = None

# 模板文件 -> 交付包内输出文件名
TEMPLATE_OUTPUTS: Dict[str, str] = {
    "Dockerfile.j2": "Dockerfile",
    "build.sh.j2": "build.sh",
    "install.conf.j2": "install.conf",
    "readme.txt.j2": "readme.txt",
    "start.sh.j2": "start.sh",
}


class TemplateRenderer:
    """按硬件后端渲染交付层模板。"""

    def __init__(self, hardware: str = "kunlun", template_dir: Optional[str] = None) -> None:
        self.hardware = hardware
        self.template_dir = Path(template_dir) if template_dir else docker_template_dir(hardware)
        if not self.template_dir.is_dir():
            raise PackageError("交付模板目录不存在: {}".format(self.template_dir))
        if Environment is None:
            raise PackageError("缺少 Jinja2 依赖, 无法渲染交付模板")
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )

    def names(self) -> List[str]:
        return sorted(TEMPLATE_OUTPUTS)

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        try:
            template = self._env.get_template(template_name)
        except Exception as err:
            raise PackageError("模板不存在或语法错误 {}: {}".format(template_name, err))
        try:
            return template.render(**context)
        except Exception as err:
            raise PackageError("模板渲染失败 {}: {}".format(template_name, err))

    def render_to(self, template_name: str, output_path, context: Dict[str, Any]) -> str:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(template_name, context), encoding="utf-8")
        return str(target)

    def render_all(self, output_dir, context: Dict[str, Any]) -> List[str]:
        """渲染全部交付层文件, 返回生成的文件路径。"""
        produced: List[str] = []
        for template_name, filename in TEMPLATE_OUTPUTS.items():
            produced.append(self.render_to(template_name, Path(output_dir) / filename, context))
        return produced
