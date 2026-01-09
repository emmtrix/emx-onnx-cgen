from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..errors import CodegenError


@dataclass(frozen=True)
class AddModel:
    name: str
    input_names: tuple[str, str]
    output_name: str
    element_count: int


class CEmitter:
    def __init__(self, template_dir: Path) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def emit_add_model(self, model: AddModel) -> str:
        try:
            template = self._env.get_template("model.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        rendered = template.render(
            model_name=model.name,
            input0=model.input_names[0],
            input1=model.input_names[1],
            output=model.output_name,
            element_count=model.element_count,
        )
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered
