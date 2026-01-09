from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..errors import CodegenError


@dataclass(frozen=True)
class BinaryModel:
    name: str
    input_names: tuple[str, str]
    output_name: str
    element_count: int
    operator: str


@dataclass(frozen=True)
class BinaryChainModel:
    name: str
    input_names: tuple[str, ...]
    output_name: str
    element_count: int
    first_inputs: tuple[str, str]
    first_operator: str
    second_input: str
    second_operator: str
    temp_on_left: bool


class CEmitter:
    def __init__(self, template_dir: Path) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def emit_binary_model(self, model: BinaryModel) -> str:
        try:
            template = self._env.get_template("model.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        rendered = template.render(
            model_name=model.name,
            op_name=f"{model.name}_op0",
            input0=model.input_names[0],
            input1=model.input_names[1],
            output=model.output_name,
            element_count=model.element_count,
            operator=model.operator,
        )
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    def emit_binary_chain_model(self, model: BinaryChainModel) -> str:
        try:
            template = self._env.get_template("model_chain.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        signature = ", ".join(
            f"const float* {name}" for name in model.input_names
        )
        rendered = template.render(
            model_name=model.name,
            op0_name=f"{model.name}_op0",
            op1_name=f"{model.name}_op1",
            signature=signature,
            output=model.output_name,
            element_count=model.element_count,
            first_input0=model.first_inputs[0],
            first_input1=model.first_inputs[1],
            first_operator=model.first_operator,
            second_input=model.second_input,
            second_operator=model.second_operator,
            temp_on_left=model.temp_on_left,
        )
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered
