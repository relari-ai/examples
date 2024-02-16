from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

from examples.eval.dataset import DatasetField

Metric = Callable
Test = Callable


@dataclass(frozen=True, eq=True)
class Tool:
    name: str
    args: Dict[str, Type]
    out_type: Type
    description: Optional[str] = field(default=None)


@dataclass(frozen=True, eq=True)
class Module:
    name: str
    input: DatasetField | Type | "Module"
    output: Type
    description: Optional[str] = field(default=None)
    expected_output: Optional[DatasetField] = field(default=None)
    eval: Optional[List["Metric"]] = field(default=None)
    tests: Optional[List["Test"]] = field(default=None)


@dataclass(frozen=True, eq=True)
class AgentModule(Module):
    tools: Optional[List[Tool]] = field(default=None)
    expected_tool_calls: Optional[DatasetField] = field(default=None)
    is_recursive: bool = field(default=False)
