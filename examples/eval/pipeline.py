from typing import get_origin, Set, Tuple, List, Dict, Any, Optional
from examples.eval.dataset import Dataset, DatasetField
from examples.eval.modules import Module
from dataclasses import dataclass, field

@dataclass
class Graph:
    nodes: Set[str]
    edges: Set[Tuple[str, str]]

class Pipeline:
    def __init__(
        self, modules: List[Module], dataset: Optional[Dataset] = None
    ) -> None:
        self._modules = modules
        self._dataset = dataset
        self._graph = self._build_graph()

    @property
    def modules(self):
        return self._modules

    def module_by_name(self, name: str) -> Module:
        for module in self._modules:
            if module.name == name:
                return module
        raise ValueError(f"Module {name} not found")

    def _validate_modules(self):
        names = set()
        for module in self._modules:
            if module.name in names:
                raise ValueError(f"Module {module.name} already exists")
            names.add(module.name)
            if self._dataset is not None and module.expected_output is not None:
                if get_origin(module.output) != get_origin(
                    self._dataset.getattr(self, module.expected_output).type
                ):
                    raise ValueError(f"Field {module.output} does not match expected type in the dataset.")

    def _build_graph(self):
        nodes = {m.name: m for m in self._modules}
        edges = set()
        for module in self._modules:
            if isinstance(module.input, Module):
                edges.add((module.input.name, module.name))
        return Graph(nodes, edges)

    def print_graph(self):
        print("Nodes:")
        for node in self._graph.nodes:
            print(node)
        print("Edges:")
        for edge in self._graph.edges:
            print(edge)
