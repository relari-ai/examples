from typing import Any, Dict, List, Tuple

import networkx as nx
from continuous_eval.eval import Module
from continuous_eval.eval import Pipeline as CEPipeline
from continuous_eval.eval.dataset import Dataset, DatasetField
from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.result_types import MetricsResults
from continuous_eval.eval.runner import EvaluationRunner
from continuous_eval.eval.tests import Test
from continuous_eval.metrics import Metric
from haystack import Pipeline as HSPipeline


class PipelineEvaluator:
    """
    A class that represents a pipeline evaluator.

    The `PipelineEvaluator` class is responsible for evaluating a pipeline by running it on a dataset and
    calculating the metrics. It provides methods to add metrics to specific modules in the pipeline and
    to run the evaluation on a dataset.

    Methods:
        __getitem__(self, module_name): Get a module from the pipeline by its name.
        add_metric(self, module: str, metric: Metric): Add a metric to a specific module.
        add_metrics(self, module: str, metrics: List[Metric]): Add a list of metrics to a specific module.
        run_evaluation(self, dataset: Dataset, input_map: Dict[str, Dict[str, Any]]) -> MetricsResults:
            Run the evaluation for the given dataset and return the results.
    """

    def __init__(self, pipeline: HSPipeline):
        """
        Initialize a new instance of the `PipelineEvaluator` class.

        Args:
            pipeline (HSPipeline): The Haystack pipeline to be evaluated.
        """
        self._haystack_pipeline = pipeline
        self._modules = self._p2p(pipeline)

    # Rest of the code...


class PipelineEvaluator:
    def __init__(self, pipeline: HSPipeline):
        self._haystack_pipeline = pipeline
        self._modules = self._p2p(pipeline)

    def __getitem__(self, module_name):
        return self._modules[module_name]

    def _p2p(self, pipeline: HSPipeline):
        in_out = {n: {"in": dict(), "out": dict()} for n in pipeline.graph.nodes}
        module_inputs = {n: set() for n in pipeline.graph.nodes}
        for n, info in pipeline.graph.nodes(data=True):
            for i in info.get("input_sockets", {}).values():
                in_out[n]["in"][i.name] = i.type
                if i.senders:
                    for s in i.senders:
                        module_inputs[n].add(s)
            for o in info.get("output_sockets", {}).values():
                in_out[n]["out"][o.name] = o.type
                if o.receivers:
                    for r in o.receivers:
                        module_inputs[r].add(n)

        __modules = {
            n: {  # new module
                "name": n,
                "input": None,
                "output": None,
                "description": None,
                "eval": None,
                "tests": None,
            }
            for n in pipeline.graph.nodes
        }

        # Set output
        for n, info in in_out.items():
            tpy = list(info["out"].values())
            __modules[n]["output"] = tpy[0] if len(tpy) == 1 else Tuple[*tpy]  # type: ignore

        # Creation and set input (we need other modules to be created first)
        modules = dict()
        for n in list(list(nx.topological_sort(pipeline.graph))):
            if module_inputs[n]:
                # Another module as input
                tmp = list(module_inputs[n])
                __modules[n]["input"] = (
                    modules[tmp[0]] if len(tmp) == 1 else [modules[t] for t in tmp]
                )
            else:
                # Dataset inputs
                tmp = list(in_out[n]["in"].keys())
                __modules[n]["input"] = (
                    DatasetField(tmp[0])
                    if len(tmp) == 1
                    else [DatasetField(t) for t in tmp]
                )
            modules[n] = Module(**__modules[n])

        return modules

    def add_metric(self, module: str, metric: Metric):
        """
        Adds a metric to the specified module.

        Args:
            module (str): The name of the module to add the metric to.
            metric (Metric): The metric to be added.

        Returns:
            None
        """
        _module = self._modules[module]
        _metrics = _module.eval if _module.eval is not None else list()
        _metrics.append(metric)
        self._modules[module] = Module(
            name=_module.name,
            input=_module.input,
            output=_module.output,
            description=_module.description,
            eval=_metrics,
            tests=_module.tests,
        )

    def add_metrics(self, module: str, metrics: List[Metric]):
        """
        Adds a list of metrics to the specified module.

        Args:
            module (str): The name of the module to add the metrics to.
            metrics (List[Metric]): A list of Metric objects to be added.

        Returns:
            None
        """
        for metric in metrics:
            self.add_metric(module, metric)

    def _run_pipeline(
        self,
        plog: PipelineLogger,
        dataset: Dataset,
        input_map: Dict[str, Dict[str, Any]],
    ) -> PipelineLogger:
        module_names = list(self._modules.keys())
        for datum in dataset.data:
            data = {
                module: {
                    key: datum[value.name] if isinstance(value, DatasetField) else value
                    for key, value in input_map[module].items()
                }
                for module in input_map
            }
            out = self._haystack_pipeline.run(data, include_outputs_from=module_names)
            for m in module_names:
                plog.log(uid=datum["uid"], module=m, value=out[m])
        return plog

    def run_evaluation(
        self, dataset: Dataset, input_map: Dict[str, Dict[str, Any]]
    ) -> MetricsResults:
        """
        Run the evaluation for the given dataset.

        Args:
            dataset (Dataset): The dataset to evaluate.
            input_map (Dict[str, Dict[str, Any]]): A dictionary mapping modules' input names to input values.

        Returns:
            MetricsResults: The results of the evaluation.
        """
        pipeline = CEPipeline(list(self._modules.values()))
        pipeline.dataset = dataset  # trick to avoid dataset fields validation
        plog = PipelineLogger(pipeline=pipeline)
        self._run_pipeline(plog=plog, dataset=dataset, input_map=input_map)
        evalrunner = EvaluationRunner(pipeline)
        return evalrunner.evaluate(plog)
