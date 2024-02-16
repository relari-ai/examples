from typing import get_origin, Type
from examples.eval.pipeline import Pipeline
from examples.eval.dataset import Dataset


def _instantiate_type(type_hint: Type):
    origin = get_origin(type_hint)
    # If the origin is None, it means type_hint is not a generic type
    # and we assume type_hint itself is directly instantiable
    if origin is None:
        origin = type_hint
    try:
        # This only works for types without required arguments in their __init__.
        instance = origin()
    except TypeError as e:
        # If instantiation fails, return an error message or raise a custom exception
        instance = None
    return instance


class EvaluationManager:
    def __init__(self):
        self._pipeline = None
        self._dataset = None
        self._samples = None

        self._idx = 0

        self._running = False

    def _build_empty_samples(self):
        assert self.pipeline is not None, "Pipeline not set"
        empty_samples = dict()
        for module in self.pipeline.modules:
            empty_samples[module.name] = _instantiate_type(module.output)
        return empty_samples

    @property
    def is_complete(self):
        return self._idx == len(self._dataset)

    @property
    def samples(self):
        return self._samples

    @property
    def pipeline(self) -> Pipeline | None:
        return self._pipeline

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    def set_pipeline(self, pipeline: Pipeline):
        self._pipeline = pipeline

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def is_running(self) -> bool:
        return self._running

    def start_run(self):
        self._running = True
        self._idx = 0
        self._samples = [
            self._build_empty_samples() for _ in range(len(self._dataset.data))
        ]

    @property
    def curr_sample(self):
        if self._idx >= len(self._dataset.data):
            return None
        return self._dataset.data[self._idx]

    def next_sample(self):
        if self._idx >= len(self._dataset.data):
            self._running = False
        else:
            self._idx += 1
        return self.curr_sample

    def log(self, key, value):
        assert get_origin(value) == get_origin(
            self._pipeline.module_by_name(key).output
        ), f"Value {value} does not match expected type in the pipeline"
        if not self._running:
            raise ValueError("Cannot log when not running")
        if key not in self._samples[self._idx]:
            raise ValueError(f"Key {key} not found, review your pipeline")
        if isinstance(self._samples[self._idx][key], list):
            self._samples[self._idx][key].append(value)
        elif isinstance(self._samples[self._idx][key], dict):
            self._samples[self._idx][key].update(value)
        elif isinstance(self._samples[self._idx][key], set):
            self._samples[self._idx][key].add(value)
        else:
            self._samples[self._idx][key] = value


eval_manager = EvaluationManager()
