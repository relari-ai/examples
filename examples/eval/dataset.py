import typing 
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, field

DatasetValue = typing.Any
UUID = str

_SAFE_DICT = {k: v for k, v in typing.__dict__.items() if not k.startswith("__")}
_SAFE_DICT["UUID"] = UUID

@dataclass(frozen=True)
class DatasetField:
    name: str
    type: type
    description: str

class Dataset:
    def __init__(self, dataset_name:str) -> None:
        base_path = Path("dataset")
        dataset_path = base_path / dataset_name
        assert dataset_path.exists(), f"Dataset {dataset_name} does not exist"
        # Load manifest
        with open(dataset_path/"manifest.yaml", "r") as manifest_file:
            self._manifest = yaml.safe_load(manifest_file)
        # load jsonl dataset
        with open(dataset_path/"dataset.jsonl", "r") as json_file:
            self._data = list(json_file)
        # create dynamic properties
        self._create_dynamic_properties()


    def _create_dynamic_properties(self):
        # Dynamically add a property for each field
        for field_name, field_info in self._manifest["fields"].items():
            try:
                _field = DatasetField(
                    name=field_name,
                    type=eval(field_info['type'],_SAFE_DICT),
                    description=field_info['description'])
                setattr(self, field_name, _field)
            except:
                raise ValueError(f"Field type {field_info['type']} not supported")

    def filed_types(self, name:str) -> type:
        return getattr(self, name).type
    # def __getitem__(self, key: DatasetKey) -> DatasetValue:
    #     return None
