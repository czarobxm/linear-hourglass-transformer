from typing import List
from dataclasses import dataclass


@dataclass
class NeptuneCfg:
    project_name: str
    api_token: str
    custom_run_name: str
    name: str
    tags: List[str]
