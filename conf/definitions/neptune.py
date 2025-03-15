from dataclasses import dataclass


@dataclass
class NeptuneCfg:
    project_name: str
    api_token: str
    rolling_window_sizes: list
    smoothing_factors: list
