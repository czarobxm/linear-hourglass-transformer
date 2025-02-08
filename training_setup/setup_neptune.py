import hashlib
from pathlib import Path
import signal
import time

import neptune

from conf.definitions import NeptuneCfg


def get_api_token(cfg_neputne: NeptuneCfg) -> str:
    script_dir = Path(__file__).parent.resolve()
    with open(f"{script_dir}/../{cfg_neputne.api_token}", "r", encoding="UTF-8") as file:
        api_token = file.read().strip()
    return api_token


def get_project_name(cfg_neptune: NeptuneCfg) -> str:
    script_dir = Path(__file__).parent.resolve()
    with open(
        f"{script_dir}/../{cfg_neptune.project_name}", "r", encoding="UTF-8"
    ) as file:
        project_name = file.read().strip()
    return project_name


def setup_neptune(cfg_neptune: NeptuneCfg) -> neptune.Run:
    run = neptune.init_run(
        project=get_project_name(cfg_neptune),
        api_token=get_api_token(cfg_neptune),
        custom_run_id=hashlib.md5(str(time.time()).encode()).hexdigest(),
    )

    def handler(sig, frame):  # pylint: disable=unused-argument
        run.stop()

    signal.signal(signal.SIGINT, handler)
    return run
