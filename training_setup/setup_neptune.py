import argparse
import hashlib
import signal
import time

import neptune


def setup_neptune(args: argparse.Namespace) -> neptune.Run:
    if args.project is None or args.api_token is None:
        raise ValueError("No project or API token provided for Neptune")
    run = neptune.init_run(
        project=args.project,
        api_token=args.api_token,
        custom_run_id=hashlib.md5(str(time.time()).encode()).hexdigest(),
        name=args.name,
        tags=args.tags,
    )

    def handler(sig, frame):  # pylint: disable=unused-argument
        run.stop()

    signal.signal(signal.SIGINT, handler)
    return run
