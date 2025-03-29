import urllib
import tarfile
from pathlib import Path


def download_lra(path: Path):
    if (path / "lra_release 3").exists():
        return None
    path.mkdir(exist_ok=True, parents=True)

    zip_path = path / "lra_release.gz"

    # Download
    if not zip_path.exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/long-range-arena/lra_release.gz", zip_path
        )

    # Unpack file
    with tarfile.open(zip_path, "r:gz") as tar:
        for member in tar.getmembers():
            tar.extract(member, path=path, filter="data")

    # Remove zip file
    zip_path.unlink()
