import os
from pathlib import Path
import hashlib
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


def sha256(f) -> str:
    sha256_hash = hashlib.sha256()
    # Read and update hash in chunks of 4M
    for byte_block in iter(lambda: f.read(4 * 1024 * 1024), b""):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def check_model(
    dir_name: Path, model_name: str, hash: str, remove_incorrect=False
) -> bool:
    target = dir_name / model_name
    relname = target.as_posix()
    logger.debug(f"checking {relname}...")
    if not os.path.exists(target):
        logger.info(f"{target} not exist.")
        return False
    with open(target, "rb") as f:
        digest = sha256(f)
        bakfile = f"{target}.bak"
        if digest != hash:
            logger.warn(f"{target} sha256 hash mismatch.")
            logger.info(f"expected: {hash}")
            logger.info(f"real val: {digest}")
            logger.warn("please add parameter --update to download the latest assets.")
            if remove_incorrect:
                if not os.path.exists(bakfile):
                    os.rename(str(target), bakfile)
                else:
                    os.remove(str(target))
            return False
        if remove_incorrect and os.path.exists(bakfile):
            os.remove(bakfile)
    return True


def check_all_assets(update=False) -> bool:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    logger.info("checking assets...")
    current_dir = BASE_DIR / "asset"
    names = [
        "Decoder.pt",
        "DVAE.pt",
        "GPT.pt",
        "spk_stat.pt",
        "tokenizer.pt",
        "Vocos.pt",
    ]
    for model in names:
        menv = model.replace(".", "_")
        if not check_model(
            current_dir, model, os.environ[f"sha256_asset_{menv}"], update
        ):
            return False

    logger.info("checking configs...")
    current_dir = BASE_DIR / "config"
    names = [
        "decoder.yaml",
        "dvae.yaml",
        "gpt.yaml",
        "path.yaml",
        "vocos.yaml",
    ]
    for model in names:
        menv = model.replace(".", "_")
        if not check_model(
            current_dir, model, os.environ[f"sha256_config_{menv}"], update
        ):
            return False

    logger.info("all assets are already latest.")
    return True


def download_and_extract_tar_gz(url: str, folder: str):
    import tarfile

    logger.info(f"downloading {url}")
    response = requests.get(url, stream=True, timeout=(5, 10))
    with BytesIO() as out_file:
        out_file.write(response.content)
        out_file.seek(0)
        logger.info(f"downloaded.")
        with tarfile.open(fileobj=out_file, mode="r:gz") as tar:
            tar.extractall(folder)
        logger.info(f"extracted into {folder}")


def download_and_extract_zip(url: str, folder: str):
    import zipfile

    logger.info(f"downloading {url}")
    response = requests.get(url, stream=True, timeout=(5, 10))
    with BytesIO() as out_file:
        out_file.write(response.content)
        out_file.seek(0)
        logger.info(f"downloaded.")
        with zipfile.ZipFile(out_file) as zip_ref:
            zip_ref.extractall(folder)
        logger.info(f"extracted into {folder}")


def download_dns_yaml(url: str, folder: str):
    logger.info(f"downloading {url}")
    response = requests.get(url, stream=True, timeout=(5, 10))
    with open(os.path.join(folder, "dns.yaml"), "wb") as out_file:
        out_file.write(response.content)
        logger.info(f"downloaded into {folder}")


def download_all_assets(tmpdir: str, version="0.2.5"):
    import subprocess
    import platform

    archs = {
        "aarch64": "arm64",
        "armv8l": "arm64",
        "arm64": "arm64",
        "x86": "386",
        "i386": "386",
        "i686": "386",
        "386": "386",
        "x86_64": "amd64",
        "x64": "amd64",
        "amd64": "amd64",
    }
    system_type = platform.system().lower()
    architecture = platform.machine().lower()
    is_win = system_type == "windows"

    architecture = archs.get(architecture, None)
    if not architecture:
        logger.error(f"architecture {architecture} is not supported")
        exit(1)
    try:
        BASE_URL = "https://github.com/fumiama/RVC-Models-Downloader/releases/download/"
        suffix = "zip" if is_win else "tar.gz"
        RVCMD_URL = BASE_URL + f"v{version}/rvcmd_{system_type}_{architecture}.{suffix}"
        cmdfile = os.path.join(tmpdir, "rvcmd")
        if is_win:
            download_and_extract_zip(RVCMD_URL, tmpdir)
            cmdfile += ".exe"
        else:
            download_and_extract_tar_gz(RVCMD_URL, tmpdir)
            os.chmod(cmdfile, 0o755)
        subprocess.run([cmdfile, "-notui", "-w", "0", "assets/chtts"])
    except Exception:
        BASE_URL = "https://raw.gitcode.com/u011570312/RVC-Models-Downloader/assets/"
        suffix = {
            "darwin_amd64": "555",
            "darwin_arm64": "556",
            "linux_386": "557",
            "linux_amd64": "558",
            "linux_arm64": "559",
            "windows_386": "562",
            "windows_amd64": "563",
        }[f"{system_type}_{architecture}"]
        RVCMD_URL = BASE_URL + suffix
        download_dns_yaml(
            "https://raw.gitcode.com/u011570312/RVC-Models-Downloader/raw/main/dns.yaml",
            tmpdir,
        )
        if is_win:
            download_and_extract_zip(RVCMD_URL, tmpdir)
            cmdfile += ".exe"
        else:
            download_and_extract_tar_gz(RVCMD_URL, tmpdir)
            os.chmod(cmdfile, 0o755)
        subprocess.run(
            [
                cmdfile,
                "-notui",
                "-w",
                "0",
                "-dns",
                os.path.join(tmpdir, "dns.yaml"),
                "assets/chtts",
            ]
        )
