from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


DEFAULT_EAST_URLS = [
    "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb",
    "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb",
]


@dataclass
class EastResolveResult:
    model_path: str | None
    downloaded: bool
    message: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_model_dir() -> Path:
    # .../apps/worker/src/pipeline/python/ocr_v2_east.py -> .../apps/worker/models/east
    return Path(__file__).resolve().parents[3] / "models" / "east"


def default_model_path() -> Path:
    return default_model_dir() / "frozen_east_text_detection.pb"


def _is_usable_model(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= 1_000_000
    except Exception:
        return False


def download_east_model(target_path: Path, urls: list[str], timeout_sec: float = 45.0) -> tuple[bool, str]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="east_model_dl_"))
    tmp_file = tmp_dir / "east.pb.tmp"
    last_err = ""
    try:
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp, tmp_file.open("wb") as out:
                    shutil.copyfileobj(resp, out)
                if tmp_file.stat().st_size < 1_000_000:
                    last_err = f"downloaded file too small from {url}"
                    continue
                shutil.move(str(tmp_file), str(target_path))
                return True, f"downloaded from {url}"
            except Exception as exc:  # pragma: no cover
                last_err = f"{url}: {exc}"
                continue
        return False, f"failed to download EAST model ({last_err})"
    finally:
        try:
            if tmp_file.exists():
                tmp_file.unlink()
            tmp_dir.rmdir()
        except Exception:
            pass


def resolve_east_model_path(
    explicit_model_path: str | None,
    model_dir: str | None,
    auto_download: bool,
    model_urls: list[str] | None = None,
    sha256: str | None = None,
) -> EastResolveResult:
    candidates: list[Path] = []
    explicit = str(explicit_model_path or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    base_dir = Path(model_dir).expanduser().resolve() if str(model_dir or "").strip() else default_model_dir()
    candidates.append(base_dir / "frozen_east_text_detection.pb")
    candidates.append(default_model_path())

    for c in candidates:
        if _is_usable_model(c):
            if sha256:
                got = _sha256_file(c)
                if got.lower() != sha256.lower():
                    return EastResolveResult(None, False, f"sha256 mismatch for {c}: expected {sha256}, got {got}")
            return EastResolveResult(str(c), False, f"using local EAST model: {c}")

    if not auto_download:
        return EastResolveResult(None, False, "EAST model not found and auto-download disabled")

    urls = list(model_urls or DEFAULT_EAST_URLS)
    target = candidates[0] if candidates else default_model_path()
    ok, msg = download_east_model(target, urls)
    if not ok:
        return EastResolveResult(None, False, msg)
    if sha256:
        got = _sha256_file(target)
        if got.lower() != sha256.lower():
            try:
                os.remove(target)
            except Exception:
                pass
            return EastResolveResult(None, False, f"sha256 mismatch after download: expected {sha256}, got {got}")
    return EastResolveResult(str(target), True, msg)

