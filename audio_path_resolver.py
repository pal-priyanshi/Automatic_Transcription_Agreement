"""Helpers for letting ASR code read normal files and tar-shard members."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

PathLike = Union[str, Path]


def is_tar_uri(audio_path: PathLike) -> bool:
    """Return True when the path points to a file inside a tar shard."""
    return str(audio_path).startswith("tar://")


def parse_tar_uri(audio_path: PathLike) -> tuple[Path, str]:
    """Split ``tar://SHARD::MEMBER`` into the tar path and inner member path."""
    value = str(audio_path)
    if not value.startswith("tar://") or "::" not in value:
        raise ValueError(f"Expected tar URI in the form tar://SHARD::MEMBER, got {value}")

    shard, inner_path = value[len("tar://") :].split("::", 1)
    if not shard or not inner_path:
        raise ValueError(f"Expected tar URI in the form tar://SHARD::MEMBER, got {value}")

    return Path(shard), inner_path


def find_tar_member(tar: tarfile.TarFile, inner_path: str) -> tarfile.TarInfo:
    """Find an audio member, allowing Emilia JSON paths to omit tar prefixes."""
    try:
        return tar.getmember(inner_path)
    except KeyError:
        pass

    matches = [
        member
        for member in tar.getmembers()
        if member.isfile() and member.name.endswith(inner_path)
    ]
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise KeyError(f"filename {inner_path!r} not found")

    match_names = [member.name for member in matches[:5]]
    raise KeyError(
        f"filename {inner_path!r} matched multiple tar members: {match_names}"
    )


@contextmanager
def resolved_audio_path(audio_path: PathLike) -> Iterator[Path]:
    """Yield a real filesystem path for normal audio files or tar members.

    Existing ATA backends expect ordinary files. For normal paths we simply
    yield the input path. For ``tar://`` paths, we copy only the requested member
    into a temporary directory and delete it after the caller is done.
    """
    if not is_tar_uri(audio_path):
        yield Path(audio_path)
        return

    shard, inner_path = parse_tar_uri(audio_path)
    suffix = Path(inner_path).suffix or ".audio"

    with tempfile.TemporaryDirectory(prefix="ata_tar_audio_") as tmp_dir:
        tmp_audio = Path(tmp_dir) / f"audio{suffix}"

        with tarfile.open(shard, "r:*") as tar:
            member = find_tar_member(tar, inner_path)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Could not read {inner_path} from {shard}")

            with tmp_audio.open("wb") as out_f:
                shutil.copyfileobj(extracted, out_f)

        yield tmp_audio
