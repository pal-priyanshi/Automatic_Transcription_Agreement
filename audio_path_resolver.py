"""Helpers for letting ASR code read normal files and tar-shard members."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np
import soundfile as sf

PathLike = Union[str, Path]
AudioArray = Tuple[np.ndarray, int]  # (samples, sample_rate)


def is_tar_uri(audio_path: PathLike) -> bool:
    """Return True when the path points to a file inside a tar shard."""
    return str(audio_path).startswith("tar://")


def parse_tar_uri(audio_path: PathLike) -> tuple[Path, str]:
    """Split ``tar://SHARD::MEMBER`` into the tar path and inner member path."""
    value = str(audio_path)
    if not value.startswith("tar://") or "::" not in value:
        raise ValueError(f"Expected tar URI in the form tar://SHARD::MEMBER, got {value}")

    shard, inner_path = value[len("tar://"):].split("::", 1)
    if not shard or not inner_path:
        raise ValueError(f"Expected tar URI in the form tar://SHARD::MEMBER, got {value}")

    return Path(shard), inner_path


def find_tar_member(tar: tarfile.TarFile, inner_path: str) -> tarfile.TarInfo:
    """Find an audio member, allowing Emilia JSON paths to omit tar prefixes."""
    try:
        return tar.getmember(inner_path)
    except KeyError:
        pass

    members = [member for member in tar.getmembers() if member.isfile()]
    matches = [member for member in members if member.name.endswith(inner_path)]
    if not matches:
        basename = Path(inner_path).name
        matches = [member for member in members if Path(member.name).name == basename]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise KeyError(f"filename {inner_path!r} not found")

    match_names = [member.name for member in matches[:5]]
    raise KeyError(
        f"filename {inner_path!r} matched multiple tar members: {match_names}"
    )


def resolved_audio_array(audio_path: PathLike) -> AudioArray:
    """Return ``(samples_array, sample_rate)`` for a normal path or tar URI.

    For regular filesystem paths, reads directly with soundfile.
    For ``tar://SHARD::MEMBER`` URIs, extracts the member bytes into a
    ``BytesIO`` buffer and reads from there — no temp files, no disk I/O.
    """
    if not is_tar_uri(audio_path):
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        return audio, int(sr)

    shard, inner_path = parse_tar_uri(audio_path)
    with tarfile.open(shard, "r:*") as tar:
        member = find_tar_member(tar, inner_path)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not read {inner_path} from {shard}")
        audio_bytes = extracted.read()

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return audio, int(sr)


def stream_tar_audio(
    tar_path: Path,
    member_names: set[str],
) -> Iterator[tuple[str, np.ndarray, int]]:
    """Stream audio arrays from a single tar, opening it only once.

    Yields ``(member_name, samples_array, sample_rate)`` for each requested
    member found in the archive, in archive order.  Use this when processing
    multiple files from the same tar to avoid reopening it per file.
    """
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if member.name not in member_names and name not in member_names:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            audio_bytes = extracted.read()
            try:
                audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
                yield member.name, audio, int(sr)
            except Exception:
                pass
