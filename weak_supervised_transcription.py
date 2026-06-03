"""Generate ASR transcripts for a dataset and optionally compute agreement scores."""

from __future__ import annotations

import argparse
import io
import logging
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from asr_model_parser import ASRParser
from transcription_agreement import enrich_dataframe

LOGGER = logging.getLogger(__name__)
ERROR_TOKEN = "<Error>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV file containing a 'wav' column.")
    parser.add_argument(
        "config",
        type=Path,
        help="YAML file with 'asr_models' entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV path. Defaults to 'transcribed_<input name>'.",
    )
    parser.add_argument(
        "--same-gpu",
        action="store_true",
        help="Load every model on the first CUDA device instead of spreading them.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Load one model at a time to reduce peak GPU memory. "
            "Streams through the tar once per model instead of loading all models simultaneously."
        ),
    )
    parser.add_argument(
        "--skip-agreement",
        action="store_true",
        help="Only store raw transcripts without computing agreement columns.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during processing.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(index: int, *, same_gpu: bool) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if same_gpu:
        return torch.device("cuda:0")
    device_count = max(torch.cuda.device_count(), 1)
    return torch.device(f"cuda:{index % device_count}")


def decode_mp3_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode raw audio bytes into a numpy array via soundfile."""
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return audio, int(sr)


def basename(inner_path: str) -> str:
    return Path(inner_path).name


def build_basename_index(group: pd.DataFrame) -> Dict[str, int]:
    """Map mp3 basename → DataFrame row index for a shard group."""
    return {
        basename(row["inner_path"]): idx
        for idx, row in group.iterrows()
    }


def stream_shard_parallel(
    shard_path: str,
    group: pd.DataFrame,
    parsers: List[Tuple[str, ASRParser]],
    model_names: Sequence[str],
    *,
    show_progress: bool,
) -> pd.DataFrame:
    """Stream tar once with all models loaded — one tar open, both models transcribe per file."""
    frame = group.copy()
    for name in model_names:
        if name not in frame.columns:
            frame[name] = ""

    basename_to_idx = build_basename_index(frame)
    progress = tqdm(
        total=len(frame),
        desc=Path(shard_path).name,
        disable=not show_progress,
    )

    with tarfile.open(shard_path, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".mp3"):
                continue
            idx = basename_to_idx.get(basename(member.name))
            if idx is None:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            try:
                audio, sr = decode_mp3_bytes(extracted.read())
            except Exception:
                LOGGER.exception("Failed to decode %s", member.name)
                for name, _ in parsers:
                    frame.at[idx, name] = ERROR_TOKEN
                progress.update(1)
                continue

            for name, parser in parsers:
                try:
                    frame.at[idx, name] = parser.transcribe((audio, sr))
                except Exception:
                    LOGGER.exception("Transcription failed for %s with %s", member.name, name)
                    frame.at[idx, name] = ERROR_TOKEN

            progress.update(1)

    progress.close()
    return frame


def stream_shard_sequential(
    shard_path: str,
    group: pd.DataFrame,
    model_names: Sequence[str],
    *,
    same_gpu: bool,
    show_progress: bool,
) -> pd.DataFrame:
    """Load one model at a time, streaming through the tar once per model.

    Uses less peak GPU memory than parallel mode at the cost of reading
    the tar N times (once per model).
    """
    frame = group.copy()
    for name in model_names:
        if name not in frame.columns:
            frame[name] = ""

    basename_to_idx = build_basename_index(frame)

    for model_idx, model_name in enumerate(model_names):
        device = resolve_device(model_idx, same_gpu=same_gpu)
        LOGGER.info("Loading %s on %s", model_name, device)
        parser = ASRParser(model_name, device=device)

        progress = tqdm(
            total=len(frame),
            desc=f"{Path(shard_path).name} [{model_name}]",
            disable=not show_progress,
        )

        with tarfile.open(shard_path, "r:*") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".mp3"):
                    continue
                idx = basename_to_idx.get(basename(member.name))
                if idx is None:
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    continue

                try:
                    audio, sr = decode_mp3_bytes(extracted.read())
                except Exception:
                    LOGGER.exception("Failed to decode %s", member.name)
                    frame.at[idx, model_name] = ERROR_TOKEN
                    progress.update(1)
                    continue

                try:
                    frame.at[idx, model_name] = parser.transcribe((audio, sr))
                except Exception:
                    LOGGER.exception("Transcription failed for %s with %s", member.name, model_name)
                    frame.at[idx, model_name] = ERROR_TOKEN

                progress.update(1)

        progress.close()
        del parser
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return frame


def compute_output_path(csv_path: Path, output: Optional[Path]) -> Path:
    if output is not None:
        return output
    return csv_path.with_name(f"transcribed_{csv_path.name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()
    config = load_config(args.config)

    model_names = config.get("asr_models", [])
    if not model_names:
        raise ValueError("Configuration must include an 'asr_models' list.")

    df = pd.read_csv(args.csv)
    for col in ("wav", "shard", "inner_path", "text"):
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    shards = list(df.groupby("shard"))
    LOGGER.info(
        "Processing %d shard(s) with %d model(s) in %s mode",
        len(shards),
        len(model_names),
        "sequential" if args.sequential else "parallel",
    )

    results: List[pd.DataFrame] = []

    if args.sequential:
        # Models loaded one at a time — tar opened once per model per shard.
        for shard_path, group in shards:
            LOGGER.info("Shard: %s (%d utterances)", shard_path, len(group))
            transcribed = stream_shard_sequential(
                shard_path,
                group,
                model_names,
                same_gpu=args.same_gpu,
                show_progress=not args.no_progress,
            )
            results.append(transcribed)
    else:
        # All models loaded upfront — tar opened once per shard.
        parsers: List[Tuple[str, ASRParser]] = []
        for idx, name in enumerate(model_names):
            device = resolve_device(idx, same_gpu=args.same_gpu)
            LOGGER.info("Loading %s on %s", name, device)
            parsers.append((name, ASRParser(name, device=device)))

        for shard_path, group in shards:
            LOGGER.info("Shard: %s (%d utterances)", shard_path, len(group))
            transcribed = stream_shard_parallel(
                shard_path,
                group,
                parsers,
                model_names,
                show_progress=not args.no_progress,
            )
            results.append(transcribed)

    combined = pd.concat(results, ignore_index=False)

    if not args.skip_agreement:
        LOGGER.info("Computing agreement metrics")
        combined = enrich_dataframe(
            combined,
            emilia_text_column="text",
            whisper_column="whisper",
            phi4_column="phi4",
            normalize_fn=ASRParser.normalize_text,
            show_progress=not args.no_progress,
        )

    output_path = compute_output_path(args.csv, args.output)
    LOGGER.info("Writing results to %s", output_path)
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
