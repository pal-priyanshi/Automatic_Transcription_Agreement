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

DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV file containing required columns.")
    parser.add_argument("config", type=Path, help="YAML file with 'asr_models' entry.")
    parser.add_argument("--output", type=Path, help="Destination CSV path.")
    parser.add_argument(
        "--same-gpu", action="store_true",
        help="Load every model on the first CUDA device instead of spreading them.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Load one model at a time (lower peak GPU memory, reads tar once per model).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of audio files to transcribe per batch (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(index: int, *, same_gpu: bool) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if same_gpu:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{index % max(torch.cuda.device_count(), 1)}")


def decode_mp3_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return audio, int(sr)


def basename(inner_path: str) -> str:
    return Path(inner_path).name


def build_basename_index(group: pd.DataFrame) -> Dict[str, int]:
    return {basename(row["inner_path"]): idx for idx, row in group.iterrows()}


def flush_batch(
    batch_indices: List[int],
    batch_audio: List[Tuple[np.ndarray, int]],
    parsers: List[Tuple[str, ASRParser]],
    frame: pd.DataFrame,
) -> None:
    """Transcribe a collected batch with all models and write results into frame."""
    for name, parser in parsers:
        try:
            results = parser.transcribe_batch(batch_audio)
            for idx, text in zip(batch_indices, results):
                frame.at[idx, name] = text
        except Exception:
            LOGGER.exception("Batch transcription failed for model %s", name)
            for idx in batch_indices:
                frame.at[idx, name] = ERROR_TOKEN


def stream_shard_parallel(
    shard_path: str,
    group: pd.DataFrame,
    parsers: List[Tuple[str, ASRParser]],
    model_names: Sequence[str],
    batch_size: int,
    *,
    show_progress: bool,
) -> pd.DataFrame:
    """Open tar once, buffer audio into batches, transcribe with all models per batch."""
    frame = group.copy()
    for name in model_names:
        if name not in frame.columns:
            frame[name] = ""

    basename_to_idx = build_basename_index(frame)
    progress = tqdm(total=len(frame), desc=Path(shard_path).name, disable=not show_progress)

    batch_indices: List[int] = []
    batch_audio: List[Tuple[np.ndarray, int]] = []

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

            batch_indices.append(idx)
            batch_audio.append((audio, sr))

            if len(batch_audio) >= batch_size:
                flush_batch(batch_indices, batch_audio, parsers, frame)
                progress.update(len(batch_indices))
                batch_indices, batch_audio = [], []

    # flush remaining
    if batch_audio:
        flush_batch(batch_indices, batch_audio, parsers, frame)
        progress.update(len(batch_indices))

    progress.close()
    return frame


def stream_shard_sequential(
    shard_path: str,
    group: pd.DataFrame,
    model_names: Sequence[str],
    batch_size: int,
    *,
    same_gpu: bool,
    show_progress: bool,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load one model at a time, stream tar once per model, process in batches."""
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

        batch_indices: List[int] = []
        batch_audio: List[Tuple[np.ndarray, int]] = []

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

                batch_indices.append(idx)
                batch_audio.append((audio, sr))

                if len(batch_audio) >= batch_size:
                    try:
                        results = parser.transcribe_batch(batch_audio)
                        for i, text in zip(batch_indices, results):
                            frame.at[i, model_name] = text
                    except Exception:
                        LOGGER.exception("Batch failed for %s", model_name)
                        for i in batch_indices:
                            frame.at[i, model_name] = ERROR_TOKEN
                    progress.update(len(batch_indices))
                    batch_indices, batch_audio = [], []

        # flush remaining
        if batch_audio:
            try:
                results = parser.transcribe_batch(batch_audio)
                for i, text in zip(batch_indices, results):
                    frame.at[i, model_name] = text
            except Exception:
                LOGGER.exception("Batch failed for %s", model_name)
                for i in batch_indices:
                    frame.at[i, model_name] = ERROR_TOKEN
            progress.update(len(batch_indices))

        progress.close()
        del parser
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Save partial results after each model completes so progress is not lost
        # if a subsequent model crashes.
        if checkpoint_path is not None:
            frame.to_csv(
                checkpoint_path,
                mode="w" if model_idx == 0 else "w",
                header=True,
                index=False,
            )
            LOGGER.info("Checkpoint saved after %s → %s", model_name, checkpoint_path)

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
        "Processing %d shard(s) with %d model(s) in %s mode (batch_size=%d)",
        len(shards), len(model_names),
        "sequential" if args.sequential else "parallel",
        args.batch_size,
    )

    output_path = compute_output_path(args.csv, args.output)
    write_header = True  # first shard writes header, rest append

    if args.sequential:
        for shard_path, group in shards:
            LOGGER.info("Shard: %s (%d utterances)", shard_path, len(group))
            transcribed = stream_shard_sequential(
                shard_path, group, model_names, args.batch_size,
                same_gpu=args.same_gpu, show_progress=not args.no_progress,
                checkpoint_path=output_path,
            )
            if not args.skip_agreement:
                transcribed = enrich_dataframe(
                    transcribed,
                    emilia_text_column="text",
                    whisper_column="whisper",
                    phi4_column="phi4",
                    normalize_fn=ASRParser.normalize_text,
                    show_progress=not args.no_progress,
                )
            transcribed.to_csv(output_path, mode="w" if write_header else "a",
                               header=write_header, index=False)
            write_header = False
            LOGGER.info("Saved progress to %s", output_path)
    else:
        parsers: List[Tuple[str, ASRParser]] = []
        for idx, name in enumerate(model_names):
            device = resolve_device(idx, same_gpu=args.same_gpu)
            LOGGER.info("Loading %s on %s", name, device)
            parsers.append((name, ASRParser(name, device=device)))

        for shard_path, group in shards:
            LOGGER.info("Shard: %s (%d utterances)", shard_path, len(group))
            transcribed = stream_shard_parallel(
                shard_path, group, parsers, model_names, args.batch_size,
                show_progress=not args.no_progress,
            )
            if not args.skip_agreement:
                transcribed = enrich_dataframe(
                    transcribed,
                    emilia_text_column="text",
                    whisper_column="whisper",
                    phi4_column="phi4",
                    normalize_fn=ASRParser.normalize_text,
                    show_progress=not args.no_progress,
                )
            transcribed.to_csv(output_path, mode="w" if write_header else "a",
                               header=write_header, index=False)
            write_header = False
            LOGGER.info("Saved progress to %s", output_path)


if __name__ == "__main__":
    main()
