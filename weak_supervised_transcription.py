"""Generate ASR transcripts for Emilia shards and compute agreement scores.

Usage
-----
# Single shard (for testing):
    python weak_supervised_transcription.py \\
        /data/EN-B000000.tar configs/config.yaml --output results.csv

# All shards via glob:
    python weak_supervised_transcription.py \\
        '/data/EN-B*.tar' configs/config.yaml --output results_all.csv --resume

Audio and metadata are read directly from each tar's .mp3 / .json pairs —
no separate manifest CSV is required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob as glob_module
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import webdataset as wds
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from asr_model_parser import ASRParser
from transcription_agreement import enrich_dataframe

LOGGER = logging.getLogger(__name__)
ERROR_TOKEN = "<Error>"
DEFAULT_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "shards", nargs="+",
        help="Tar shard path(s) or glob pattern(s), e.g. '/data/EN-B*.tar'.",
    )
    parser.add_argument("config", type=Path, help="YAML config with 'asr_models' list.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Samples per GPU batch (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--same-gpu", action="store_true",
        help="Put all models on cuda:0 instead of spreading across devices.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Load one model at a time (lower peak GPU memory, streams shard once per model).",
    )
    parser.add_argument("--skip-agreement", action="store_true",
                        help="Skip agreement scoring (faster, transcriptions only).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress tqdm bars.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to an existing output CSV, skipping shards already present.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Shard discovery & resume
# ---------------------------------------------------------------------------

def expand_shards(patterns: List[str]) -> List[str]:
    """Expand glob patterns into a sorted, deduplicated list of shard paths."""
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob_module.glob(pattern))
        if not matched:
            LOGGER.warning("No files matched pattern: %s", pattern)
        paths.extend(matched)
    return sorted(set(paths))


def load_completed_shards(output_path: Path) -> set:
    """Return the set of shard paths already written to output_path."""
    if not output_path.exists():
        return set()
    try:
        return set(pd.read_csv(output_path, usecols=["shard"])["shard"].unique())
    except Exception:
        LOGGER.warning("Could not read %s for resume; starting fresh.", output_path)
        return set()


# ---------------------------------------------------------------------------
# WebDataset helpers
# ---------------------------------------------------------------------------

def decode_sample(sample: dict) -> Optional[dict]:
    """Convert raw WebDataset bytes dict into a decoded (audio, meta) dict.

    Returns None on any failure so .select() can filter it out silently.

    Expected tar contents per sample:
        <key>.mp3   – audio file
        <key>.json  – Emilia metadata (text, speaker, duration, language, dnsmos, …)
    """
    try:
        audio_bytes = sample.get("mp3") or sample.get("wav")
        if not audio_bytes:
            LOGGER.warning("No audio found for sample %s", sample.get("__key__"))
            return None
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        meta = json.loads(sample["json"].decode("utf-8"))
        return {
            "audio":   (np.asarray(audio, dtype=np.float32), int(sr)),
            "meta":    meta,
            "__key__": sample["__key__"],
            "__url__": sample["__url__"],
        }
    except Exception as exc:
        LOGGER.warning("Skipping sample %s: %s", sample.get("__key__", "?"), exc)
        return None


def _make_dataloader(shard_path: str) -> DataLoader:
    """Single-shard WebDataset DataLoader.

    num_workers=1  – one worker process for I/O + decoding, completely
                     overlapped with GPU inference in the main process.
    prefetch_factor=4 – worker stays ~4 samples ahead; enough buffer for a
                        full batch to be ready when inference finishes.
    """
    dataset = (
        wds.WebDataset(shard_path, shardshuffle=False)
           .map(decode_sample)
           .select(lambda x: x is not None)
    )
    return DataLoader(
        dataset,
        num_workers=1,
        batch_size=None,       # we batch manually (variable-length audio)
        collate_fn=lambda x: x,
        prefetch_factor=4,
    )


def _build_row(sample: dict, shard_path: str, model_names: Sequence[str]) -> dict:
    """Build an output row dict from a decoded WebDataset sample.

    Field mapping from Emilia JSON:
        text      → text
        speaker   → spk_id    (fallback: spk_id)
        duration  → duration
        language  → language
        dnsmos    → dnsmos
    Adjust the .get() keys here if your JSON uses different field names.
    """
    meta = sample["meta"]
    row: dict = {
        "source_id": sample["__key__"],
        "shard":     shard_path,
        "text":      meta.get("text", ""),
        "spk_id":    meta.get("speaker", meta.get("spk_id", "")),
        "duration":  meta.get("duration", ""),
        "language":  meta.get("language", ""),
        "dnsmos":    meta.get("dnsmos", ""),
    }
    for name in model_names:
        row[name] = ""
    return row


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resolve_device(index: int, *, same_gpu: bool) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if same_gpu:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{index % max(torch.cuda.device_count(), 1)}")


def _transcribe_with_oom_retry(
    parser: ASRParser,
    batch_audio: List[Tuple[np.ndarray, int]],
    model_name: str,
) -> List[str]:
    """Transcribe batch; on CUDA OOM halve the batch and retry recursively.

    Keeps halving until the sub-batch fits or a single sample still OOMs
    (returned as ERROR_TOKEN in that case).
    """
    if not batch_audio:
        return []
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return parser.transcribe_batch(batch_audio)
    except torch.cuda.OutOfMemoryError:
        if len(batch_audio) == 1:
            LOGGER.error("OOM on single sample for %s — marking as error.", model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [ERROR_TOKEN]
        half = len(batch_audio) // 2
        LOGGER.warning(
            "OOM for %s at batch_size=%d — retrying as two halves of %d.",
            model_name, len(batch_audio), half,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            _transcribe_with_oom_retry(parser, batch_audio[:half], model_name)
            + _transcribe_with_oom_retry(parser, batch_audio[half:], model_name)
        )


def _flush_batch_parallel(
    batch_rows: List[dict],
    batch_audio: List[Tuple[np.ndarray, int]],
    parsers: List[Tuple[str, ASRParser]],
) -> None:
    """Run all models on a batch concurrently across GPUs.

    Each model runs in its own thread.  GPU operations release the GIL, so
    models on different CUDA devices genuinely overlap — GPU 0 and GPU 1
    both run at the same time instead of waiting for each other.
    """
    def _run(name: str, parser: ASRParser):
        return name, _transcribe_with_oom_retry(parser, batch_audio, name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parsers)) as pool:
        futures = [pool.submit(_run, name, parser) for name, parser in parsers]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, results = future.result()
                for row, text in zip(batch_rows, results):
                    row[name] = text
            except Exception:
                LOGGER.exception("Batch transcription failed for a model.")
                for row in batch_rows:
                    for name, _ in parsers:
                        if name not in row or not row[name]:
                            row[name] = ERROR_TOKEN


def _flush_batch_sequential(
    batch_audio: List[Tuple[np.ndarray, int]],
    batch_keys: List[str],
    parser: ASRParser,
    model_name: str,
    rows_by_key: Dict[str, dict],
    progress: tqdm,
) -> None:
    """Transcribe one model's batch; write results into rows_by_key in-place."""
    if not batch_audio:
        return
    try:
        results = _transcribe_with_oom_retry(parser, batch_audio, model_name)
        for key, text in zip(batch_keys, results):
            rows_by_key[key][model_name] = text
    except Exception:
        LOGGER.exception("Batch failed for %s", model_name)
        for key in batch_keys:
            rows_by_key[key][model_name] = ERROR_TOKEN
    progress.update(len(batch_keys))
    batch_audio.clear()
    batch_keys.clear()


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

def process_shard_parallel(
    shard_path: str,
    parsers: List[Tuple[str, ASRParser]],
    model_names: Sequence[str],
    batch_size: int,
    *,
    show_progress: bool,
) -> pd.DataFrame:
    """Stream shard once; run all models on each batch before moving to the next."""
    loader = _make_dataloader(shard_path)
    progress = tqdm(desc=Path(shard_path).name, unit="sample", disable=not show_progress)

    all_rows:   List[dict]                    = []
    batch_rows: List[dict]                    = []
    batch_audio: List[Tuple[np.ndarray, int]] = []

    def flush() -> None:
        if not batch_audio:
            return
        _flush_batch_parallel(batch_rows, batch_audio, parsers)
        all_rows.extend(batch_rows)
        progress.update(len(batch_rows))
        batch_rows.clear()
        batch_audio.clear()

    for sample in loader:
        batch_rows.append(_build_row(sample, shard_path, model_names))
        batch_audio.append(sample["audio"])
        if len(batch_audio) >= batch_size:
            flush()

    flush()  # remaining
    progress.close()
    return pd.DataFrame(all_rows)


def process_shard_sequential(
    shard_path: str,
    model_names: Sequence[str],
    batch_size: int,
    *,
    same_gpu: bool,
    show_progress: bool,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Stream shard once per model so only one model occupies GPU memory at a time.

    On the first model's pass the row dicts (metadata) are built and stored
    in rows_by_key.  Subsequent passes look up existing rows by __key__ and
    append their transcriptions.
    """
    rows_by_key: Dict[str, dict] = {}

    for model_idx, model_name in enumerate(model_names):
        device = resolve_device(model_idx, same_gpu=same_gpu)
        LOGGER.info("Loading %s on %s", model_name, device)
        parser = ASRParser(model_name, device=device)
        loader = _make_dataloader(shard_path)

        progress = tqdm(
            desc=f"{Path(shard_path).name} [{model_name}]",
            unit="sample",
            disable=not show_progress,
        )

        batch_audio: List[Tuple[np.ndarray, int]] = []
        batch_keys:  List[str]                    = []

        for sample in loader:
            key = sample["__key__"]

            if model_idx == 0:
                # First pass: build the row dict from metadata.
                rows_by_key[key] = _build_row(sample, shard_path, model_names)
            elif key not in rows_by_key:
                LOGGER.warning("Key %s absent from first pass — skipping.", key)
                continue

            batch_audio.append(sample["audio"])
            batch_keys.append(key)

            if len(batch_audio) >= batch_size:
                _flush_batch_sequential(
                    batch_audio, batch_keys, parser, model_name,
                    rows_by_key, progress,
                )

        # Flush remaining samples.
        _flush_batch_sequential(
            batch_audio, batch_keys, parser, model_name,
            rows_by_key, progress,
        )
        progress.close()

        del parser
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Save partial shard results after each model so progress is not lost
        # if the next model crashes.
        if checkpoint_path is not None:
            pd.DataFrame(list(rows_by_key.values())).to_csv(
                checkpoint_path, mode="w", header=True, index=False,
            )
            LOGGER.info("Checkpoint saved after %s → %s", model_name, checkpoint_path)

    return pd.DataFrame(list(rows_by_key.values()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    config = load_config(args.config)

    model_names: List[str] = config.get("asr_models", [])
    if not model_names:
        raise ValueError("Config must include a non-empty 'asr_models' list.")

    shard_paths = expand_shards(args.shards)
    if not shard_paths:
        raise FileNotFoundError("No shard files found for the given pattern(s).")

    # Resume: skip shards already present in the output CSV.
    done_shards = load_completed_shards(args.output) if args.resume else set()
    if done_shards:
        before = len(shard_paths)
        shard_paths = [s for s in shard_paths if s not in done_shards]
        LOGGER.info("Resume: skipping %d/%d already-completed shard(s).", len(done_shards), before)

    LOGGER.info(
        "Processing %d shard(s) | %d model(s) | %s mode | batch_size=%d",
        len(shard_paths), len(model_names),
        "sequential" if args.sequential else "parallel",
        args.batch_size,
    )

    appending   = args.resume and args.output.exists() and bool(done_shards)
    write_header = not appending

    # In parallel mode load all models once upfront.
    parsers: List[Tuple[str, ASRParser]] = []
    if not args.sequential:
        for idx, name in enumerate(model_names):
            device = resolve_device(idx, same_gpu=args.same_gpu)
            LOGGER.info("Loading %s on %s", name, device)
            parsers.append((name, ASRParser(name, device=device)))

    for shard_path in shard_paths:
        LOGGER.info("Shard: %s", shard_path)

        if args.sequential:
            transcribed = process_shard_sequential(
                shard_path, model_names, args.batch_size,
                same_gpu=args.same_gpu,
                show_progress=not args.no_progress,
                checkpoint_path=args.output,
            )
        else:
            transcribed = process_shard_parallel(
                shard_path, parsers, model_names, args.batch_size,
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

        transcribed.to_csv(
            args.output,
            mode="a" if appending else "w",
            header=write_header,
            index=False,
        )
        appending    = True
        write_header = False
        LOGGER.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
