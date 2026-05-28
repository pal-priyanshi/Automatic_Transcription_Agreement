"""Generate ASR transcripts for a dataset and optionally compute agreement scores."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import yaml
from pydub import AudioSegment
from tqdm import tqdm

from audio_path_resolver import resolved_audio_path
from asr_model_parser import ASRParser
from transcription_agreement import enrich_dataframe

LOGGER = logging.getLogger(__name__)
SILENCE_TOKEN = "<Sil>"
ERROR_TOKEN = "<Error>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV file containing a 'wav' column.")
    parser.add_argument(
        "config",
        type=Path,
        help="YAML file with 'asr_models' and 'threshold' entries.",
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
        help="Load models one at a time to reduce GPU memory pressure.",
    )
    parser.add_argument(
        "--skip-agreement",
        action="store_true",
        help="Only store raw transcripts without computing agreement columns.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-50.0,
        help="dBFS threshold used to treat an utterance as silence.",
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


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {', '.join(missing)}")
    return list(columns)


def resolve_device(index: int, *, same_gpu: bool) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if same_gpu:
        return torch.device("cuda:0")
    device_count = max(torch.cuda.device_count(), 1)
    return torch.device(f"cuda:{index % device_count}")


def detect_silence(audio_path: str | Path, threshold: float) -> bool:
    try:
        with resolved_audio_path(audio_path) as resolved_path:
            audio = AudioSegment.from_file(resolved_path)
    except FileNotFoundError:
        LOGGER.warning("Audio file %s not found; treating as non-silence.", audio_path)
        return False
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to load %s for silence detection.", audio_path)
        return False
    return audio.dBFS < threshold


def compute_silence_flags(
    df: pd.DataFrame, threshold: float, *, show_progress: bool
) -> Dict[int, bool]:
    iterator = df["wav"].items()
    if show_progress:
        iterator = tqdm(iterator, total=df.shape[0], desc="Silence detection")

    flags: Dict[int, bool] = {}
    for idx, path in iterator:
        flags[idx] = detect_silence(path, threshold)
    return flags


def ensure_output_columns(frame: pd.DataFrame, model_names: Sequence[str]) -> None:
    for model in model_names:
        if model not in frame.columns:
            frame[model] = ""


def transcribe_parallel(
    df: pd.DataFrame,
    model_names: Sequence[str],
    silence_flags: Dict[int, bool],
    *,
    same_gpu: bool,
    show_progress: bool,
) -> pd.DataFrame:
    parsers: List[Tuple[str, ASRParser]] = []
    for idx, model_name in enumerate(model_names):
        device = resolve_device(idx, same_gpu=same_gpu)
        LOGGER.info("Loading %s on %s", model_name, device)
        parsers.append((model_name, ASRParser(model_name, device=device)))

    frame = df.copy()
    ensure_output_columns(frame, model_names)

    iterator = frame.index
    if show_progress:
        iterator = tqdm(iterator, total=frame.shape[0], desc="Utterances")

    for idx in iterator:
        if silence_flags.get(idx, False):
            for model_name, _ in parsers:
                frame.at[idx, model_name] = SILENCE_TOKEN
            continue

        audio_path = frame.at[idx, "wav"]
        for model_name, parser in parsers:
            try:
                frame.at[idx, model_name] = parser.transcribe(audio_path)
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "Transcription failed for %s with model %s", audio_path, model_name
                )
                frame.at[idx, model_name] = ERROR_TOKEN
    return frame


def transcribe_sequential(
    df: pd.DataFrame,
    model_names: Sequence[str],
    silence_flags: Dict[int, bool],
    *,
    same_gpu: bool,
    show_progress: bool,
) -> pd.DataFrame:
    frame = df.copy()
    ensure_output_columns(frame, model_names)

    for idx, is_silence in silence_flags.items():
        if is_silence:
            for model_name in model_names:
                frame.at[idx, model_name] = SILENCE_TOKEN

    for model_idx, model_name in enumerate(model_names):
        device = resolve_device(model_idx, same_gpu=same_gpu)
        LOGGER.info("Processing %s on %s", model_name, device)
        parser = ASRParser(model_name, device=device)

        iterator = frame.index
        if show_progress:
            iterator = tqdm(iterator, total=frame.shape[0], desc=model_name)

        for idx in iterator:
            if silence_flags.get(idx, False):
                continue
            audio_path = frame.at[idx, "wav"]
            try:
                frame.at[idx, model_name] = parser.transcribe(audio_path)
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "Transcription failed for %s with model %s", audio_path, model_name
                )
                frame.at[idx, model_name] = ERROR_TOKEN

        del parser
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return frame


def compute_output_path(csv_path: Path, output: Path | None) -> Path:
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

    threshold = float(config.get("threshold", 0.0))

    df = pd.read_csv(args.csv)
    ensure_columns(df, ["wav"])

    LOGGER.info("Detecting silent utterances")
    silence_flags = compute_silence_flags(
        df, args.silence_threshold, show_progress=not args.no_progress
    )

    LOGGER.info(
        "Generating transcripts with %d model(s) using %s scheduling",
        len(model_names),
        "sequential" if args.sequential else "parallel",
    )

    if args.sequential:
        transcribed = transcribe_sequential(
            df,
            model_names,
            silence_flags,
            same_gpu=args.same_gpu,
            show_progress=not args.no_progress,
        )
    else:
        transcribed = transcribe_parallel(
            df,
            model_names,
            silence_flags,
            same_gpu=args.same_gpu,
            show_progress=not args.no_progress,
        )

    if not args.skip_agreement:
        LOGGER.info("Computing agreement metrics")
        transcribed = enrich_dataframe(
            transcribed,
            model_names,
            threshold,
            show_progress=not args.no_progress,
        )

    output_path = compute_output_path(args.csv, args.output)
    LOGGER.info("Writing results to %s", output_path)
    transcribed.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
