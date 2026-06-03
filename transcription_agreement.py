"""Helper utilities to compute transcription agreement metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


SPECIAL_TOKEN_PATTERN = ("<", ">")


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate using a simple Levenshtein distance."""

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return float(bool(hyp_tokens))

    previous_row = list(range(len(hyp_tokens) + 1))
    for i, ref_word in enumerate(ref_tokens, start=1):
        current_row = [i]
        for j, hyp_word in enumerate(hyp_tokens, start=1):
            substitutions = previous_row[j - 1] + (ref_word != hyp_word)
            insertions = current_row[j - 1] + 1
            deletions = previous_row[j] + 1
            current_row.append(min(substitutions, insertions, deletions))
        previous_row = current_row

    return previous_row[-1] / len(ref_tokens)


def symmetric_wer(a: str, b: str) -> float:
    """Return the minimum of wer(a, b) and wer(b, a)."""
    return min(word_error_rate(a, b), word_error_rate(b, a))


def is_special_token(text: str) -> bool:
    return text.startswith(SPECIAL_TOKEN_PATTERN[0]) and text.endswith(
        SPECIAL_TOKEN_PATTERN[1]
    )


def majority_vote(
    emilia: str,
    whisper: str,
    phi4: str,
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    """Given three normalized transcriptions, return the majority transcript.

    Majority is defined as the transcript shared (lowest WER) by at least
    two of the three.  All three pairwise WER scores are always returned so
    callers can inspect the full distribution.

    Returns
    -------
    majority_transcript : str or None
        The transcript that represents the majority, or None if all three differ.
    wer_emilia_whisper : float
    wer_emilia_phi4 : float
    wer_whisper_phi4 : float
    """
    wer_ew = symmetric_wer(emilia, whisper)
    wer_ep = symmetric_wer(emilia, phi4)
    wer_wp = symmetric_wer(whisper, phi4)

    # The pair with the lowest WER is the "most agreed" pair.
    best_pair = min(
        [("emilia_whisper", wer_ew), ("emilia_phi4", wer_ep), ("whisper_phi4", wer_wp)],
        key=lambda x: x[1],
    )

    pair_name = best_pair[0]
    if pair_name == "emilia_whisper":
        majority = emilia  # both say the same thing; use emilia text as canonical
    elif pair_name == "emilia_phi4":
        majority = emilia
    else:
        majority = whisper  # whisper and phi4 agree; use whisper as canonical

    return majority, wer_ew, wer_ep, wer_wp


def enrich_dataframe(
    df: pd.DataFrame,
    emilia_text_column: str,
    whisper_column: str,
    phi4_column: str,
    normalize_fn,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Add agreement-related columns to *df*.

    New columns added
    -----------------
    emilia_normalized     : normalized Emilia text
    majority_transcript   : transcript agreed on by at least 2/3
    emilia_agrees         : True if Emilia is part of the majority pair, else False
    wer_emilia_whisper    : WER between normalized Emilia and Whisper output
    wer_emilia_phi4       : WER between normalized Emilia and Phi-4 output
    wer_whisper_phi4      : WER between Whisper and Phi-4 outputs
    """
    frame = df.copy()

    new_cols = [
        "emilia_normalized",
        "majority_transcript",
        "emilia_agrees",
        "wer_emilia_whisper",
        "wer_emilia_phi4",
        "wer_whisper_phi4",
    ]
    for col in new_cols:
        if col not in frame.columns:
            frame[col] = None

    iterator = frame.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=frame.shape[0], desc="Agreement")

    for idx, row in iterator:
        emilia_raw = row.get(emilia_text_column, "")
        whisper_out = row.get(whisper_column, "")
        phi4_out = row.get(phi4_column, "")

        # Skip rows with missing or special-token outputs.
        if not isinstance(emilia_raw, str) or not emilia_raw.strip():
            continue
        if not isinstance(whisper_out, str) or is_special_token(whisper_out):
            continue
        if not isinstance(phi4_out, str) or is_special_token(phi4_out):
            continue

        emilia_norm = normalize_fn(emilia_raw)
        frame.at[idx, "emilia_normalized"] = emilia_norm

        majority, wer_ew, wer_ep, wer_wp = majority_vote(
            emilia_norm, whisper_out, phi4_out
        )

        frame.at[idx, "majority_transcript"] = majority
        frame.at[idx, "wer_emilia_whisper"] = round(wer_ew, 4)
        frame.at[idx, "wer_emilia_phi4"] = round(wer_ep, 4)
        frame.at[idx, "wer_whisper_phi4"] = round(wer_wp, 4)

        # emilia_agrees = True when emilia is part of the closest-matching pair.
        best_pair_wer = min(wer_ew, wer_ep, wer_wp)
        emilia_in_best = (wer_ew == best_pair_wer) or (wer_ep == best_pair_wer)
        frame.at[idx, "emilia_agrees"] = emilia_in_best

    return frame
