"""Utilities for loading and running multiple ASR backends with a unified API."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    GenerationConfig,
    pipeline,
)

from audio_path_resolver import resolved_audio_array

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, Path]
AudioInput = Union[PathLike, Tuple[np.ndarray, int]]  # path or (array, sample_rate)
AudioBatch = List[Tuple[np.ndarray, int]]             # list of (array, sample_rate)


class ASRParser:
    """Wrapper around a collection of pre-trained ASR models.

    Parameters
    ----------
    model_name:
        Identifier of the model to load. Supported values: ``"whisper"``, ``"phi4"``.
    device:
        Optional device hint (e.g. ``"cuda:0"``). Defaults to CUDA if available.
    """

    _MODEL_LOADERS: Dict[str, str] = {
        "whisper": "_load_whisper",
        "phi4": "_load_phi4",
    }

    def __init__(self, model_name: str, device: Optional[Union[str, torch.device]] = None) -> None:
        self.model_name = model_name.lower()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.processor = None
        self.pipe = None
        self.generation_config = None
        self.prompt = None

        loader_name = self._MODEL_LOADERS.get(self.model_name)
        if loader_name is None:
            raise NotImplementedError(
                f"The model '{model_name}' is not supported. "
                f"Supported models: {', '.join(self.supported_models())}"
            )

        getattr(self, loader_name)()
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: AudioInput,
        *,
        generate_kwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> str:
        """Transcribe a single audio input.

        Parameters
        ----------
        audio:
            Either a filesystem path / tar URI string, or a pre-loaded
            ``(numpy_array, sample_rate)`` tuple.
        """
        results = self.transcribe_batch([self._resolve(audio)], generate_kwargs=generate_kwargs)
        return results[0]

    def transcribe_batch(
        self,
        batch: AudioBatch,
        *,
        generate_kwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> List[str]:
        """Transcribe a batch of audio inputs, returning one string per input.

        Parameters
        ----------
        batch:
            List of ``(numpy_array, sample_rate)`` tuples.
        """
        if not batch:
            return []

        if self.model_name == "whisper":
            return self._transcribe_batch_whisper(batch, generate_kwargs)
        if self.model_name == "phi4":
            return self._transcribe_batch_phi4(batch, generate_kwargs)

        raise RuntimeError(f"No transcription routine registered for '{self.model_name}'.")

    @staticmethod
    def normalize_text(transcription: str) -> str:
        """Normalize transcription output for easier comparison."""
        text = transcription.upper()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\b(UM|UH|AH|ER)\b", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def supported_models(cls) -> tuple:
        return tuple(sorted(cls._MODEL_LOADERS.keys()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve(self, audio: AudioInput) -> Tuple[np.ndarray, int]:
        """Convert a path or (array, sr) tuple into (array, sr)."""
        if isinstance(audio, tuple):
            return audio
        if not Path(str(audio)).exists() and not str(audio).startswith("tar://"):
            raise FileNotFoundError(f"Audio file '{audio}' was not found.")
        return resolved_audio_array(audio)

    def _load_whisper(self) -> None:
        model_id = "openai/whisper-large-v3"
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def _load_phi4(self) -> None:
        model_path = "microsoft/Phi-4-multimodal-instruct"
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        # Use Flash Attention 2 if available (pip install flash-attn --no-build-isolation).
        # Falls back to "eager" (slow) if not installed — install it on the cluster for
        # a significant speedup on Phi-4's long audio conformer sequences.
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            LOGGER.warning(
                "flash-attn not installed — using eager attention (slow). "
                "Run: pip install flash-attn --no-build-isolation"
            )
            attn_impl = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            _attn_implementation=attn_impl,
        ).to(self.device)

        self.generation_config = GenerationConfig.from_pretrained(model_path)

        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        speech_prompt = (
            "Based on the attached audio, generate a comprehensive "
            "text transcription of the spoken content"
        )
        self.prompt = (
            f"{user_prompt}<|audio_1|>{speech_prompt}"
            f"{prompt_suffix}{assistant_prompt}"
        )

    def _transcribe_batch_whisper(
        self,
        batch: AudioBatch,
        generate_kwargs: Optional[Dict],
    ) -> List[str]:
        """Pass the whole batch to the HuggingFace pipeline at once."""
        inputs = [{"array": audio, "sampling_rate": sr} for audio, sr in batch]
        kwargs = {"language": "en"}
        if generate_kwargs:
            kwargs.update(generate_kwargs)
        results = self.pipe(inputs, batch_size=len(inputs), generate_kwargs=kwargs)
        return [self.normalize_text(r["text"]) for r in results]

    def _transcribe_batch_phi4(
        self,
        batch: AudioBatch,
        generate_kwargs: Optional[Dict],
    ) -> List[str]:
        """Process phi4 batch — pad inputs and decode all at once."""
        inputs = self.processor(
            text=[self.prompt] * len(batch),
            audios=batch,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        kwargs = {
            # Emilia clips are typically 2–30 s → well under 256 output tokens.
            # Keeping this low saves significant KV-cache memory during generation.
            "max_new_tokens": 256,
            "generation_config": self.generation_config,
            "num_logits_to_keep": 1,
        }
        if generate_kwargs:
            kwargs.update(generate_kwargs)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **kwargs)

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        transcriptions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [self.normalize_text(t) for t in transcriptions]


# Backwards compatibility for previous naming convention.
ASR_parser = ASRParser
