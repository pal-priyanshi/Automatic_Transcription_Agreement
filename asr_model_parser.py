"""Utilities for loading and running multiple ASR backends with a unified API."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

import soundfile as sf
import torch
from speechbrain.inference.ASR import EncoderDecoderASR
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    GenerationConfig,
    pipeline,
)

from audio_path_resolver import is_tar_uri, resolved_audio_path

LOGGER = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ASRParser:
    """Wrapper around a collection of pre-trained ASR models.

    Parameters
    ----------
    model_name:
        Identifier of the model to load. Supported values can be obtained via
        :meth:`supported_models`.
    device:
        Optional device hint (e.g. ``"cuda:0"``). If not provided the parser
        automatically chooses ``cuda`` when available, otherwise ``cpu``.
    """

    _MODEL_LOADERS: Dict[str, str] = {
        "whisper": "_load_whisper",
        "phi4": "_load_phi4",
        "conformer": "_load_conformer",
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
        audio_file: PathLike,
        *,
        generate_kwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> str:
        """Run transcription with the configured backend.

        Parameters
        ----------
        audio_file:
            Path to an audio file readable by the underlying model.
        generate_kwargs:
            Optional keyword arguments forwarded to generative models (e.g.
            Whisper, Phi-4).
        """

        if not is_tar_uri(audio_file) and not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file '{audio_file}' was not found.")

        with resolved_audio_path(audio_file) as path:
            if self.model_name == "whisper":
                return self._transcribe_with_whisper(path, generate_kwargs)
            if self.model_name == "phi4":
                return self._transcribe_with_phi4(path, generate_kwargs)
            if self.model_name == "conformer":
                return self._transcribe_with_conformer(path)

        raise RuntimeError(
            f"No transcription routine registered for model '{self.model_name}'."
        )

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
        """Return a tuple of the recognised model identifiers."""

        return tuple(sorted(cls._MODEL_LOADERS.keys()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_conformer(self) -> None:
        run_device = str(self.device)
        self.model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-conformer-largescaleasr",
            savedir="pretrained_models/asr-conformer-largescaleasr",
            run_opts={"device": run_device},
        )

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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            _attn_implementation="eager",
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

    def _transcribe_with_whisper(
        self,
        audio_file: Path,
        generate_kwargs: Optional[Dict[str, Union[int, float]]],
    ) -> str:
        kwargs = generate_kwargs or {}
        result = self.pipe(str(audio_file), generate_kwargs=kwargs)
        return self.normalize_text(result["text"])

    def _transcribe_with_phi4(
        self,
        audio_file: Path,
        generate_kwargs: Optional[Dict[str, Union[int, float]]],
    ) -> str:
        audio, sample_rate = sf.read(str(audio_file))
        inputs = self.processor(
            text=self.prompt,
            audios=[(audio, sample_rate)],
            return_tensors="pt",
        ).to(self.device)

        kwargs = {"max_new_tokens": 1000, "generation_config": self.generation_config, "num_logits_to_keep": 1}
        if generate_kwargs:
            kwargs.update(generate_kwargs)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **kwargs)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self.normalize_text(transcription)

    def _transcribe_with_conformer(self, audio_file: Path) -> str:
        predicted_text = self.model.transcribe_file(str(audio_file))
        return self.normalize_text(predicted_text)


# Backwards compatibility for previous naming convention.
ASR_parser = ASRParser
