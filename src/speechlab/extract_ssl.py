"""Whisper-small encoder and HuBERT Base hidden states per segment.

Both models give 13 hidden states (pre-transformer output + 12 layers) at 50 frames/s.
Whisper pads every input to 30 s (1500 frames); we keep only the frames that cover
real audio, ceil(n_samples / 320), before pooling. HuBERT runs at batch size 1, so
no padding is involved.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from transformers import AutoFeatureExtractor, HubertModel, WhisperFeatureExtractor, WhisperModel

SR = 16000
SAMPLES_PER_FRAME = 320  # 20 ms at 16 kHz for both models
WHISPER_MAX_FRAMES = 1500


def whisper_valid_frames(n_samples: int) -> int:
    return min(WHISPER_MAX_FRAMES, math.ceil(n_samples / SAMPLES_PER_FRAME))


class SSLExtractor:
    def __init__(self, kind: str, model_id: str, revision: str):
        self.kind = kind
        if kind == "whisper":
            self.fe = WhisperFeatureExtractor.from_pretrained(model_id, revision=revision)
            self.model = WhisperModel.from_pretrained(model_id, revision=revision).encoder
        elif kind == "hubert":
            self.fe = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
            self.model = HubertModel.from_pretrained(model_id, revision=revision)
        else:
            raise ValueError(kind)
        self.model.eval()

    @torch.inference_mode()
    def hidden_states(self, wav: np.ndarray) -> torch.Tensor:
        """Returns float32 tensor [13, T, 768] restricted to frames covering real audio."""
        if self.kind == "whisper":
            feats = self.fe(wav, sampling_rate=SR, return_tensors="pt").input_features
            hs = self.model(feats, output_hidden_states=True).hidden_states
            t = whisper_valid_frames(len(wav))
            return torch.stack([h[0, :t] for h in hs])
        inputs = self.fe(wav, sampling_rate=SR, return_tensors="pt")
        hs = self.model(inputs.input_values, output_hidden_states=True).hidden_states
        return torch.stack([h[0] for h in hs])


def pool(hs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Mean and (population) std over frames -> two [13, 768] float32 arrays."""
    return hs.mean(dim=1).numpy(), hs.std(dim=1, unbiased=False).numpy()
