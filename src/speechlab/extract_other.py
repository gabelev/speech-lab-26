"""eGeMAPS functionals (openSMILE) and text-model embeddings per segment."""

from __future__ import annotations

import numpy as np
import opensmile
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

SR = 16000


class EGeMAPS:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def __call__(self, wav: np.ndarray) -> pd.Series:
        return self.smile.process_signal(wav, SR).iloc[0]


class RobertaLayers:
    """Mean over non-special tokens for each of the 13 hidden states -> [13, 768]."""

    def __init__(self, model_id: str, revision: str, batch_size: int = 16):
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).eval()
        self.bs = batch_size

    @torch.inference_mode()
    def __call__(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i: i + self.bs], padding=True, truncation=True, max_length=512,
                           return_tensors="pt", return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            hs = torch.stack(self.model(**enc, output_hidden_states=True).hidden_states, dim=1)  # B,13,L,D
            mask = (enc["attention_mask"] * (1 - special)).float()
            # empty text -> fall back to all attended tokens (incl. <s></s>)
            empty = mask.sum(1) == 0
            mask[empty] = enc["attention_mask"][empty].float()
            m = mask[:, None, :, None]
            out.append(((hs * m).sum(2) / m.sum(2)).numpy())
        return np.concatenate(out)


class SentenceEmb:
    def __init__(self, model_id: str, revision: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, revision=revision, device="cpu")

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
