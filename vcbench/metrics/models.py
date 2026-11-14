"""Model-based metrics: speaker similarity, ASR error, F0 tracking, predicted MOS.

All models take 16 kHz mono float32 numpy arrays.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16000


class SpeakerEmbedder:
    """WavLM x-vector speaker verification embeddings (unit-norm)."""

    def __init__(self, device: str, model_id: str = "microsoft/wavlm-base-plus-sv"):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        x = self.fe([wav], sampling_rate=SR, return_tensors="pt", padding=True).to(self.device)
        e = self.model(**x).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def centroid(self, wavs: list[np.ndarray]) -> np.ndarray:
        c = np.mean([self.embed(w) for w in wavs], axis=0)
        return c / np.linalg.norm(c)


class Transcriber:
    """Whisper ASR with Whisper's own English text normaliser."""

    def __init__(self, device: str, model_id: str = "openai/whisper-large-v3-turbo"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.proc = WhisperProcessor.from_pretrained(model_id)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        )
        self.device = device

    def normalize(self, text: str) -> str:
        return self.proc.tokenizer.normalize(text)

    @torch.inference_mode()
    def transcribe(self, wavs: list[np.ndarray], batch_size: int = 16) -> list[str]:
        out: list[str] = []
        for i in range(0, len(wavs), batch_size):
            feats = self.proc(wavs[i : i + batch_size], sampling_rate=SR, return_tensors="pt").input_features
            ids = self.model.generate(
                feats.to(self.device, torch.float16), language="en", task="transcribe"
            )
            out += [self.normalize(t) for t in self.proc.batch_decode(ids, skip_special_tokens=True)]
        return out


class PitchTracker:
    """CREPE F0 at a 10 ms hop with a periodicity-based voicing decision."""

    def __init__(self, device: str, voiced_threshold: float = 0.5):
        self.device = device
        self.thr = voiced_threshold

    @torch.inference_mode()
    def track(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torchcrepe

        x = torch.from_numpy(wav)[None].to(self.device)
        f0, per = torchcrepe.predict(
            x, SR, hop_length=160, fmin=50.0, fmax=800.0, model="full",
            batch_size=1024, device=self.device, return_periodicity=True,
        )
        per = torchcrepe.filter.median(per, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        return f0[0].cpu().numpy(), (per[0] > self.thr).cpu().numpy()


def f0_agreement(src: tuple[np.ndarray, np.ndarray], out: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """(log-F0 Pearson correlation on mutually voiced frames, voicing error rate).

    Correlation ignores a constant transposition, so a pitch-shifted conversion
    that follows the source contour still scores ~1.
    """
    (f_s, v_s), (f_o, v_o) = src, out
    n = min(len(f_s), len(f_o))
    f_s, v_s, f_o, v_o = f_s[:n], v_s[:n], f_o[:n], v_o[:n]
    both = v_s & v_o
    vuv_err = float((v_s != v_o).mean())
    if both.sum() < 10:
        return float("nan"), vuv_err
    return float(np.corrcoef(np.log(f_s[both]), np.log(f_o[both]))[0, 1]), vuv_err


class MosPredictor:
    """UTMOS22 (strong) predicted naturalness MOS, 1-5."""

    def __init__(self, device: str):
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def score(self, wav: np.ndarray) -> float:
        return float(self.model(torch.from_numpy(wav)[None].to(self.device), SR)[0])


class PhoneRecognizer:
    """Phone sequence from a CTC recogniser with no language model.

    Whisper repairs slurred speech from context; this cannot, so smeared or
    dropped consonants show up as phone errors against the source audio.
    """

    def __init__(self, device: str, model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"):
        import json

        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        # The vocabulary is read directly: the bundled tokenizer needs espeak.
        vocab = json.loads(open(hf_hub_download(model_id, "vocab.json")).read())
        self.id2tok = {i: t for t, i in vocab.items()}
        self.skip = {vocab[t] for t in ("<pad>", "<s>", "</s>", "<unk>") if t in vocab}
        self.blank = vocab["<pad>"]
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def phones(self, wav: np.ndarray) -> str:
        """Space-separated phones (greedy CTC decode)."""
        x = self.fe([wav], sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        ids = self.model(x).logits[0].argmax(-1).tolist()
        out, prev = [], self.blank
        for i in ids:
            if i != prev and i not in self.skip:
                out.append(self.id2tok[i])
            prev = i
        return " ".join(out)
