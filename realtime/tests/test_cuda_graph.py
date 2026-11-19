"""A voice converted through a recorded CUDA graph sounds the same as block by block.

The graph replays one block's kernels with the stream's state updated in place;
the pitch settings live in a tensor the graph reads, so changing them carries
over without recording again. Skipped without CUDA or a CausalVC voice.
"""

import unittest

import numpy as np
import torch

from infer import causalvc_backend
from tests.test_causalvc_backend import find_voice

RATE = 48000


def speech_like(seconds=3.0):
    t = np.arange(int(RATE * seconds)) / RATE
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)  # a gliding voice
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t))
    return torch.from_numpy((0.05 * x).astype(np.float32))


@unittest.skipUnless(torch.cuda.is_available() and find_voice(), "needs CUDA and a CausalVC voice")
class CudaGraphTest(unittest.TestCase):
    def convert(self, graphs, key, change_to=None):
        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        voice = causalvc_backend.CausalVC(key, str(find_voice().model_path), device, cuda_graphs=graphs)
        block = 960 * 2
        voice.warm(block)
        if not graphs:  # the graph runtime starts from four seconds of silence: give this one the same start
            voice.vc.prepare(block)
            for _ in range(13):
                voice.vc.process(torch.zeros(32 * 480, device=device))
            for _ in range(3):
                voice.vc.process(torch.zeros(block, device=device))
        self.assertEqual(voice.uses_cuda_graph, graphs)
        x = speech_like().to(device)
        torch.manual_seed(0)  # the excitation's noise
        out = []
        for i in range(0, x.shape[0] - block + 1, block):
            if change_to is not None and i == x.shape[0] // 2 // block * block:
                voice.change_key(change_to)
            out.append(voice.process(x[i : i + block]))
        return torch.cat(out).cpu()

    def test_graph_output_matches_block_by_block(self):
        a, b = self.convert(True, 0.0), self.convert(False, 0.0)
        self.assertLess(float((a - b).pow(2).mean().sqrt()), 1e-3 * float(b.pow(2).mean().sqrt()))

    def test_pitch_changes_reach_a_recorded_graph(self):
        a, b = self.convert(True, 0.0, change_to=5.0), self.convert(False, 0.0, change_to=5.0)
        self.assertLess(float((a - b).pow(2).mean().sqrt()), 1e-3 * float(b.pow(2).mean().sqrt()))
        same = self.convert(True, 0.0)
        half = a.shape[0] // 2 + RATE // 5
        self.assertGreater(float((a[half:] - same[half:]).abs().max()), 1e-2)  # the change was heard

    def test_the_index_slider_reaches_a_recorded_graph(self):
        """With an index, rate 0 sounds as without one, a set rate matches block by block, and
        moving the slider changes the output without recording again."""
        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        torch.manual_seed(2)
        frames = torch.randn(512, 768)

        def run(graphs, rate, index=True, change_to=None):
            voice = causalvc_backend.CausalVC(0.0, str(find_voice().model_path), device, cuda_graphs=graphs)
            if index:
                voice.vc.set_index(frames)
            voice.change_index_rate(rate)
            voice.warm(1920)
            if not graphs:
                voice.vc.prepare(1920)
                for _ in range(13):
                    voice.vc.process(torch.zeros(32 * 480, device=device))
                for _ in range(3):
                    voice.vc.process(torch.zeros(1920, device=device))
            x = speech_like(2.0).to(device)
            torch.manual_seed(0)
            out = []
            for i in range(0, x.shape[0] - 1919, 1920):
                if change_to is not None and i == x.shape[0] // 2 // 1920 * 1920:
                    voice.change_index_rate(change_to)
                out.append(voice.process(x[i : i + 1920]))
            return torch.cat(out).cpu()

        rms = lambda t: float(t.pow(2).mean().sqrt())  # noqa: E731
        plain = run(True, 0.0, index=False)
        self.assertLess(rms(run(True, 0.0) - plain), 1e-3 * rms(plain))
        graph, eager = run(True, 0.5), run(False, 0.5)
        self.assertLess(rms(graph - eager), 1e-3 * rms(eager))
        self.assertGreater(rms(graph - plain), 1e-2 * rms(plain))
        moved = run(True, 0.0, change_to=0.5)
        half = moved.shape[0] // 2 + RATE // 5
        self.assertGreater(float((moved[half:] - plain[half:]).abs().max()), 1e-2)

    def test_reset_starts_the_stream_again(self):
        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        voice = causalvc_backend.CausalVC(0.0, str(find_voice().model_path), device, cuda_graphs=True)
        voice.warm(1920)
        x = speech_like(1.0).to(device)
        runs = []
        for _ in range(2):
            voice.reset_stream_state()
            torch.manual_seed(0)
            runs.append(torch.cat([voice.process(x[i : i + 1920]) for i in range(0, x.shape[0] - 1919, 1920)]).cpu())
        self.assertLess(float((runs[0] - runs[1]).abs().max()), 1e-4)

    def test_expressive_and_tone_decoder_records_cuda_graph(self):
        import tempfile
        from pathlib import Path
        from causalvc.decoder.model import Decoder
        from causalvc.fast import FastStreamingVC
        from causalvc.voice import load

        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        spec = load(find_voice().model_path)
        ck = torch.load(spec["decoder"], map_location="cpu")
        cfg = {**ck["cfg"], "tone": 2, "expressive": True}
        dec = Decoder(len(ck["speakers"]), **cfg)
        own = dec.state_dict()
        for k, v in ck["generator"].items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
        dec.load_state_dict(own)
        with torch.no_grad():
            dec.tone.weight.fill_(0.02)
            dec.tone.bias.fill_(0.1)
            dec.f0_proj.weight.fill_(0.02)
            dec.reg_cond.weight.fill_(0.02)
        with tempfile.TemporaryDirectory() as tmp:
            dec_path = str(Path(tmp) / "decoder.pt")
            torch.save({"generator": dec.state_dict(), "cfg": cfg, "speakers": ck["speakers"], "step": 1}, dec_path)
            x = speech_like(1.5).to(device)
            graph_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), spec["encoder"], graphs=True)
            graph_vc.set_voice(spec["speaker"], semitones=4.0)
            graph_vc.prepare(960)
            self.assertIsNotNone(graph_vc.graph)
            eager_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), spec["encoder"], graphs=False)
            eager_vc.set_voice(spec["speaker"], semitones=4.0)
            for _ in range(13):
                eager_vc.process(torch.zeros(32 * 480, device=device))
            for _ in range(3):
                eager_vc.process(torch.zeros(960, device=device))
            torch.manual_seed(0)
            yg = torch.cat([graph_vc.process(x[i : i + 960]) for i in range(0, len(x), 960)]).cpu()
            torch.manual_seed(0)
            ye = torch.cat([eager_vc.process(x[i : i + 960]) for i in range(0, len(x), 960)]).cpu()
            self.assertLess(float((yg - ye).pow(2).mean().sqrt()), 1e-3 * float(ye.pow(2).mean().sqrt()))


if __name__ == "__main__":
    unittest.main()
