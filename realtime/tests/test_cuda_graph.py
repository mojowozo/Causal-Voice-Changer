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

    def test_v5_antialiased_snake_decoder_matches_whole_and_records_10ms_cuda_graph(self):
        self.check_v5_stream(aa_filter=1)

    def test_aa_filter_3_decoder_matches_whole_and_records_10ms_cuda_graph(self):
        # Stronger filters, and the pre-upsampler Snakes oversampled: their filter history crosses blocks.
        self.check_v5_stream(aa_filter=3)

    def test_aper_fine_crossfade_decoder_streams_and_records_10ms_cuda_graph(self):
        # dec-s15 config: aa_filter=3 + 5 fine breath bands + causal_aper + crossfade wired into 10 ms CUDA graphs.
        import tempfile
        from pathlib import Path
        from causalvc.decoder.model import Decoder, Source
        from causalvc.fast import FastStreamingVC
        from causalvc.stream import StreamingVC
        from causalvc.voice import load

        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        spec = load(find_voice().model_path)
        cfg = {
            "channels": 128,
            "dim": 96,
            "rates": [12, 10, 2, 2],
            "antialias": True,
            "snake": True,
            "bands": 4,
            "glide": "next",
            "lookahead": 0,
            "mel_cond": False,
            "exc": "bandlimit",
            "smooth_in": "p",
            "content_hz": 100,
            "tone": 2,
            "expressive": True,
            "aa_filter": 3,
            "aper": 5,
            "breath_edges": list(Source.FINE_BREATH_EDGES),
            "causal_aper": True,
            "crossfade": True,
        }
        torch.manual_seed(9)
        dec = Decoder(1, **cfg).eval()
        with torch.no_grad():
            dec.tone.weight.normal_(0.0, 0.02)
            dec.f0_proj.weight.normal_(0.0, 0.02)
            dec.reg_cond.weight.normal_(0.0, 0.02)
            dec.source.depth.fill_(0.7)
            for mod in dec.modules():
                if hasattr(mod, "a") and hasattr(mod, "b"):
                    mod.a.uniform_(-0.3, 0.3)
                    mod.b.uniform_(-0.3, 0.3)

        with tempfile.TemporaryDirectory() as tmp:
            dec_path = str(Path(tmp) / "decoder.pt")
            torch.save({"generator": dec.state_dict(), "cfg": cfg, "speakers": [spec["speaker"]], "step": 1}, dec_path)

            x = speech_like(1.0).to(device)
            graph_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), None, graphs=True)
            graph_vc.set_voice(spec["speaker"], semitones=4.0)
            torch.manual_seed(1)
            graph_vc.prepare(480)
            self.assertIsNotNone(graph_vc.graph)
            self.assertIsNotNone(graph_vc.breath_white)

            ref_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), None, graphs=False)
            ref_vc.set_voice(spec["speaker"], semitones=4.0)
            torch.manual_seed(1)
            for _ in range(13):
                ref_vc.process(torch.zeros(32 * 480, device=device))
            for _ in range(3):
                ref_vc.process(torch.zeros(480, device=device))
            torch.manual_seed(0)
            yg = torch.cat([graph_vc.process(x[i : i + 480]) for i in range(0, len(x), 480)]).cpu()
            torch.manual_seed(0)
            ye = torch.cat([ref_vc.process(x[i : i + 480]) for i in range(0, len(x), 480)]).cpu()
            self.assertLess(float((yg - ye).abs().max().item()), 1e-4)

            # StreamingVC also runs cleanly with aper + crossfade + causal FIR breath at 10 ms blocks.
            slow_vc = StreamingVC(dec_path, spec["pitch"], str(device), None)
            slow_vc.set_voice(spec["speaker"], semitones=4.0)
            ys = torch.cat([slow_vc.process(x[i : i + 480]) for i in range(0, 12 * 480, 480)]).cpu()
            self.assertEqual(ys.shape, (12 * 480,))
            self.assertTrue(torch.isfinite(ys).all())

    def check_v5_stream(self, aa_filter):
        import tempfile
        from pathlib import Path
        from causalvc.decoder.model import Decoder
        from causalvc.fast import FastStreamingVC
        from causalvc.voice import load

        device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        spec = load(find_voice().model_path)
        cfg = {
            "channels": 128,
            "dim": 96,
            "rates": [12, 10, 2, 2],
            "antialias": True,
            "snake": True,
            "bands": 4,
            "glide": "next",
            "lookahead": 0,
            "mel_cond": False,
            "exc": "bandlimit",
            "smooth_in": "p",
            "content_hz": 100,
            "tone": 2,
            "expressive": True,
            "aa_filter": aa_filter,
        }
        torch.manual_seed(7)
        dec = Decoder(1, **cfg).eval()
        with torch.no_grad():
            dec.tone.weight.normal_(0.0, 0.02)
            dec.f0_proj.weight.normal_(0.0, 0.02)
            dec.reg_cond.weight.normal_(0.0, 0.02)
            for mod in dec.modules():
                if hasattr(mod, "a") and hasattr(mod, "b"):
                    mod.a.uniform_(-0.3, 0.3)
                    mod.b.uniform_(-0.3, 0.3)

        with tempfile.TemporaryDirectory() as tmp:
            dec_path = str(Path(tmp) / "decoder.pt")
            torch.save({"generator": dec.state_dict(), "cfg": cfg, "speakers": [spec["speaker"]], "step": 1}, dec_path)

            # 1) Whole-sequence Decoder.generate vs block-by-block FastStreamingVC._generate at 10 ms (480 samples).
            eager_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), None, graphs=False)
            eager_vc.set_voice(spec["speaker"], semitones=4.0)
            T = 16
            h = torch.randn(1, cfg["dim"], T, device=device)
            g = eager_vc.dec.speaker(eager_vc.spk)
            f0 = torch.full((1, T), 220.0, device=device)
            v = torch.full((1, T), 0.9, device=device)
            reg = eager_vc.dec.register_vec(f0, v)
            eager_vc.dec.source.noise = 0.0
            exc = eager_vc.dec.source(f0, voicing=v)
            # TF32 convolutions round differently for different lengths (~1e-4); in full float32 the
            # blocks must give exactly the whole sequence, so any lost filter history shows.
            tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
            torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.no_grad():
                    y_whole = eager_vc.dec.generate(h, g, exc, reg=reg)[0, 0]
                    eager_vc.reset()
                    y_blocks = torch.cat(
                        [
                            eager_vc._generate(h[..., t : t + 2], g, exc[..., t * 480 : (t + 2) * 480], reg=reg[:, t : t + 2])
                            for t in range(0, T, 2)
                        ]
                    )
            finally:
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
            self.assertLess(float((y_whole - y_blocks).abs().max().item()), 1e-6)

            # 2) Full 10 ms (480-sample) CUDA Graph recording and replay parity.
            x = speech_like(1.0).to(device)
            graph_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), None, graphs=True)
            graph_vc.set_voice(spec["speaker"], semitones=4.0)
            torch.manual_seed(1)
            graph_vc.prepare(480)
            self.assertIsNotNone(graph_vc.graph)
            ref_vc = FastStreamingVC(dec_path, spec["pitch"], str(device), None, graphs=False)
            ref_vc.set_voice(spec["speaker"], semitones=4.0)
            torch.manual_seed(1)
            for _ in range(13):
                ref_vc.process(torch.zeros(32 * 480, device=device))
            for _ in range(3):
                ref_vc.process(torch.zeros(480, device=device))
            torch.manual_seed(0)
            yg = torch.cat([graph_vc.process(x[i : i + 480]) for i in range(0, len(x), 480)]).cpu()
            torch.manual_seed(0)
            ye = torch.cat([ref_vc.process(x[i : i + 480]) for i in range(0, len(x), 480)]).cpu()
            self.assertLess(float((yg - ye).abs().max().item()), 1e-4)


if __name__ == "__main__":
    unittest.main()
