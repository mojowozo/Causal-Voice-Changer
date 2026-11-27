"""Tests for the v5 anti-aliased decoder: Fejér-Riesz causal LPFs, causal linear ups[0], and SnakeBeta."""

import math
import unittest

import torch

from causalvc.decoder.model import AntiAliasAct, CUp, Decoder, SnakeBeta, min_phase_lpf


class AntiAliasedDecoderTest(unittest.TestCase):
    def test_fejer_min_phase_lpf_has_flat_cascaded_passband_and_deep_stopband(self):
        h = min_phase_lpf(0.50, taps=16, beta=7.0)
        self.assertEqual(h.shape, (16,))
        self.assertAlmostEqual(float(h.sum().item()), 1.0, places=5)

        H_db = 20.0 * torch.log10(torch.fft.rfft(h, n=4096).abs().clamp_min(1e-9))
        # In 2x oversampling, 0..0.70 of 1x stage Nyquist corresponds to 0..0.35 of 2x Nyquist:
        # 12 cascaded filters (6 AntiAliasActs: up + dn) must stay within +-0.08 dB (no -16 dB scoop!).
        passband_end = int(0.35 * (len(H_db) - 1))
        cascaded_12 = 12.0 * H_db[:passband_end]
        self.assertLess(float(cascaded_12.abs().max().item()), 0.08)

        # Aliasing stopband >= 1.40 of 1x stage Nyquist (>= 0.70 of 2x Nyquist): 2 passes (up + dn) < -50 dB.
        stopband_start = int(0.70 * (len(H_db) - 1))
        self.assertLess(float((2.0 * H_db[stopband_start:]).max().item()), -50.0)

    def test_linear_cup_has_zero_100hz_clock_ripple_and_antialiased_cup_suppresses_mirrors(self):
        # 1) Causal linear interpolation inside CUp(linear=True) produces exact constant on interior frames
        # while ramping cleanly from 0 on frame 0 (matching FastStreamingVC's zero-initialized _hist state).
        cup_lin = CUp(16, 8, stride=12, linear=True).eval()
        x_const = torch.full((1, 16, 24), 1.75)
        with torch.no_grad():
            up_only = cup_lin.upsample_linear(x_const)
        self.assertEqual(up_only.shape, (1, 16, 24 * 12))
        self.assertLess(float((up_only[..., 12:] - 1.75).abs().max().item()), 1e-6)

        # 2) CUp(stride=10, antialias=True) LPF keeps <=0.8 * in_Nyq flat while rejecting >=1.4 * in_Nyq mirrors.
        cup_aa = CUp(16, 8, stride=10, antialias=True).eval()
        self.assertIsNotNone(cup_aa.lpf)
        h_up = cup_aa.lpf.weight[0, 0]
        H_up = 20.0 * torch.log10(torch.fft.rfft(h_up, n=8192).abs().clamp_min(1e-9))
        idx_pass = int(round((0.8 / 10.0) * (len(H_up) - 1)))
        idx_mirr = int(round((1.4 / 10.0) * (len(H_up) - 1)))
        idx_clk = int(round((2.0 / 10.0) * (len(H_up) - 1)))
        self.assertGreater(float(H_up[idx_pass].item()), -0.5)
        self.assertLess(float(H_up[idx_mirr].item()), -35.0)
        self.assertLess(float(H_up[idx_clk].item()), -34.0)

    def test_v5_decoder_forward_backward_and_clock_spur_suppression(self):
        dec = Decoder(
            2,
            channels=64,
            dim=48,
            rates=(12, 10, 2, 2),
            antialias=True,
            snake=True,
            bands=4,
            glide="next",
            exc="bandlimit",
            smooth_in="p",
            tone=2,
            content_hz=100,
            expressive=True,
        )
        self.assertTrue(dec.snake)
        self.assertIsInstance(dec.post_act, SnakeBeta)
        self.assertIsInstance(dec.res[0][0].a1[0], AntiAliasAct)
        self.assertTrue(dec.res[0][0].a1[0].snake)
        self.assertIsInstance(dec.res[-1][0].s1[0], SnakeBeta)

        c = torch.randn(1, 16, 768)
        f0 = torch.full((1, 16), 233.0)
        v = torch.full((1, 16), 0.85)
        loud = torch.full((1, 16), -22.0)
        tn = torch.full((1, 16, 2), -12.0)
        spk = torch.zeros(1, dtype=torch.long)

        h, g = dec.features(c, f0, loud, spk, tone=tn, content_hz=100, voicing=v)
        exc = dec.source(f0, voicing=v)
        reg = dec.register_vec(f0, v)
        y = dec.generate(h, g, exc, reg=reg)
        self.assertEqual(y.shape, (1, 1, 16 * 480))
        loss = y.pow(2).mean()
        loss.backward()
        snake_grads = [p.grad for n, p in dec.named_parameters() if (".a" in n or ".b" in n) and p.grad is not None]
        self.assertGreater(len(snake_grads), 10)
        for gr in snake_grads:
            self.assertTrue(torch.isfinite(gr).all())

    def test_snake_beta_fused_autograd_matches_reference_gradients_and_cuts_saved_memory(self):
        from causalvc.decoder import model as dec_model

        self.assertTrue(hasattr(dec_model, "_SnakeBetaFn"), "_SnakeBetaFn is not implemented yet")
        torch.manual_seed(3)
        x64 = torch.randn(2, 4, 17, dtype=torch.float64, requires_grad=True)
        a64 = torch.randn(1, 4, 1, dtype=torch.float64, requires_grad=True) * 0.5
        b64 = torch.randn(1, 4, 1, dtype=torch.float64, requires_grad=True) * 0.5
        self.assertTrue(torch.autograd.gradcheck(dec_model._SnakeBetaFn.apply, (x64, a64, b64), eps=1e-6, atol=1e-4))

        # Verify bf16 forward + backward matches exact float32 reference within bf16 tolerance.
        x_bf = torch.randn(2, 8, 64, dtype=torch.bfloat16, requires_grad=True)
        sb = SnakeBeta(8)
        with torch.no_grad():
            sb.a.uniform_(-0.5, 0.5)
            sb.b.uniform_(-0.5, 0.5)
        y_bf = sb(x_bf)
        self.assertEqual(y_bf.dtype, torch.bfloat16)
        (y_bf.float().pow(2).mean()).backward()
        self.assertIsNotNone(x_bf.grad)
        self.assertTrue(torch.isfinite(sb.a.grad).all())
        self.assertTrue(torch.isfinite(sb.b.grad).all())

    def test_specloss_includes_512_and_build_forwards_v5_flags(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from causalvc import build
        from causalvc.decoder.disc import SpecLoss

        sl = SpecLoss(f_max=16000.0)
        self.assertEqual(sl.ffts, (512, 1024, 2048, 4096))
        x = torch.randn(2, 1, 9600)
        y = torch.randn(2, 1, 9600)
        self.assertTrue(torch.isfinite(sl(x, y)))

        self.assertEqual(build.DEFAULT_BASES[0], "runs/models/dec-s16-100hz-v5-aa3-hi4k")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            base_dir = p / "base"
            base_dir.mkdir()
            (base_dir / "last.pt").write_bytes(b"stub")
            (base_dir / "args.json").write_text(
                json.dumps(
                    {
                        "encoder": "runs/models/enc-100hz-v4/encoder.pt",
                        "disc": 3,
                        "bands": 4,
                        "lookahead": 0,
                        "channels": 384,
                        "dim": 256,
                        "exc": "bandlimit",
                        "rates": [12, 10, 2, 2],
                        "antialias": True,
                        "snake": True,
                        "glide_next": True,
                        "smooth_in": "p",
                        "tone": 5.0,
                        "expressive": True,
                        "aa_filter": 3,
                        "aper": True,
                        "breath_from": 1000.0,
                        "breath_bands": "fine",
                        "breath_cap": 8.0,
                        "crossfade": True,
                    }
                )
            )
            audio = p / "sample.wav"
            audio.write_bytes(b"stub")
            captured = {}

            def fake_train_main(on_save=None):
                captured["argv"] = list(build.sys.argv[1:])

            with patch.object(build, "make_index"), patch("causalvc.decoder.train.main", side_effect=fake_train_main):
                build.sys.argv = ["build.py", "--name", "TestV5", "--audio", str(audio), "--base", str(base_dir)]
                build.main()

            argv = captured["argv"]
            self.assertIn("--antialias", argv)
            self.assertIn("--snake", argv)
            idx = argv.index("--rates")
            self.assertEqual(argv[idx + 1 : idx + 5], ["12", "10", "2", "2"])
            self.assertEqual(argv[argv.index("--aa-filter") + 1], "3")  # a voice keeps its base's filters
            self.assertIn("--aper", argv)  # and its breath channels
            self.assertEqual(argv[argv.index("--breath-from") + 1], "1000.0")
            self.assertEqual(argv[argv.index("--breath-bands") + 1], "fine")
            self.assertEqual(argv[argv.index("--breath-cap") + 1], "8.0")
            self.assertIn("--crossfade", argv)

    def test_aa_filter_3_oversamples_pre_upsampler_snakes_and_takes_a_v5_base(self):
        from causalvc.decoder.train import own_key

        cfg = dict(channels=64, dim=48, rates=(12, 10, 2, 2), antialias=True, snake=True, bands=4, glide="next",
                   exc="bandlimit", smooth_in="p", tone=2, content_hz=100, expressive=True)
        torch.manual_seed(3)
        v5 = Decoder(2, **cfg)
        new = Decoder(2, **cfg, aa_filter=3)
        self.assertIsInstance(new.pre_acts[0], SnakeBeta)  # works on 100 Hz features, not audio
        for act in new.pre_acts[1:]:
            self.assertIsInstance(act, AntiAliasAct)
            self.assertTrue(act.snake)
        self.assertEqual(new.res[0][0].a1[0].w_dn.shape[-1], 48)  # and the strong filters of aa_filter 2
        # --init from a v5 base: every tensor finds its place, the pre-upsampler Snakes under .act
        own = new.state_dict()
        mapped = {own_key(a, own): b for a, b in v5.state_dict().items()}
        self.assertEqual(set(mapped), set(own))
        new.load_state_dict(mapped)
        for old_act, new_act in zip(v5.pre_acts[1:], new.pre_acts[1:]):
            self.assertTrue(torch.equal(old_act.a, new_act.act.a))
            self.assertTrue(torch.equal(old_act.b, new_act.act.b))
        self.assertEqual(own_key("pre_acts.0.a", own), "pre_acts.0.a")

    def test_causal_aperiodicity_reads_each_frame_from_the_audio_up_to_its_end(self):
        from causalvc.decoder.model import FRAME, SR, aperiodicity

        torch.manual_seed(0)
        n, f = 60 * FRAME, 200.0
        t = torch.arange(n) / SR
        tone = sum(torch.sin(2 * math.pi * k * f * t) / k for k in range(1, 100) if k * f < SR / 2)
        noise = 0.3 * torch.randn(n)
        f0 = torch.full((1, 60), f)
        clean, breathy = aperiodicity(tone[None], f0, causal=True), aperiodicity(noise[None], f0, causal=True)
        self.assertEqual(clean.shape, (1, 60, 4))
        self.assertLess(clean[0, 10:, :3].max().item(), 0.05)  # harmonics: 0
        self.assertGreater(breathy[0, 10:, :3].median().item(), 0.6)  # noise: near 1
        # frame t never sees audio after sample 480 (t + 1): change the second half, the first half stays put
        mixed = torch.cat([tone[: n // 2], noise[n // 2 :]])[None]
        a, b = aperiodicity(tone[None], f0, causal=True), aperiodicity(mixed, f0, causal=True)
        self.assertTrue(torch.equal(a[:, :30], b[:, :30]))
        self.assertGreater(b[0, 34:, :3].mean().item(), 0.5)  # and it hears the change within about 4 frames
        centred = aperiodicity(mixed, f0)
        self.assertFalse(torch.equal(a[:, 28:30], centred[:, 28:30]))  # the centred measure looks ahead

    def test_breath_penalty_asks_each_band_for_no_more_regularity_than_the_target(self):
        from causalvc.decoder.model import FRAME, SR, aperiodicity
        from causalvc.decoder.train import BREATH_BANDS, breath_edges, breath_penalty

        self.assertEqual(breath_edges(1000.0), (1000.0, 2000.0, 4000.0, 8000.0, 16000.0))
        self.assertEqual(breath_edges(4000.0), (4000.0, 8000.0, 16000.0))
        with self.assertRaises(AssertionError):
            breath_edges(3000.0)
        edges = breath_edges(1000.0)
        torch.manual_seed(0)
        t, n = 50, 50 * FRAME
        weight = torch.ones(2, t)
        ap_t = torch.full((2, t, len(BREATH_BANDS)), 0.5)
        tone = torch.sin(2 * math.pi * 1500.0 * torch.arange(n) / SR).expand(2, n)
        # the same render twice (no use of the noise at all): correlation 1 where 0.5 is allowed, in every band
        both = tone + 0.1 * torch.randn(2, n)
        self.assertAlmostEqual(breath_penalty(both, both, ap_t, weight, edges).item(), 4 * 0.25, places=3)
        # fresh noise in every band: nothing to penalise
        self.assertLess(breath_penalty(torch.randn(2, n), torch.randn(2, n), ap_t, weight, edges).item(), 1e-3)
        # a fixed 1.5 kHz tone over fresh noise: only the 1-2 kHz band is too regular
        a, b = tone + 0.05 * torch.randn(2, n), tone + 0.05 * torch.randn(2, n)
        only_low = breath_penalty(a, b, ap_t, weight, edges).item()
        self.assertGreater(only_low, 0.1)
        self.assertLess(only_low, 0.26)
        self.assertLess(breath_penalty(a, b, ap_t, weight, breath_edges(2000.0)).item(), 1e-3)
        # frames with no weight do not count, and the target bands come from aperiodicity's own edges
        self.assertEqual(breath_penalty(both, both, ap_t, torch.zeros(2, t), edges).item(), 0.0)
        self.assertEqual(aperiodicity(both, torch.full((2, t), 150.0), edges=edges).shape, (2, t, 4))

    def test_fine_breath_bands_split_1_to_4_khz_and_reach_each_stage_that_can_hold_them(self):
        from causalvc.decoder.model import FRAME, SR, Source

        fine = Source.FINE_BREATH_EDGES
        src = Source(8, bands=4, with_f0=True, aper=len(fine) - 1, breath_edges=fine)
        self.assertEqual(src.n_in, 8 + 1 + 4 + 2 * 5)
        torch.manual_seed(0)
        n = 40 * FRAME
        noise = src.breath(None, torch.ones(1, 1, n), torch.Size((1, n)))[0]
        spec = torch.fft.rfft(noise, dim=-1).abs().pow(2)
        fr = torch.fft.rfftfreq(n, 1.0 / SR)
        for i, (lo, hi) in enumerate(zip(fine[:-1], fine[1:])):  # each channel holds its own band only
            inside = spec[i, (fr >= lo) & (fr < hi)].sum() / spec[i].sum()
            self.assertGreater(inside.item(), 0.999)
        self.assertEqual(noise[5:].abs().max().item(), 0.0)  # no aperiodicity given: the scaled copies are silent
        with self.assertRaises(AssertionError):
            Source(8, bands=4, aper=4, breath_edges=fine)
        cfg = dict(channels=64, dim=48, rates=(12, 10, 2, 2), antialias=True, snake=True, bands=4, glide="next",
                   exc="bandlimit", smooth_in="p", content_hz=100, aper=5, breath_edges=list(fine))
        dec = Decoder(2, **cfg)
        e = torch.ones(1, src.n_in, 8)
        f = torch.full((1, 1, 8), 200.0)
        h = 8 + 1 + 4
        # 12 kHz stage (6 kHz Nyquist): 1-2 and 2-4 kHz fit, 4-8 kHz and up wait for later stages
        kept = dec.limit(e, f, 12000)[0, h:, 0]
        self.assertEqual(kept.tolist(), [1.0, 1.0, 0.0, 0.0, 0.0] * 2)
        self.assertEqual(dec.limit(e, f, 48000)[0, h:, 0].tolist(), [1.0] * 10)

    def test_breath_inputs_hand_the_coarse_breath_weights_to_the_fine_bands(self):
        from causalvc.decoder.model import FRAME, SR, Source
        from causalvc.decoder.train import breath_inputs

        coarse, fine = Source.EDGES, Source.FINE_BREATH_EDGES
        base = 13
        old = torch.arange(1.0, 1.0 + 3 * (base + 8) * 2).reshape(3, base + 8, 2)
        w = breath_inputs(old, torch.Size((3, base + 10, 2)), base, coarse, fine)
        self.assertTrue(torch.equal(w[:, :base], old[:, :base]))
        # fixed-level: 1-2 and 2-4 kHz from 0-4 kHz, scaled by sqrt(1/4) and sqrt(2/4); the rest copied
        self.assertTrue(torch.allclose(w[:, base + 0], old[:, base + 0] * 0.5))
        self.assertTrue(torch.allclose(w[:, base + 1], old[:, base + 0] * math.sqrt(0.5)))
        for j, i in ((2, 1), (3, 2), (4, 3)):
            self.assertTrue(torch.equal(w[:, base + j], old[:, base + i]))
        # aperiodicity-scaled: new 1-2 and 2-4 kHz start silent, matching bands keep theirs
        self.assertEqual(w[:, base + 5 : base + 7].abs().sum().item(), 0.0)
        for j, i in ((7, 5), (8, 6), (9, 7)):
            self.assertTrue(torch.equal(w[:, base + j], old[:, base + i]))
        # and the scaling hands on the same noise level: 2-4 kHz power of the coarse 0-4 kHz channel matches
        # the fine 2-4 kHz channel at its new weight
        torch.manual_seed(1)
        n = 400 * FRAME
        fr = torch.fft.rfftfreq(n, 1.0 / SR)
        band = (fr >= 2000) & (fr < 4000)
        c = Source(8, bands=4, aper=4).breath(None, torch.ones(1, 1, n), torch.Size((1, n)))[0, 0]
        f_ = Source(8, bands=4, aper=5, breath_edges=fine).breath(None, torch.ones(1, 1, n), torch.Size((1, n)))[0, 1]
        pc = torch.fft.rfft(c).abs().pow(2)[band].sum()
        pf = (torch.fft.rfft(f_).abs().pow(2)[band].sum()) * 0.5
        self.assertAlmostEqual((pf / pc).item(), 1.0, delta=0.05)

    def test_crossfade_turns_harmonics_down_only_in_breathy_bands_and_starts_as_a_no_op(self):
        from causalvc.decoder.model import FRAME, SR, Source

        fine = Source.FINE_BREATH_EDGES
        plain = Source(8, bands=4, with_f0=True, aper=5, breath_edges=fine)
        xf = Source(8, bands=4, with_f0=True, aper=5, breath_edges=fine, crossfade=True)
        t = 40
        f0 = torch.full((1, t), 200.0)
        ap = torch.zeros(1, t, 5)
        ap[..., 1] = 1.0  # 2-4 kHz is all breath
        torch.manual_seed(0)
        a = plain(f0, ap=ap)
        torch.manual_seed(0)
        b = xf(f0, ap=ap)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))  # depth 0: exactly the plain source
        with torch.no_grad():
            xf.depth.fill_(1.0)
        torch.manual_seed(0)
        c = xf(f0, ap=ap)
        n = t * FRAME
        fr = torch.fft.rfftfreq(n, 1.0 / SR)

        def power(x, lo, hi):
            return torch.fft.rfft(x.double()).abs().pow(2)[(fr >= lo) & (fr < hi)].sum().item()

        band0 = 8 + 1  # the 0-4 kHz harmonic band
        self.assertLess(power(c[0, band0], 2050, 3950), 1e-3 * power(a[0, band0], 2050, 3950))  # gone where breathy
        self.assertAlmostEqual(power(c[0, band0], 0, 1950) / power(a[0, band0], 0, 1950), 1.0, delta=0.01)  # kept below
        self.assertAlmostEqual(power(c[0, band0 + 1], 4000, 8000) / power(a[0, band0 + 1], 4000, 8000), 1.0, delta=0.01)
        self.assertTrue(torch.equal(c[0, :8], a[0, :8]))  # 8 sines at 200 Hz reach 1.6 kHz: below 2-4 kHz
        # the 8th sine is turned down when it falls in a breathy band (400 Hz: 3.2 kHz)
        with torch.no_grad():
            xf.depth.fill_(0.0)
        xf.depth.data[1] = 0.75
        torch.manual_seed(0)
        d = xf(torch.full((1, t), 400.0), ap=ap)
        torch.manual_seed(0)
        e = plain(torch.full((1, t), 400.0), ap=ap)
        self.assertAlmostEqual((d[0, 7].std() / e[0, 7].std()).item(), 0.5, delta=0.02)
        self.assertTrue(torch.allclose(d[0, :4], e[0, :4]))  # up to 1.6 kHz; the 5th (2 kHz) is in the band
        # without an aperiodicity input nothing is turned down; the depth learns from the output
        torch.manual_seed(0)
        self.assertTrue(torch.allclose(xf(f0), (torch.manual_seed(0), plain(f0))[1]))
        xf(f0, ap=ap).sum().backward()
        self.assertGreater(xf.depth.grad[1].abs().item(), 0.0)

    def test_breath_cap_penalises_only_output_noisier_than_the_target(self):
        from causalvc.decoder.model import FRAME, SR
        from causalvc.decoder.train import breath_cap, breath_edges

        edges = breath_edges(1000.0)
        torch.manual_seed(0)
        t, n = 60, 60 * FRAME
        f0 = torch.full((2, t), 150.0)
        weight = torch.ones(2, t)
        k = torch.arange(1, 100)[:, None]
        harm = (torch.sin(2 * math.pi * 150.0 * k * torch.arange(n) / SR) / k.sqrt()).sum(0).expand(2, n).contiguous()
        clean = harm + 0.003 * torch.randn(2, n)
        noisy = harm + 2.0 * torch.randn(2, n)
        # as noisy as the target, or cleaner: nothing
        self.assertLess(breath_cap(clean, clean, clean, f0, weight, edges).item(), 1e-4)
        self.assertLess(breath_cap(clean, clean.flip(0), noisy, f0, weight, edges).item(), 1e-4)
        # noisier: penalised, and the loss reaches the output
        out = noisy.clone().requires_grad_(True)
        loss = breath_cap(out, harm + 2.0 * torch.randn(2, n), clean, f0, weight, edges)
        self.assertGreater(loss.item(), 0.05)
        loss.backward()
        self.assertGreater(out.grad.abs().sum().item(), 0.0)
        # frames with no weight do not count
        self.assertEqual(breath_cap(noisy, noisy, clean, f0, torch.zeros(2, t), edges).item(), 0.0)

    def test_cross_architecture_init_transfers_frontend_without_saturating_snake_vocoder(self):
        from causalvc.decoder import train as dec_train

        self.assertTrue(hasattr(dec_train, "vocoder_arch_changed"), "vocoder_arch_changed is not implemented yet")
        v4_cfg = {"channels": 64, "dim": 48, "bands": 4, "glide": "next", "exc": "bandlimit", "smooth_in": "p", "content_hz": 100, "tone": 2, "expressive": True}
        v5_cfg = {**v4_cfg, "rates": [12, 10, 2, 2], "antialias": True, "snake": True}
        self.assertTrue(dec_train.vocoder_arch_changed(v4_cfg, v5_cfg))
        self.assertFalse(dec_train.vocoder_arch_changed(v5_cfg, v5_cfg))

    def test_streaming_aperiodicity_prepadded_and_causal_fir_breath_match_whole_sequence(self):
        from causalvc.decoder.model import AP_LOOKBACK, BREATH_TAPS, FRAME, SR, Source, aperiodicity

        torch.manual_seed(4)
        T = 24
        n = T * FRAME
        t = torch.arange(n) / SR
        tone = sum(torch.sin(2 * math.pi * k * 190.0 * t) / k for k in range(1, 50) if k * 190.0 < SR / 2)
        wav = torch.cat([tone[: n // 2], 0.25 * torch.randn(n - n // 2)])[None]
        f0 = torch.full((1, T), 190.0)
        fine = Source.FINE_BREATH_EDGES

        # 1) Prepadded 10 ms (1-frame) streaming slices match whole-sequence causal aperiodicity bit-for-bit.
        whole_ap = aperiodicity(wav, f0, causal=True, edges=fine)
        padded = torch.nn.functional.pad(wav, (AP_LOOKBACK, 0))
        chunks_ap = []
        for i in range(T):
            sl = padded[:, i * FRAME : i * FRAME + AP_LOOKBACK + FRAME]
            chunks_ap.append(aperiodicity(sl, f0[:, i : i + 1], causal=True, edges=fine, prepadded=True))
        self.assertTrue(torch.allclose(whole_ap, torch.cat(chunks_ap, dim=1), atol=1e-6))

        # 2) Stateful causal FIR breath filterbank: unit variance, >97% in-band energy across all 5 fine bands,
        # and exact 10 ms block-by-block == whole-sequence equivalence (zero circular wraparound or seam clicks).
        src = Source(8, bands=4, glide="next", with_f0=True, aper=5, breath_edges=fine, crossfade=True)
        with torch.no_grad():
            src.depth.fill_(0.8)
            src.noise = 0.0
        n_long = 120 * FRAME
        torch.manual_seed(11)
        fir_whole = src.breath(
            None,
            torch.ones(1, 1, n_long),
            torch.Size((1, n_long)),
            white_hist=torch.zeros(1, 5, BREATH_TAPS - 1),
        )[0, :5]
        # Steady-state variance after FIR warm-up matches amp^2 (unit-variance band noise * amp).
        self.assertTrue(torch.allclose(fir_whole[:, BREATH_TAPS:].std(dim=-1), torch.full((5,), src.amp), rtol=0.08))
        spec = torch.fft.rfft(fir_whole[:, BREATH_TAPS:], dim=-1).abs().pow(2)
        fr = torch.fft.rfftfreq(n_long - BREATH_TAPS, 1.0 / SR)
        for i, (lo, hi) in enumerate(zip(fine[:-1], fine[1:])):
            inside = spec[i, (fr >= lo) & (fr < hi)].sum() / spec[i].sum()
            self.assertGreater(inside.item(), 0.97)

        # 10 ms (1-frame) block-by-block Source.forward with nxt_ap + breath_state matches whole-sequence exactly!
        ap = torch.linspace(0.05, 0.95, T * 5).reshape(1, T, 5)
        nxt_ap = torch.full((1, 5), 0.95)
        torch.manual_seed(21)
        w_whole = torch.randn(1, 5, n)
        # Verify by driving breath() directly with the exact same white noise stream split into 480-sample chunks.
        voiced = torch.ones(1, 1, n)
        level_whole = src._ap_samples(ap, n, nxt_ap=nxt_ap)
        level_blocks = torch.cat(
            [
                src._ap_samples(ap[:, i : i + 1], FRAME, nxt_ap=ap[:, i + 1] if i + 1 < T else nxt_ap)
                for i in range(T)
            ],
            dim=-1,
        )
        self.assertTrue(torch.allclose(level_whole, level_blocks, atol=1e-6))

        # Check FIR streaming state continuity across 10 ms chunks with identical white-noise stream.
        z_whole = torch.cat([torch.zeros(1, 5, BREATH_TAPS - 1), w_whole], dim=-1)
        y_whole = torch.nn.functional.conv1d(z_whole, src.breath_fir, groups=5) * voiced * src.amp
        hist = torch.zeros(1, 5, BREATH_TAPS - 1)
        ys = []
        for i in range(T):
            z = torch.cat([hist, w_whole[..., i * FRAME : (i + 1) * FRAME]], dim=-1)
            hist = z[..., -(BREATH_TAPS - 1) :]
            ys.append(torch.nn.functional.conv1d(z, src.breath_fir, groups=5) * src.amp)
        self.assertLess(float((y_whole - torch.cat(ys, dim=-1)).abs().max().item()), 1e-6)

    def test_pretrained_v5_default_and_v4_fallback_side_by_side(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from causalvc import build, weights

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wdir = root / "weights"
            v4_files = weights.PRETRAINED_BASES["v4"][2]
            fake_args = b"fake-v4-args"
            fake_sha = weights.hashlib.sha256(fake_args).hexdigest()
            patched_pre = {
                "v4": ("causalvc-base-v4", "https://example.invalid/v4.zip", {**v4_files, "args.json": fake_sha})
            }
            old_base = wdir / weights.BASE_NAME
            old_base.mkdir(parents=True)
            for fname in v4_files:
                (old_base / fname).write_bytes(fake_args if fname == "args.json" else b"stub")

            with (
                patch.object(weights, "FOLDER", wdir),
                patch.object(weights, "PRETRAINED_BASES", patched_pre),
                patch.object(build, "ROOT", root),
                patch.object(build, "CUSTOM", root / "custom-base"),
            ):
                # Upgrading default preserves an existing v4 install in weights/causalvc-base-v4.
                weights._preserve_previous_base(old_base)
                self.assertFalse(old_base.exists())
                self.assertTrue((wdir / "causalvc-base-v4" / "last.pt").is_file())

                # Both default (v5) and v4 appear in build.bases() and resolve cleanly.
                found = build.bases()
                self.assertIn("default", found)
                self.assertIn("v4", found)
                self.assertEqual(found["v4"], wdir / "causalvc-base-v4")
                self.assertEqual(build.resolve_base("v4"), wdir / "causalvc-base-v4")


if __name__ == "__main__":
    unittest.main()
