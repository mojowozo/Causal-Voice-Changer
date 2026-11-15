"""Do content features carry the moment-to-moment brightness and vowel openness of a sound?

A linear read-out is fitted from each encoder's features to per-frame brightness and openness
on a mix of speech and laughter from some EARS speakers, and scored (correlation) on other
speakers, separately for speech and laughter. Low on laughter means the decoder is not told
how the laugh sounds and has to guess.
"""
import os, sys, glob
import torch, torchaudio
sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, load48
from causalvc.encoder.train import load_teacher
from causalvc.infer import load_encoder
from causalvc.voice import shared

def targets(x):  # 50 Hz frames
    S = torch.stft(x, 1024, 960, window=torch.hann_window(1024), return_complex=True).abs().pow(2); f = torch.arange(513) * 48000 / 1024
    e = lambda lo, hi: 10 * torch.log10(S[(f >= lo) & (f < hi)].sum(0) + 1e-10)
    return torch.stack([e(2000, 8000) - e(250, 2000), e(700, 2000) - e(250, 700)], 1), e(80, 16000)

teacher = load_teacher(DEV); T = lambda w: teacher(w).last_hidden_state
encs = {"teacher (ContentVec)": T, "current encoder": load_encoder(shared("encoder.pt"), DEV).encode}
if len(sys.argv) > 1: encs["new encoder"] = load_encoder(sys.argv[1], DEV).encode
spk = sorted(glob.glob("data/raw/ears/p*")); train, test = spk[5:45:2], spk[60:100:4]
KIND = {"speech": ("rainbow_01_regular", "freeform_speech_01"), "laughter": ("nonverbal_laughter_open", "nonverbal_laughter_closed")}

def collect(dirs, stems, enc):
    X, Y = [], []
    for d in dirs:
        for s in stems:
            p = f"{d}/{s}.wav"
            if not os.path.exists(p): continue
            x = load48(p)[: 15 * 48000]; y, lev = targets(x)
            with torch.no_grad(): c = enc(torchaudio.functional.resample(x, 48000, 16000)[None].to(DEV))[0].float().cpu()
            n = min(len(c), len(y)); m = lev[:n] > lev.max() - 35; X.append(c[:n][m]); Y.append(y[:n][m])
    return torch.cat(X), torch.cat(Y)

print(f"{'':24}{'speech: brightness':>20}{'openness':>10}{'laughter: brightness':>24}{'openness':>10}")
for name, enc in encs.items():
    Xs, Ys = zip(*(collect(train, st, enc) for st in KIND.values())); X, Y = torch.cat(Xs), torch.cat(Ys)
    mu, sd = X.mean(0), X.std(0) + 1e-5; Z = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1)
    W = torch.linalg.solve(Z.T @ Z + 50 * torch.eye(Z.shape[1]), Z.T @ Y); row = ""
    for st in KIND.values():
        Xt, Yt = collect(test, st, enc); P = torch.cat([(Xt - mu) / sd, torch.ones(len(Xt), 1)], 1) @ W
        row += "".join(f"{float(torch.corrcoef(torch.stack([P[:, i], Yt[:, i]]))[0, 1]):{w}.2f}" for i, w in ((0, 20 if st[0].startswith('r') else 24), (1, 10)))
    print(f"{name:24}{row}", flush=True)
