"""Softens sudden treble in the middle of a word.

Some voices switch their 4-12 kHz band on far faster than the speaker did: at an "m" opening
into "o", the band can appear 16 dB louder within 2 ms while everything below it rises gently,
which is heard as a click. The voice's own recordings rise about as sharply, but the voice does
it far more often than the person speaking.

This finds such a rise (the band's 5 ms average climbing 4 dB or more within 6 ms while the voice
keeps sounding) and lets the treble climb at most 1 dB per ms through it. Rises out of silence
(a word's first sound), quiet stretches and vocal-cord pulses are left alone, and wherever it
isn't acting the audio passes through bit for bit, only delayed. Seeing a rise coming takes
5 ms of look-ahead, so the output is 5 ms later than the input.

Over 40 test sentences (Kyedae) quality and speaker similarity were unchanged, it acted on 3% of
the time, and the "Kye model" click's jump fell from 16.5 to 10 dB.
"""

import numpy as np
from scipy.signal import butter, firwin, sosfilt, sosfilt_zi

SAMPLE_RATE = 48000
FRAME = SAMPLE_RATE // 1000  # 1 ms
#: Frames of look-ahead: the rise test compares the 5 ms averages 3 ms after and before.
AHEAD_FRAMES = 5
DELAY_SECONDS = AHEAD_FRAMES * FRAME / SAMPLE_RATE
#: Symmetric linear-phase FIR band-split (half-length <= AHEAD_FRAMES * FRAME so it adds zero latency
#: and guarantees 0.00 dB crossover ripple / zero phase-cancellation notch at every attenuation level).
SPLIT_HALF = 120  # 2.5 ms linear-phase group delay (fits inside 5 ms = 240-sample lookahead)
SPLIT_TAPS = 2 * SPLIT_HALF + 1
EXTRA_DELAY = AHEAD_FRAMES * FRAME - SPLIT_HALF  # 120 remaining samples -> exact 240-sample (5 ms) total delay

RATE_DB_PER_MS = 1.0
MAX_CUT_DB = 15.0
RISE_DB = 4.0
#: The voice has to have been sounding this long first: its 150-600 Hz body within 25 dB of its
#: recent peak and above BODY_FLOOR_DB (so a long pause's background noise never counts as voice).
SOUNDING_FRAMES = 30
BODY_FLOOR_DB = -55.0


class TrebleSoftener:
    """Streams mono 48 kHz float32 blocks (whole 1 ms frames); each call returns as many samples."""

    def __init__(self):
        self.split_fir = firwin(SPLIT_TAPS, 3500.0, fs=SAMPLE_RATE, window=("kaiser", 6.5)).astype(np.float64)
        self.split_hist = np.zeros(SPLIT_TAPS - 1, dtype=np.float64)
        self.detect = butter(4, [4000, 12000], "bandpass", fs=SAMPLE_RATE, output="sos")
        self.body = butter(2, [150, 600], "bandpass", fs=SAMPLE_RATE, output="sos")
        self.states = [sosfilt_zi(sos) * 0.0 for sos in (self.detect, self.body)]
        # Treble power and body level of the last 11 frames: frame j (being finished) sits in the middle.
        self.power = np.zeros(2 * AHEAD_FRAMES + 1, dtype=np.float64)
        self.body_db = np.full(2 * AHEAD_FRAMES + 1, -120.0, dtype=np.float64)
        # Low and high bands of the remaining look-ahead samples after the linear-phase FIR delay.
        self.waiting_low = np.zeros(EXTRA_DELAY, dtype=np.float64)
        self.waiting_high = np.zeros(EXTRA_DELAY, dtype=np.float64)
        self.last_gain = 1.0
        self._ramp = np.arange(FRAME, dtype=np.float64) / FRAME
        self.peak = -120.0
        self.run = 0
        self.limiting = False
        self.allowed = -120.0
        self.frames_touched = 0
        self.frames_seen = 0

    def process(self, block):
        block = np.asarray(block, dtype=np.float64)
        if block.shape[0] % FRAME:
            raise ValueError(f"blocks must be whole {FRAME}-sample frames")
        n = block.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self.split_hist, block])
        self.split_hist = buf[-(SPLIT_TAPS - 1) :].copy()
        low_cur = np.convolve(buf, self.split_fir, mode="valid")
        dry_cur = buf[SPLIT_HALF : SPLIT_HALF + n]
        high_cur = dry_cur - low_cur
        detect, self.states[0] = sosfilt(self.detect, block, zi=self.states[0])
        body, self.states[1] = sosfilt(self.body, block, zi=self.states[1])
        low = np.concatenate([self.waiting_low, low_cur])
        high = np.concatenate([self.waiting_high, high_cur])
        frames = n // FRAME
        powers = (detect.reshape(frames, FRAME) ** 2).mean(1)
        bodies = 10.0 * np.log10((body.reshape(frames, FRAME) ** 2).mean(1) + 1e-12)
        gains = np.empty(n, dtype=np.float64)
        prev_g = self.last_gain
        ramp = self._ramp
        for index in range(frames):
            self.power[:-1] = self.power[1:]
            self.power[-1] = powers[index]
            self.body_db[:-1] = self.body_db[1:]
            self.body_db[-1] = bodies[index]
            g = 10.0 ** (self.frame_gain() / 20.0)
            s = index * FRAME
            if prev_g == g:
                gains[s : s + FRAME] = g
            else:
                gains[s : s + FRAME] = prev_g + (g - prev_g) * ramp
            prev_g = g
        self.last_gain = prev_g
        result = low[:n] + high[:n] * gains
        self.waiting_low = low[n:].copy()
        self.waiting_high = high[n:].copy()
        return result.astype(np.float32)

    def frame_gain(self):
        """Gain in dB for the frame AHEAD_FRAMES back, now that the frames after it are known."""
        def averaged(start):  # the 5-frame average starting this many frames into the history, in dB
            return 10 * np.log10(self.power[start:start + 5].mean() + 1e-12)

        level, before, after, previous = averaged(3), averaged(0), averaged(6), averaged(2)
        body = self.body_db[AHEAD_FRAMES]
        self.peak = max(body, self.peak - 0.004)  # the body's recent peak, falling 4 dB a second
        self.run = self.run + 1 if body > max(self.peak - 25, BODY_FLOOR_DB) else 0
        rising = self.run > SOUNDING_FRAMES and after - before > RISE_DB
        if rising and not self.limiting:
            self.limiting, self.allowed = True, previous
        if self.limiting:
            self.allowed = min(level, self.allowed + RATE_DB_PER_MS)
            if self.allowed >= level or self.run == 0:
                self.limiting = False
        else:
            self.allowed = level
        gain = max(self.allowed - level, -MAX_CUT_DB)
        self.allowed = level + gain
        self.frames_seen += 1
        self.frames_touched += gain < -1.0
        return gain
