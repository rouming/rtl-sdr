# RTL-SDR FM RDS Receiver

Real-time FM Radio Data System (RDS) decoder using a low-cost RTL-SDR dongle.
Parses station metadata (programme service name, radio text, programme type,
and programme identifier) directly from any FM broadcast.

Inspired by and based on the [gr-rds](https://github.com/bastibl/gr-rds)
GNURadio block implementation by Bastian Bloessl.

---

## Hardware

An RTL2832U-based USB dongle (RTL-SDR). Cheap and widely available. Typical
frequency coverage is 25–1725 MHz depending on the tuner chip, which includes
the entire FM broadcast band.

---

## Files

| File | Description |
|---|---|
| `fm-rds-gnuradio.py` | GNURadio flowgraph implementation. Instantiates native GNURadio and gr-rds blocks. Requires GNURadio and gr-rds installed on the host. |
| `fm-rds.py` | Pure Python reimplementation of the same signal chain. No GNURadio dependency. Uses NumPy/SciPy for DSP and Numba `@njit` for performance-critical loops. |
| `rds_decoder.py` | RDS group decoder: syndrome-based error detection and correction over the received bitstream. |
| `rds_parser.py` | RDS group parser: extracts PS, RadioText, PTY, PI from decoded groups. |

---

## Signal Processing Pipeline

Both implementations follow the same processing chain, derived from the
`rds_rx.grc` GNURadio flowgraph in gr-rds.

```
RTL-SDR IQ samples
  1,920,000 Hz  (complex64)
        |
        v
+-------------------+
|  XLating FIR #1   |  Shift +250 kHz (DC offset correction)
|                   |  LPF cutoff: 135 kHz  transition: 20 kHz
|                   |  Decimation: 6x
+-------------------+
        |
  320,000 Hz  (complex64)
        |
        v
+-------------------+
|   FM Demodulator  |  Quadrature demod: angle(x[n] * conj(x[n-1]))
|                   |  Gain = fs / (2*pi * 75000)
+-------------------+
        |
  320,000 Hz  (float32)   <-- MPX baseband
        |
        v
+-------------------+
|  XLating FIR #2   |  Shift -57 kHz (center RDS subcarrier)
|                   |  LPF cutoff: 7.5 kHz  transition: 5 kHz
|                   |  Decimation: 10x
+-------------------+
        |
   32,000 Hz  (complex64)
        |
        v
+-------------------+
| Rational Resampler|  32,000 Hz -> 19,000 Hz  (factor 19/32)
|                   |  Anti-alias FIR, zero-insertion upsampling
+-------------------+
        |
   19,000 Hz  (complex64)   -- 16 samples/symbol at 1187.5 baud
        |
        v
+-------------------+
| RRC Matched Filter|  Root-Raised-Cosine taps (alpha=1.0, 151 taps)
|                   |  Manchester difference: taps[n] - taps[n+8]
+-------------------+
        |
        v
+-------------------+
|       AGC         |  rate=2e-3  reference=0.585
|                   |  initial gain=53  max gain=1000
+-------------------+
        |
        v
+-------------------+
|   Symbol Sync     |  Timing Error Detector: Zero Crossing (3-tap)
|                   |  2nd-order loop filter  BW=0.01  damping=1.0
|                   |  MMSE 8-tap interpolator, 128-phase polyphase bank
|                   |  SPS=16  max deviation=0.1
+-------------------+
        |
  ~1187.5 symbols/sec  (complex64)
        |
        v
+-------------------+
| Constellation     |  BPSK Costas loop: decision-directed phase recovery
| Receiver          |  Loop BW = 2*pi/100   freq clamp: +/-0.002 rad/sample
+-------------------+
        |
        v
+-------------------+
|  Diff Decoder     |  NRZ-M:  out[n] = (in[n] - in[n-1]) mod 2
+-------------------+
        |
        v
+-------------------+
|   RDS Decoder     |  Syndrome error detection/correction
|                   |  Reconstructs 26-bit code words, extracts groups
+-------------------+
        |
        v
+-------------------+
|   RDS Parser      |  Group 0: PS name
|                   |  Group 2: RadioText
|                   |  Group 0/1: PTY, PI
+-------------------+
        |
        v
  Decoded RDS metadata  (PS, RadioText, PTY, PI)
```

### Key frequencies

| Point in chain | Sample rate | Signal |
|---|---|---|
| RTL-SDR output | 1,920,000 Hz | IQ (complex), FM station + guard |
| After XLating #1 | 320,000 Hz | FM station baseband IQ |
| After FM demod | 320,000 Hz | MPX composite audio (real) |
| After XLating #2 | 32,000 Hz | RDS subcarrier IQ |
| After resampler | 19,000 Hz | RDS IQ, 16 samples/symbol |
| After symbol sync | ~1187.5 sym/s | BPSK symbols |

---

## Installation

### Pure Python implementation (`fm-rds.py`)

```bash
virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

```
pyrtlsdr
numpy
scipy
numba
```

### GNURadio implementation (`fm-rds-gnuradio.py`)

GNURadio and gr-rds must be installed on the host system. The virtualenv must
be created with `--system-site-packages` so the GNURadio Python bindings
(installed system-wide) are visible inside the environment.

```bash
virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install GNURadio and gr-rds via your distribution's package manager, for example:

```bash
sudo apt install gnuradio gr-rds   # Debian/Ubuntu
```

---

## Usage

Pass the FM station frequency in MHz via `--fm-freq`:

```bash
# Pure Python
python fm-rds.py --fm-freq 103.4

# GNURadio
python fm-rds-gnuradio.py --fm-freq 103.4
```

The receiver tunes 250 kHz below the target frequency to avoid the RTL-SDR
DC spike, then shifts back in the first translating filter stage.

Output is printed to the terminal as RDS groups are decoded:

```
⠖ CONSTELLATION: [____##________##____] | RadioText: ENERGY - HIT MUSIC ONLY !
```

---

## References

- [gr-rds by Bastian Bloessl](https://github.com/bastibl/gr-rds) — original GNURadio RDS implementation this project is based on
- IEC 62106 — Specification of the Radio Data System (RDS) for VHF/FM sound broadcasting
