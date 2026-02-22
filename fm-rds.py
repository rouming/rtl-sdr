#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys
import time
import numpy as np
import osmosdr
import asyncio
import queue
import concurrent
from scipy import signal
from rtlsdr.rtlsdraio import RtlSdrAio
from numba import njit

from rds_decoder import RDSDecoder
from rds_parser import RDSParser

def generate_rrc_taps(gain, fs, symbol_rate, alpha, ntaps):
    """Generate Root Raised Cosine (RRC) FIR filter taps.

    This function computes a discrete-time Root Raised Cosine
    pulse-shaping filter equivalent to GNU Radio's
    firdes.root_raised_cosine. The RRC filter is commonly used to
    minimize inter-symbol interference (ISI). When cascaded with
    another RRC filter at the receiver, the combined response forms a
    raised cosine filter that satisfies the Nyquist zero-ISI
    criterion.

    The implementation follows the closed-form RRC impulse response sampled at
    the given sampling rate. Special cases are handled explicitly to avoid
    numerical instability at the known singularities:

    - t = 0
    - t = ±T/(4α)

    where T is the symbol period and α is the roll-off factor.
    """

    T = 1.0 / symbol_rate
    # t is the time vector in seconds
    t = (np.arange(ntaps) - (ntaps - 1) / 2.0) / fs

    # x is time normalized by symbol period (Textbook definition)
    x = t / T
    h = np.zeros(ntaps)

    # Singularity at center (t=0)
    h[x == 0] = gain * (1.0 - alpha + (4.0 * alpha / np.pi))

    # Singularities at t = +/- T / (4*alpha)
    # This prevents division by zero in the denominator
    mask_sing = np.isclose(np.abs(4.0 * alpha * x), 1.0)
    if np.any(mask_sing):
        h[mask_sing] = (gain * alpha / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
            (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
        )

    # General Case
    mask_gen = ~((x == 0) | mask_sing)
    # Using normalized x (t/T) makes the math match firdes
    num = gain * (np.sin(np.pi * x[mask_gen] * (1.0 - alpha)) +
                 4.0 * alpha * x[mask_gen] * np.cos(np.pi * x[mask_gen] * (1.0 + alpha)))
    den = np.pi * x[mask_gen] * (1.0 - (4.0 * alpha * x[mask_gen])**2)
    h[mask_gen] = num / den

    return h.astype(np.float32)


class XlatingFilter:
    @staticmethod
    def compute_ntaps_for_hamming(sampling_freq, transition_width):
        # https://github.com/gnuradio/gnuradio/blob/main/gr-fft/lib/window.cc#L53
        # Attenuation in DB for Hamming window
        a = 53
        ntaps = int((a * sampling_freq / (22.0 * transition_width)))
        if (ntaps & 1) == 0:
            ntaps += 1

        return ntaps

    def __init__(self, decimation, samp_rate, freq_offset,
                 cutoff_freq, translation_width, dtype):
        self.decim = decimation
        self.fs = samp_rate
        self.f_off = freq_offset

        # Filter design
        n_taps = XlatingFilter.compute_ntaps_for_hamming(samp_rate, translation_width)
        self.taps = signal.firwin(n_taps, cutoff_freq, fs=samp_rate).astype(dtype)

        # Filter state: last (N-1) samples
        self.zi = np.zeros(len(self.taps) - 1, dtype=dtype)

        # Phase state: oscillator position
        self.phase_acc = 0.0

        # Decimation state: which sample index to take next
        self.decim_offset = 0

    def process(self, samples):
        # STEP A: FREQUENCY SHIFT
        # Calculate time vector starting from 0 for THIS chunk
        n = np.arange(len(samples))

        # Shift = e^(-j * 2pi * f * t + phase_offset)
        # We divide n by self.fs to get actual time in seconds
        phase_vec = 2 * np.pi * self.f_off * (n / self.fs) + self.phase_acc
        shifted = samples * np.exp(-1j * phase_vec)

        # Update phase_acc for the next chunk
        self.phase_acc = (self.phase_acc + 2 * np.pi * self.f_off * len(samples) / self.fs) % (2 * np.pi)

        # STEP B: LOW PASS FILTER
        # Keep state in 'zi' for continuation
        y, self.zi = signal.lfilter(self.taps, 1.0, shifted, zi=self.zi)

        # STEP C: DECIMATION WITH OFFSET
        output = y[self.decim_offset::self.decim]

        # Calculate offset for the next chunk decimation
        total_len = len(y)
        remainder = (total_len - self.decim_offset) % self.decim
        self.decim_offset = (self.decim - remainder) % self.decim

        return output


class QuadDemod:
    def __init__(self, gain):
        self.gain = gain
        # The last complex sample of the previous chunk
        self.last_sample = None

    def process(self, samples):
        # If this is the very first time, we 'prime' it with the first sample
        if self.last_sample is None:
            self.last_sample = samples[0]

        # Prepend the last sample to the current array, this allows us
        # to do vectorized math on all pairs
        extended_samples = np.insert(samples, 0, self.last_sample)

        # Update memory for the next call
        self.last_sample = samples[-1]

        # Calculate phase difference
        product = extended_samples[1:] * np.conj(extended_samples[:-1])
        mpx = np.angle(product) * self.gain

        return mpx


class RationalResampler:
    def __init__(self, up, down, dtype):
        self.up = up
        self.down = down
        self.dtype = dtype

        # Design anti-aliasing filter
        # Cutoff is the minimum of the two Nyquist frequencies
        # Gain is multiplied by 'up' to maintain signal power after zero insertion
        num_taps = 10 * max(up, down) + 1
        cutoff = 1.0 / max(up, down)
        self.taps = (signal.firwin(num_taps, cutoff) * up).astype(self.dtype)

        # Filter state
        self.zi = np.zeros(len(self.taps) - 1, dtype=self.dtype)
        self.offset = 0

    def process(self, samples):
        # Upsample (Zero Insertion)
        upsampled_len = len(samples) * self.up
        up_buf = np.zeros(upsampled_len, dtype=self.dtype)
        up_buf[::self.up] = samples

        # Filter (Streaming State)
        # This smooths the zeros into the final interpolated waveform
        y, self.zi = signal.lfilter(self.taps, 1.0, up_buf, zi=self.zi)

        # Downsample (with persistent offset)
        result = y[self.offset::self.down]

        # Update offset for the next call
        remainder = (len(y) - self.offset) % self.down
        self.offset = (self.down - remainder) % self.down

        return result


class AGC:
    def __init__(self, rate, reference, initial_gain, max_gain):
        self.rate = rate
        self.reference = reference
        self.gain = initial_gain
        self.max_gain = max_gain

    def process(self, samples):
        # We work on a copy to avoid modifying the input in-place unexpectedly
        out = np.zeros_like(samples, dtype=np.complex64)

        for i in range(len(samples)):
            # Apply the current gain
            out[i] = samples[i] * self.gain

            # Calculate the magnitude (signal level)
            mag = np.abs(out[i])

            # Update the gain based on the error
            # Error = Reference - Current Magnitude
            # We move the gain slowly (rate) to minimize the error
            self.gain += self.rate * (self.reference - mag)

            # Clip the gain to the safety limit
            self.gain = np.clip(self.gain, 0.0, self.max_gain)

        return out


# Generate full 128-step polyphase filterbank
def generate_mmse_filterbank(n_filters=128, ntaps=8):
    """Generate MMSE interpolator filterbank using Windowed Sinc"""
    filters = np.zeros((n_filters, ntaps), dtype=np.float32)

    for i in range(n_filters):
        mu = i / float(n_filters)
        for j in range(ntaps):
            t = j - ntaps/2 + mu
            # Sinc with Kaiser window approximation
            if abs(t) < 1e-6:
                filters[i, j] = 1.0
            else:
                filters[i, j] = np.sinc(t)

        # Normalize to preserve DC gain
        s = np.sum(filters[i])
        if abs(s) > 1e-6:
            filters[i] = filters[i] / s

    return filters


@njit(cache=True)
def interpolate_sample(filters, history, mu, n_filters):
    """Interpolate sample at fractional offset mu"""
    imu = int(mu * n_filters + 0.5)
    if imu >= n_filters: imu = n_filters - 1
    elif imu < 0: imu = 0

    result = 0.0 + 0.0j
    ntaps = len(history)
    for i in range(ntaps):
        result += filters[imu, i] * history[i]

    return result


@njit(cache=True)
def ted_zero_crossing_3tap(input_hist, decision_hist):
    """
    GNU Radio TED_ZERO_CROSSING - CORRECTED POLARITY

    Formula: (Oldest - Newest) * Middle

    Indices in our shift array:
      [2] = newest (future)
      [1] = middle (current zero-crossing)
      [0] = oldest (past)
    """
    error_real = (decision_hist[0].real - decision_hist[2].real) * input_hist[1].real
    error_imag = (decision_hist[0].imag - decision_hist[2].imag) * input_hist[1].imag

    return error_real + error_imag


@njit(cache=True)
def slicer_bpsk(sample):
    """BPSK hard decision slicer"""
    real = 1.0 if sample.real >= 0 else -1.0
    imag = 1.0 if sample.imag >= 0 else -1.0
    return complex(real, imag)


@njit(cache=True)
def symbol_sync_core(input_samples, filters, history, input_hist, decision_hist,
                     mu, avg_period, omega_mid, omega_lim, alpha, beta, ted_gain,
                     n_filters, sps, is_symbol_clock):
    """
    Core symbol sync processing - Dual Clock Architecture
    """
    ntaps = len(history)
    n_in = len(input_samples)

    # Pre-allocate output (expecting ~1 sample per symbol, so length / sps)
    output = np.zeros(int(n_in * 2 / sps), dtype=np.complex64)
    out_idx = 0

    max_period = omega_mid + omega_lim
    min_period = omega_mid - omega_lim

    inst_period = avg_period
    interp_period = inst_period / 2.0  # We interpolate at 2x symbol rate

    ii = 0

    while True:
        # Check if we need to consume more input samples to satisfy phase `mu`
        while mu >= 1.0:
            mu = mu - 1.0
            ii += 1
            if ii >= n_in:
                break

            # Shift interpolation history
            for i in range(ntaps - 1, 0, -1):
                history[i] = history[i - 1]
            history[0] = input_samples[ii]

        if ii >= n_in:
            break

        # Interpolate at current mu
        interp_sample = interpolate_sample(filters, history, mu, n_filters)

        # Update TED history (shift left, newest at index 2)
        input_hist[0] = input_hist[1]
        input_hist[1] = input_hist[2]
        input_hist[2] = interp_sample

        # Make decisions
        decision_hist[0] = decision_hist[1]
        decision_hist[1] = decision_hist[2]
        decision_hist[2] = slicer_bpsk(interp_sample)

        # Toggle our internal clock (runs at 2x symbol rate)
        is_symbol_clock = not is_symbol_clock

        # Only evaluate the TED and update output on the actual symbol peak
        if is_symbol_clock:
            output[out_idx] = interp_sample
            out_idx += 1

            # Compute timing error (Oldest - Newest) * Middle
            error = ted_zero_crossing_3tap(input_hist, decision_hist) * ted_gain

            # Update loop filter (2nd order PLL)
            avg_period = avg_period + beta * error

            # Clamp average period
            if avg_period > max_period:
                avg_period = max_period
            elif avg_period < min_period:
                avg_period = min_period

            inst_period = avg_period + alpha * error

        # Advance phase by the half-symbol period
        interp_period = inst_period / 2.0
        mu = mu + interp_period

    return output[:out_idx], mu, avg_period, input_hist, decision_hist, history, is_symbol_clock

class SymbolSync:
    def __init__(self,
                 sps=16.0,
                 loop_bw=0.01,
                 damping=1.0,
                 ted_gain=1.0,
                 max_dev=0.1,
                 osps=1,
                 n_filters=128):

        self.sps = float(sps)
        self.osps = float(osps)
        self.damping = damping
        self.ted_gain = ted_gain
        self.max_dev = max_dev
        self.n_filters = n_filters

        # Nominal period is the total symbol period
        self.omega_mid = self.sps
        self.omega_lim = max_dev * self.sps

        # Loop filter coefficients
        denom = 1.0 + 2.0 * damping * loop_bw + loop_bw * loop_bw
        self.alpha = (4.0 * damping * loop_bw) / denom
        self.beta = (4.0 * loop_bw * loop_bw) / denom

        self.ntaps = 8
        self.filters = generate_mmse_filterbank(n_filters, self.ntaps).astype(np.float32)

        self.reset()

    def reset(self):
        """Reset synchronizer state"""
        self.mu = 0.0
        self.avg_period = self.omega_mid
        self.history = np.zeros(self.ntaps, dtype=np.complex64)
        self.input_hist = np.zeros(3, dtype=np.complex64)
        self.decision_hist = np.zeros(3, dtype=np.complex64)
        # Start False so the first shift creates the middle/zero-crossing sample
        self.is_symbol_clock = False

    def process(self, input_samples):
        if input_samples.dtype != np.complex64:
            input_samples = input_samples.astype(np.complex64)

        output, self.mu, self.avg_period, self.input_hist, self.decision_hist, \
        self.history, self.is_symbol_clock = symbol_sync_core(
                input_samples, self.filters, self.history, self.input_hist,
                self.decision_hist, self.mu, self.avg_period, self.omega_mid,
                self.omega_lim, self.alpha, self.beta, self.ted_gain,
                self.n_filters, self.sps, self.is_symbol_clock
            )

        return output


@njit(cache=True)
def costas_loop_core(samples, const_points, phase, freq, alpha, beta, fmax, fmin):
    """
    Numba-optimized Decision-Directed Costas Loop.

    Args:
        samples: Complex baseband symbols
        const_points: Array of ideal complex constellation points (e.g., BPSK, QPSK)
        phase: Current NCO phase
        freq: Current NCO frequency
        alpha: PLL proportional gain
        beta: PLL integral gain
        fmax, fmin: Frequency clamping limits
    """
    n = len(samples)
    num_points = len(const_points)

    out_symbols = np.zeros(n, dtype=np.complex64)
    out_indices = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        # 1. Derotate the sample by the current NCO phase estimate
        rot_complex = complex(math.cos(-phase), math.sin(-phase))
        rotated = samples[i] * rot_complex
        out_symbols[i] = rotated

        # Slicer: Find the nearest constellation point (Hard Decision)
        min_dist = 1e9
        best_idx = 0
        for j in range(num_points):
            pt = const_points[j]
            # Euclidean distance squared
            dist = (rotated.real - pt.real)**2 + (rotated.imag - pt.imag)**2
            if dist < min_dist:
                min_dist = dist
                best_idx = j

        out_indices[i] = best_idx
        decision = const_points[best_idx]

        # Phase Error Calculation: exact phase difference:
        # angle(rotated * conj(decision))
        cross_real = rotated.real * decision.real + rotated.imag * decision.imag
        cross_imag = rotated.imag * decision.real - rotated.real * decision.imag
        error = math.atan2(cross_imag, cross_real)

        # Loop Filter (2nd Order PLL)
        freq = freq + beta * error

        if freq > fmax:
            freq = fmax
        elif freq < fmin:
            freq = fmin

        phase = phase + freq + alpha * error

        # Wrap phase to [-pi, pi]
        while phase > math.pi:
            phase -= 2 * math.pi
        while phase < -math.pi:
            phase += 2 * math.pi

    return out_symbols, out_indices, phase, freq


class ConstellationReceiver:
    """
    Generic Constellation Receiver supporting any PSK/QAM mapping.
    """
    def __init__(self, constellation_points, loop_bw=2*math.pi/100, fmin=-0.002, fmax=0.002, damping=0.707):
        # Ensure constellation points are complex64
        self.const_points = np.array(constellation_points, dtype=np.complex64)

        self.loop_bw = float(loop_bw)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.damping = float(damping)

        # Calculate 2nd-order PLL coefficients
        denom = 1.0 + 2.0 * self.damping * self.loop_bw + self.loop_bw * self.loop_bw
        self.alpha = (4.0 * self.damping * self.loop_bw) / denom
        self.beta = (4.0 * self.loop_bw * self.loop_bw) / denom

        self.phase = 0.0
        self.freq = 0.0

    def reset(self):
        self.phase = 0.0
        self.freq = 0.0

    def process(self, samples):
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64)

        out_symbols, out_indices, self.phase, self.freq = costas_loop_core(
            samples, self.const_points, self.phase, self.freq,
            self.alpha, self.beta, self.fmax, self.fmin
        )

        return out_indices, out_symbols


@njit(cache=True)
def diff_decoder_core(samples, prev_sample, modulus):
    """
    Numba-optimized differential decoding.
    Formula: out[i] = (samples[i] - prev_sample) % modulus
    """
    n = len(samples)
    out = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        # In Python, negative numbers modulo positive numbers wrap correctly
        # e.g., (0 - 1) % 2 == 1
        out[i] = (samples[i] - prev_sample) % modulus
        prev_sample = samples[i]

    return out, prev_sample


class DiffDecoder:
    """
    Python equivalent of digital.diff_decoder_bb
    Expects and outputs byte/integer arrays (e.g., np.uint8).
    """
    def __init__(self, modulus=2):
        self.modulus = int(modulus)
        self.prev_sample = 0

    def reset(self):
        """Reset the historical state of the decoder."""
        self.prev_sample = 0

    def process(self, samples):
        """
        Process an array of hard-decision integers.
        """
        # Ensure we are working with unsigned 8-bit ints (like _bb in GR)
        if samples.dtype != np.uint8:
            samples = samples.astype(np.uint8)

        out, self.prev_sample = diff_decoder_core(samples, self.prev_sample, self.modulus)

        return out


class RDSPipeline:
    """
    Implements the EXACT signal processing chain from 'rds_rx.grc'.

    Pipeline Stages:
      1. XLating Filter 1: Shift -250k (DC offset removal) -> Decimate by 6.
      2. FM Demod:         Quad demod with gain correction.
      3. XLating Filter 2: Shift -57k (RDS isolation) -> Decimate by 10.
      4. Resampler:        32k -> 19k (Rational Factor 19/32).
      5. Matched Filter:   RRC Taps with Manchester Difference ([n]-[n+8]).
      6. AGC:              Normalize amplitude (Ref 0.585).
      7. Symbol Sync:      Timing Recovery (Zero Crossing).
      8. Costas Loop:      Phase Recovery (BPSK).
      9. Diff Decoder:     NRZ-M Decoding.
    """

    def __init__(self):
        self.sample_rate_hw = 1_920_000  # YAML: samp_rate
        self.rate_if        = self.sample_rate_hw // 6  # 320,000 Hz


        # 1. Frequency-translating FIR filter (complex -> complex)
        #    Shift -250k (DC offset removal) -> low pass (135 kHz) -> decimate by 6
        #    Isolates the whole station
        self.freq_xlating_0 = XlatingFilter(decimation=6,
                                            samp_rate=self.sample_rate_hw,
                                            freq_offset=250_000,
                                            cutoff_freq=135_000,
                                            translation_width=20_000,
                                            dtype=np.complex64)

        # 2. Quadrature demodulator (complex -> float)
        # Gain does two things:
        #  1. Converts to Frequency (Hz) by `angle * (freq / 2π)`
        #  2. Normalizes audio to the maximum volume, which
        #     corresponds to 75 kHz deviation in FM radio (since this
        #     is frequency modulation, loud audio increases frequency,
        #     while silent audio decreases frequency, and in FM the
        #     allowed deviation is 75 kHz).
        demod_gain = (self.rate_if) / (2 * np.pi * 75_000)
        self.quad_demod = QuadDemod(gain=demod_gain)

        # 3. Frequency-translating FIR filter (float -> complex)
        #    Shift -57k (RDS isolation) -> low pass (7.5 kHz) -> decimate by 10
        #    Isolates RDS
        self.freq_xlating_1 = XlatingFilter(decimation=10,
                                            samp_rate=self.rate_if,
                                            freq_offset=57_000, # Center of RDS subcarrier
                                            cutoff_freq=7_500,
                                            translation_width=5_000,
                                            dtype=np.float32)

        # 4. Resampling 32k -> 19k (ratio 19/32)
        self.resampling_32k_19k = RationalResampler(up=19, down=32,
                                                    dtype=np.complex64)

        # 5. RRC Manchester Matched Filter
        rrc_taps = generate_rrc_taps(
            gain=1.0,
            fs=19_000,
            symbol_rate=19_000 // 8, # RDS symbol (chip) rate
            alpha=1.0,               # Alpha
            ntaps=151
        )
        # Create difference filter that looks for the transition from
        # positive to negative (or vice versa).
        # "rrc_taps[n]" matches the first half of the bit (the first chip).
        # "-rrc_taps[n+8]" matches the second half of the bit (the second chip,
        #  which is inverted).
        self.taps_manchester = rrc_taps[:-8] - rrc_taps[8:]
        self.zi_manchester   = np.zeros(len(self.taps_manchester) - 1, dtype=np.complex64)

        # 6. AGC (Automatic Gain Control) ensures the BPSK signal
        # always has a consistent amplitude
        self.agc = AGC(rate=2e-3, reference=0.585, initial_gain=53, max_gain=1000)

        # 7. Symbol Sync - Timing Error Detector (TED) - looks
        # at three points: the current sample, the previous sample,
        # and the "midpoint" between them. If the midpoint isn't a
        # zero-crossing, it knows the clock is drifting and adjusts
        # the "sampling strobe".
        self.symbol_sync = SymbolSync(sps=16)

        # 8. Constellation Receiver makes hard decisions about the
        # received symbols (using a constellation BPSK points) and
        # also fine tunes phase synchronization. The phase and
        # frequency synchronization are based on a Costas loop that
        # finds the error of the incoming signal point compared to its
        # nearest constellation point.
        # Constellation: [-1, +1]
        bpsk_points = [-1.0 + 0j, 1.0 + 0j]
        self.constellation_receiver = ConstellationReceiver(bpsk_points)

        # 9. Differential Decoder uses current and previous symbols
        # and the alphabet modulus (2 for BPSK) to perform
        # differential decoding: y[0] = (x[0] - x[-1]) % 2.
        self.diff_decoder = DiffDecoder(modulus=2)

        # 10. RDS Decoder and Parser
        # https://github.com/bastibl/gr-rds/tree/maint-3.10/lib
        self.rds_decoder = RDSDecoder()
        self.rds_parser = RDSParser(pty_locale=1) # 0=US, 1=EU


    def process(self, samples):
        # STEP 1: OFFSET TUNING CORRECTION & DECIMATION
        # YAML: freq_xlating_fir_filter_xxx_0 (Decim 6)
        samples = self.freq_xlating_0.process(samples).astype(np.complex64)

        # STEP 2: FM DEMODULATION
        # YAML: analog_quadrature_demod_cf_0
        samples = self.quad_demod.process(samples).astype(np.float32)

        # STEP 3: RDS ISOLATION & DECIMATION
        # YAML: freq_xlating_fir_filter_xxx_1_0 (Decim 10)
        samples = self.freq_xlating_1.process(samples).astype(np.complex64)

        # STEP 4: EXACT RESAMPLING
        # YAML: rational_resampler_xxx_1
        samples = self.resampling_32k_19k.process(samples).astype(np.complex64)

        # STEP 5: MANCHESTER MATCHED FILTER
        # YAML: fir_filter_xxx_2
        samples, self.zi_manchester = signal.lfilter(self.taps_manchester, 1.0, samples,
                                                     zi=self.zi_manchester)

        # STEP 6: AGC LOOP
        # YAML: analog_agc_xx_0 (rate=2e-3, ref=0.585)
        samples = self.agc.process(samples)

        # STEP 7: SYMBOL SYNC (Timing Recovery)
        # YAML: digital_symbol_sync_xx_0 (TED: ZeroCrossing, SPS: 16)
        samples = self.symbol_sync.process(samples)

        # STEP 8: COSTAS LOOP
        # YAML: digital_constellation_receiver_cb_0 (BW=2pi/100)
        bits, symbols = self.constellation_receiver.process(samples)

        # STEP 9: DIFF DECODER
        # YAML: digital_diff_decoder_bb_0
        bits = self.diff_decoder.process(bits)

        # STEP 10: RDS DECODER/PARSER
        groups = self.rds_decoder.process(bits)
        self.rds_parser.process(groups)

        return symbols


class RDSAndConstellationMonitor:
    def __init__(self, rds_parser, window_size=200, print_interval=0.2):
        self.rds_parser = rds_parser
        self.window_size = window_size
        self.print_interval = print_interval
        self.constellation = []
        self.last_print = 0
        self.n_calls = 0
        # Use Braille patterns for representing progress,
        # see here: https://www.unicode.org/charts/nameslist/c_2800.html
        self.progress = [0x2826, 0x2816, 0x2832, 0x2834]

    def report(self, symbols):
        self.constellation.extend(symbols)
        now = time.time()
        if now - self.last_print < self.print_interval:
            return
        self.last_print = now

        pts = np.array(self.constellation[-self.window_size:])
        hist, _ = np.histogram(np.real(pts), bins=20, range=(-2, 2))

        graph = ""
        for h in hist:
            if h > 10:
                graph += "#"
            elif h > 2:
                graph += "."
            else:
                graph += "_"

        self.n_calls += 1

        state = self.rds_parser.get_state()
        progress = chr(self.progress[self.n_calls % len(self.progress)])

        print(f"\r{progress} CONSTELLATION: [{graph}] | RadioText: {state['radiotext']} \033[K", end="", flush=True)

        # Keep only the rolling window
        self.constellation = self.constellation[-self.window_size:]


async def processing_pipeline(queue, loop, pool, rds_pipeline):
    rds_monitor = RDSAndConstellationMonitor(rds_pipeline.rds_parser)
    while True:
        # Wait for data (yields control if empty)
        raw_samples = await queue.get()

        # await for thread to finish with preprocessed result
        symbols = await loop.run_in_executor(
            pool,
            rds_pipeline.process,
            raw_samples
        )
        rds_monitor.report(symbols)
        queue.task_done()


async def run_pipeline(sdr, samples_chunk_size):
    # Buffer size: Decouples reading from processing bursts
    queue = asyncio.Queue(maxsize=50)
    loop = asyncio.get_running_loop()

    # Create a ThreadPool for the heavy math
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rds_pipeline = RDSPipeline()

        # Start the background worker
        worker_task = asyncio.create_task(
            processing_pipeline(queue, loop, pool, rds_pipeline)
        )

        try:
            async for samples in sdr.stream(num_samples_or_bytes=samples_chunk_size):
                await queue.put(samples)

        finally:
            # Cleanup
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM RDS receiver (pure Python)")
    parser.add_argument("--fm-freq", type=float, required=True,
                        metavar="MHz", help="FM station frequency in MHz (e.g. 103.4)")
    args = parser.parse_args()

    print("Starting RDS Receiver...")

    fm_freq = args.fm_freq * 1e6
    freq_offset = 250_000
    samp_rate = 1_920_000
    samples_chunk_size = 4096

    sdr = RtlSdrAio(device_index=0)
    sdr.center_freq = fm_freq - freq_offset
    sdr.sample_rate = samp_rate
    sdr.gain = 25

    asyncio.run(run_pipeline(sdr, samples_chunk_size))
