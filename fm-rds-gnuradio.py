#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys
import time
import numpy as np
from gnuradio import gr, filter, analog, digital, blocks
import osmosdr
import pmt

try:
    import rds
except ImportError:
    print("Error: gr-rds not installed")
    sys.exit(1)


class RDSAndConstellationMonitor(gr.sync_block):
    """
    Hierarchical block that combines:
    - Constellation diagnostics (stream of complex samples)
    - RDS message handling (from rds_parser message port)
    """
    def __init__(self, window_size=200, print_interval=0.2):
        gr.sync_block.__init__(self,
            name="RDSAndConstellationMonitor",
            in_sig=[np.complex64],
            out_sig=None # no stream output
        )


        self.window_size = window_size
        self.print_interval = print_interval
        self.constellation = []
        self.last_print = 0
        self.n_calls = 0
        self.text = ""
        # Use Braille patterns for representing progress,
        # see here: https://www.unicode.org/charts/nameslist/c_2800.html
        self.progress = [0x2826, 0x2816, 0x2832, 0x2834]

        # Message input port for RDS
        self.message_port_register_in(pmt.intern('rds_in'))
        self.set_msg_handler(pmt.intern('rds_in'), self.handle_rds)

    # ---------- Stream handler ----------
    def work(self, input_items, output_items):
        samples = input_items[0]
        self.constellation.extend(samples)
        now = time.time()

        if now - self.last_print >= self.print_interval:
            self.last_print = now
            self.report_constellation()
        return len(samples)

    def report_constellation(self):
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
        progress = chr(self.progress[self.n_calls % len(self.progress)])

        print(f"\r{progress} CONSTELLATION: [{graph}] | RadioText: {self.text} \033[K", end="", flush=True)

        # Keep only the rolling window
        self.constellation = self.constellation[-self.window_size:]

    # ---------- Message handler ----------
    def handle_rds(self, msg_pmt):
        """
        Called when rds_parser sends a message.
        """
        if pmt.is_dict(msg_pmt):
            data = pmt.to_python(msg_pmt)
            # Print only the most useful fields for readability
            pi = data.get('pi', '?')
            ps = data.get('ps', '?')
            pty = data.get('pty', '?')

            #print(f" | RDS: PI={pi} PS={ps} PTY={pty}", end="", flush=True)
        else:
            block_index, ps_text = pmt.to_python(msg_pmt)
            if block_index == 4:
                # Block D, RadioText
                self.text = ps_text.strip()


class RDSReceiver(gr.hier_block2):
    """
    RDS Receiver with complex IQ input, following the YAML flowgraph.
    """

    def __init__(self, samp_rate, freq_offset, decimation):
        gr.hier_block2.__init__(
            self,
            "RDSReceiver",
            gr.io_signature(1, 1, gr.sizeof_gr_complex),
            gr.io_signature(0, 0, 0)
        )

        # First frequency-translating FIR filter (complex -> complex)
        self.freq_xlating_0 = filter.freq_xlating_fir_filter_ccc(
            decimation,
            filter.firdes.low_pass(1, samp_rate, 135000, 20000),
            freq_offset,
            samp_rate
        )

        # Quadrature demodulator (complex -> float)
        gain_demod = (samp_rate / decimation) / (2 * math.pi * 75000)
        self.quadrature_demod = analog.quadrature_demod_cf(gain=gain_demod)

        # Second frequency-translating FIR filter (float -> complex)
        decim2 = 10
        self.freq_xlating_1 = filter.freq_xlating_fir_filter_fcc(
            decim2,
            filter.firdes.low_pass(1.0, samp_rate / decimation, 7.5e3, 5e3),
            57e3,
            samp_rate / decimation
        )

        # Rational resampler to 19 kHz
        interp = 19000
        decim_resamp = int(samp_rate // decimation // 10)
        self.resampler = filter.rational_resampler_ccc(
            interpolation=interp,
            decimation=decim_resamp,
            taps=[],
            fractional_bw=0
        )

        # Standard RRC taps
        rrc_taps = filter.firdes.root_raised_cosine(
            1.0,              # Gain
            19000,            # Sample rate
            19000 // 8,       # Symbol rate
            1.0,              # Alpha, roll-off factor
            151               # Number of taps
        )

        # Manchester differential taps
        rrc_taps_manchester = [rrc_taps[n] - rrc_taps[n+8] for n in range(len(rrc_taps)-8)]

        # FIR filter for Manchester
        self.fir_filter = filter.fir_filter_ccc(1, rrc_taps_manchester)

        # AGC
        self.agc = analog.agc_cc(rate=2e-3, reference=0.585, gain=53)
        self.agc.set_max_gain(1000)

        # Symbol sync
        self.symbol_sync = digital.symbol_sync_cc(
            detector_type=digital.TED_ZERO_CROSSING,
            sps=16,
            loop_bw=0.01,
            damping_factor=1.0,
            ted_gain=1.0,
            max_deviation=0.1,
            osps=1,
            slicer=digital.constellation_bpsk().base(),
            interp_type=digital.IR_MMSE_8TAP,
            n_filters=128,
            taps=[],
        )

        # Constellation receiver
        self.constellation_receiver = digital.constellation_receiver_cb(
            digital.constellation_bpsk().base(),
            loop_bw=2 * math.pi / 100,
            fmin=-0.002,
            fmax=0.002
        )

        # Differential decoder
        self.diff_decoder = digital.diff_decoder_bb(2)

        # RDS decoder + parser
        self.rds_decoder = rds.decoder(debug=False, log=False)
        self.rds_parser = rds.parser(debug=False, log=False, pty_locale=0)

        # =================================================================
        # Connections: matches YAML topology
        # =================================================================
        self.connect(self, self.freq_xlating_0)
        self.connect(self.freq_xlating_0, self.quadrature_demod)
        self.connect(self.quadrature_demod, self.freq_xlating_1)
        self.connect(self.freq_xlating_1, self.resampler)
        self.connect(self.resampler, self.fir_filter)
        self.connect(self.fir_filter, self.agc)
        self.connect(self.agc, self.symbol_sync)
        self.connect(self.symbol_sync, self.constellation_receiver)
        self.connect(self.constellation_receiver, self.diff_decoder)
        self.connect(self.diff_decoder, self.rds_decoder)
        self.msg_connect((self.rds_decoder, 'out'), (self.rds_parser, 'in'))

        # Null sinks for unused outputs
        self.null1 = blocks.null_sink(gr.sizeof_int)
        self.null2 = blocks.null_sink(gr.sizeof_int)
        self.null3 = blocks.null_sink(gr.sizeof_int)

        self.connect((self.constellation_receiver, 1), self.null1)
        self.connect((self.constellation_receiver, 2), self.null2)
        self.connect((self.constellation_receiver, 3), self.null3)

        # Constellation diagnostics and RDS messages
        self.rds_monitor = RDSAndConstellationMonitor(window_size=200, print_interval=0.2)
        self.connect((self.constellation_receiver, 4), (self.rds_monitor, 0))
        self.msg_connect((self.rds_parser, 'out'), (self.rds_monitor, 'rds_in'))


# =================================================================
# Top block with RTL-SDR
# =================================================================
class ExampleTopBlock(gr.top_block):
    def __init__(self, freq):
        gr.top_block.__init__(self, "RDS Top Block")

        freq_offset = 250_000
        samp_rate = 1_920_000
        decimation = 6

        rf_gain = 25
        if_gain = 20
        bb_gain = 20

        # With offset to avoid DC
        freq_tune = freq - freq_offset

        # RTL-SDR source
        self.source = osmosdr.source(args="rtl=0,numchan=1")
        self.source.set_sample_rate(samp_rate)
        self.source.set_center_freq(freq_tune, 0)
        self.source.set_gain_mode(False, 0)
        self.source.set_gain(rf_gain, 0)
        self.source.set_if_gain(if_gain, 0)
        self.source.set_bb_gain(bb_gain, 0)
        self.source.set_bandwidth(0)
        self.source.set_freq_corr(0, 0)
        self.source.set_dc_offset_mode(0, 0)
        self.source.set_iq_balance_mode(0, 0)

        # Instantiate RDS Receiver
        self.rds = RDSReceiver(samp_rate=samp_rate,
                               freq_offset=freq_offset,
                               decimation=decimation)

        # Connect source to hierarchical RDS block
        self.connect(self.source, self.rds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FM RDS receiver (GNURadio)")
    parser.add_argument("--fm-freq", type=float, required=True,
                        metavar="MHz", help="FM station frequency in MHz (e.g. 103.4)")
    args = parser.parse_args()

    print("Starting RDS Receiver...")
    tb = ExampleTopBlock(freq=args.fm_freq * 1e6)
    tb.start()
    print("Running. Press Enter to stop.")
    input()
    tb.stop()
    tb.wait()
