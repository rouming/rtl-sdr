import numpy as np
from numba import int32, uint32, int64, boolean, uint16, uint8, float64
from numba.experimental import jitclass


# Annex F of RBDS Standard Table F.1 (US) and Table F.2 (EU)
# Format: PTY_TABLE[code][locale] where locale 0=US, 1=EU
PTY_TABLE = [
    ("Undefined",             "Undefined"),
    ("News",                  "News"),
    ("Current Affairs",       "Information"),
    ("Information",           "Sports"),
    ("Sport",                 "Talk"),
    ("Education",             "Rock"),
    ("Drama",                 "Classic Rock"),
    ("Culture",               "Adult Hits"),
    ("Science",               "Soft Rock"),
    ("Varied",                "Top 40"),
    ("Pop Music",             "Country"),
    ("Rock Music",            "Oldies"),
    ("Easy Listening",        "Soft"),
    ("Light Classical",       "Nostalgia"),
    ("Serious Classical",     "Jazz"),
    ("Other Music",           "Classical"),
    ("Weather",               "Rhythm & Blues"),
    ("Finance",               "Soft Rhythm & Blues"),
    ("Children's Programmes", "Language"),
    ("Social Affairs",        "Religious Music"),
    ("Religion",              "Religious Talk"),
    ("Phone-In",              "Personality"),
    ("Travel",                "Public"),
    ("Leisure",               "College"),
    ("Jazz Music",            "Spanish Talk"),
    ("Country Music",         "Spanish Music"),
    ("National Music",        "Hip Hop"),
    ("Oldies Music",          "Unassigned"),
    ("Folk Music",            "Unassigned"),
    ("Documentary",           "Weather"),
    ("Alarm Test",            "Emergency Test"),
    ("Alarm",                 "Emergency")
]

# page 71, Annex D, table D.1 in the standard
PI_COUNTRY_CODES = [
    ["DE","GR","MA","__","MD"], ["DZ","CY","CZ","IE","EE"], ["AD","SM","PL","TR","__"],
    ["IL","CH","VA","MK","__"], ["IT","JO","SK","__","__"], ["BE","FI","SY","__","UA"],
    ["RU","LU","TN","__","__"], ["PS","BG","__","NL","PT"], ["AL","DK","LI","LV","SI"],
    ["AT","GI","IS","LB","__"], ["HU","IQ","MC","__","__"], ["MT","GB","LT","HR","__"],
    ["DE","LY","YU","__","__"], ["__","RO","ES","SE","__"], ["EG","FR","NO","BY","BA"]
]

COVERAGE_AREA_CODES = [
    "Local", "International", "National", "Supra-regional", "Regional 1",
    "Regional 2", "Regional 3", "Regional 4", "Regional 5", "Regional 6",
    "Regional 7", "Regional 8", "Regional 9", "Regional 10", "Regional 11", "Regional 12"
]

@jitclass
class RDSParserCore:
    program_identification: uint32
    program_type: uint8
    radiotext: uint8[:]
    radiotext_segment_flags: uint32
    program_service_name: uint8[:]
    program_service_name_segment_flags: uint32
    radiotext_AB_flag: uint8
    traffic_program: boolean
    traffic_announcement: boolean
    music_speech: boolean
    mono_stereo: boolean
    artificial_head: boolean
    compressed: boolean
    dynamic_pty: boolean

    # Flags to notify the Python wrapper that a string is ready to print
    ps_ready: boolean
    rt_ready: boolean
    ct_ready: boolean

    # Clock Time States
    ct_year: uint32
    ct_month: uint32
    ct_day: uint32
    ct_hour: uint32
    ct_minute: uint32
    ct_offset: float64

    def __init__(self):
        self.program_service_name = np.full(8, 32, dtype=np.uint8) # 32 = Space
        self.radiotext = np.full(64, 32, dtype=np.uint8)
        self.reset()

    def reset(self):
        """Reset internal decoder state"""
        self.radiotext_segment_flags = 0
        self.program_service_name_segment_flags = 0
        self.radiotext_AB_flag = 0
        self.traffic_program = False
        self.traffic_announcement = False
        self.music_speech = False
        self.program_identification = 0xFFFFFFFF
        self.program_type = 0
        self.mono_stereo = False
        self.artificial_head = False
        self.compressed = False
        self.dynamic_pty = False
        self.ps_ready = False
        self.rt_ready = False
        self.ct_ready = False

    def decode_type0(self, group, B):
        """BASIC TUNING: decodes Program Service Name and flags"""
        self.traffic_program = (group[1] >> 10) & 0x01
        self.traffic_announcement = (group[1] >> 4) & 0x01
        self.music_speech = (group[1] >> 3) & 0x01

        decoder_control_bit = (group[1] >> 2) & 0x01
        segment_address = group[1] & 0x03

        ps_1 = (group[3] >> 8) & 0xFF
        ps_2 = group[3] & 0xFF

        if self.program_service_name_segment_flags & (1 << segment_address):
            # Check if chars changed
            if (self.program_service_name[segment_address * 2] != ps_1 or
                self.program_service_name[segment_address * 2 + 1] != ps_2):
                self.program_service_name[segment_address * 2] = ps_1
                self.program_service_name[segment_address * 2 + 1] = ps_2
                self.program_service_name_segment_flags = (1 << segment_address)
        else:
            self.program_service_name[segment_address * 2] = ps_1
            self.program_service_name[segment_address * 2 + 1] = ps_2
            self.program_service_name_segment_flags |= (1 << segment_address)

            # If all 4 segments (8 chars) are received (0xF = 1111)
            if self.program_service_name_segment_flags == 0xF:
                self.ps_ready = True

        # Decode DI flags
        if segment_address == 0: self.dynamic_pty = decoder_control_bit
        elif segment_address == 1: self.compressed = decoder_control_bit
        elif segment_address == 2: self.artificial_head = decoder_control_bit
        elif segment_address == 3: self.mono_stereo = decoder_control_bit

    def decode_type2(self, group, B):
        """RADIO TEXT"""
        text_segment_address_code = group[1] & 0x0F
        new_ab_flag = (group[1] >> 4) & 0x01

        # Flush if A/B flag toggled
        if self.radiotext_AB_flag != new_ab_flag:
            self.radiotext_segment_flags = 0
            for i in range(64):
                self.radiotext[i] = 32 # Fill with spaces

        self.radiotext_AB_flag = new_ab_flag

        if not B: # Type 2A (4 chars)
            self.radiotext[text_segment_address_code * 4] = (group[2] >> 8) & 0xFF
            self.radiotext[text_segment_address_code * 4 + 1] = group[2] & 0xFF
            self.radiotext[text_segment_address_code * 4 + 2] = (group[3] >> 8) & 0xFF
            self.radiotext[text_segment_address_code * 4 + 3] = group[3] & 0xFF
        else: # Type 2B (2 chars)
            self.radiotext[text_segment_address_code * 2] = (group[3] >> 8) & 0xFF
            self.radiotext[text_segment_address_code * 2 + 1] = group[3] & 0xFF

        self.radiotext_segment_flags |= (1 << text_segment_address_code)
        self.rt_ready = True # In Python wrapper, we will check for completeness/tail

    def decode_type4(self, group, B):
        """CLOCK TIME and DATE"""
        if B: return

        self.ct_hour = ((group[2] & 0x1) << 4) | ((group[3] >> 12) & 0x0F)
        self.ct_minute = (group[3] >> 6) & 0x3F
        self.ct_offset = 0.5 * (group[3] & 0x1F)

        if (group[3] >> 5) & 0x1:
            self.ct_offset *= -1.0

        mjd = ((group[1] & 0x03) << 15) | ((group[2] >> 1) & 0x7FFF)

        # MJD to Date conversion
        year = int((mjd - 15078.2) / 365.25)
        month = int((mjd - 14956.1 - int(year * 365.25)) / 30.6001)
        day = int(mjd - 14956 - int(year * 365.25) - int(month * 30.6001))

        K = 1 if (month == 14 or month == 15) else 0
        self.ct_year = 1900 + year + K
        self.ct_month = month - 1 - K * 12
        self.ct_day = day

        self.ct_ready = True

    def parse_group(self, bytes_arr):
        """Main entry point for a 12-byte group"""
        group = np.zeros(4, dtype=np.uint16)
        group[0] = bytes_arr[1] | (bytes_arr[0] << 8)
        group[1] = bytes_arr[3] | (bytes_arr[2] << 8)
        group[2] = bytes_arr[5] | (bytes_arr[4] << 8)
        group[3] = bytes_arr[7] | (bytes_arr[6] << 8)

        group_type = (group[1] >> 12) & 0x0F
        ab = True if ((group[1] >> 11) & 0x1) else False

        if self.program_identification != group[0]:
            self.reset()

        self.program_identification = group[0]
        self.program_type = (group[1] >> 5) & 0x1F

        if group_type == 0:
            self.decode_type0(group, ab)
        elif group_type == 2:
            self.decode_type2(group, ab)
        elif group_type == 4:
            self.decode_type4(group, ab)


class RDSParser:
    """
    Python wrapper that uses the JIT core and maintains the current RDS state.
    """
    def __init__(self, pty_locale=1): # 0=US, 1=EU
        self.core = RDSParserCore()
        self.pty_locale = pty_locale
        self.current_pi = None

        # Maintain the current state of all parsed information
        self.state = {
            "pi_code": None,
            "program_type": None,
            "country": None,
            "area": None,
            "program_reference_number": None,
            "program_service_name": "",
            "radiotext": "",
            "radiotext_ab": None,
            "clock_time": None
        }

    def get_state(self):
        """Returns the current parsed RDS state dictionary."""
        return self.state

    def process(self, decoded_groups):
        """
        Expects a 2D numpy array of shape (N, 12) from the Decoder block.
        Updates the internal state dictionary.
        """
        for i in range(decoded_groups.shape[0]):
            self.core.parse_group(decoded_groups[i])

            # 1. Update Station Details if PI changed or initialized
            if self.current_pi != self.core.program_identification:
                self.current_pi = self.core.program_identification
                self.state["pi_code"] = f"{self.current_pi:04X}"

                # Extract Country/Area info
                country_idx = ((self.current_pi >> 12) & 0xF) - 1
                area_idx = (self.current_pi >> 8) & 0xF
                prog_num = self.current_pi & 0xFF

                self.state["country"] = PI_COUNTRY_CODES[country_idx][0] if country_idx >= 0 else "??"
                self.state["area"] = COVERAGE_AREA_CODES[area_idx]
                self.state["program_type"] = PTY_TABLE[self.core.program_type][self.pty_locale]
                self.state["program_reference_number"] = prog_num

            # 2. Extract Program Service Name (PS)
            if self.core.ps_ready:
                # Convert uint8 array to ASCII string, ignoring invalid characters
                ps_str = "".join([chr(c) if 32 <= c <= 126 else ' ' for c in self.core.program_service_name])
                self.state["program_service_name"] = ps_str.strip()
                self.core.ps_ready = False # Reset flag

            # 3. Extract RadioText (RT)
            if self.core.rt_ready:
                # Check how many segments we actually have
                valid_segments = 0
                for seg in range(16):
                    if self.core.radiotext_segment_flags & (1 << seg):
                        valid_segments += 1
                    else:
                        break

                # Check for carriage return (0x0D) to terminate early
                rt_bytes = list(self.core.radiotext)
                tail_idx = valid_segments * 4 # Max chars valid
                if 13 in rt_bytes[:tail_idx]:
                    tail_idx = rt_bytes.index(13)

                if tail_idx > 0 and (tail_idx == 64 or 13 in rt_bytes):
                    rt_str = "".join([chr(c) if 32 <= c <= 126 else ' ' for c in rt_bytes[:tail_idx]])
                    self.state["radiotext"] = rt_str.strip()
                    self.state["radiotext_ab"] = 'B' if self.core.radiotext_AB_flag else 'A'

                self.core.rt_ready = False

            # 4. Extract Clock Time (CT)
            if self.core.ct_ready:
                time_str = f"{self.core.ct_day:02d}.{self.core.ct_month:02d}.{self.core.ct_year:04d} {self.core.ct_hour:02d}:{self.core.ct_minute:02d} ({self.core.ct_offset:+.1f}h)"
                self.state["clock_time"] = time_str
                self.core.ct_ready = False
