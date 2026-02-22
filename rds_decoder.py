import numpy as np
from numba import njit, int32, uint32, int64, boolean, uint16, uint8
from numba.experimental import jitclass

@njit(cache=True)
def calc_syndrome(message, mlen):
    """
    Calculate the 10-bit syndrome for RDS error correction.
    Matches Annex B, page 64 of the RDS standard.
    """
    reg = 0
    poly = 0x5B9
    plen = 10

    for i in range(mlen, 0, -1):
        reg = (reg << 1) | ((message >> (i - 1)) & 0x01)
        if reg & (1 << plen):
            reg = reg ^ poly

    for i in range(plen, 0, -1):
        reg = reg << 1
        if reg & (1 << plen):
            reg = reg ^ poly

    return reg & ((1 << plen) - 1)


@jitclass
class RDSDecoder:
    # State variables mapped directly from GNU Radio's decoder_impl.h
    d_state: int32
    reg: uint32
    bit_counter: int64
    presync: boolean
    lastseen_offset: uint8
    lastseen_offset_counter: int64
    block_bit_counter: uint32
    block_number: uint8
    wrong_blocks_counter: uint32
    blocks_counter: uint32
    group_assembly_started: boolean
    group_good_blocks_counter: uint32
    group: uint16[:]
    offset_chars: uint8[:]

    def __init__(self):
        # Initialize internal arrays
        self.group = np.zeros(4, dtype=np.uint16)
        self.offset_chars = np.zeros(4, dtype=np.uint8)
        self.enter_no_sync()

    def enter_no_sync(self):
        self.d_state = 0
        self.reg = 0
        self.bit_counter = 0
        self.presync = False
        self.lastseen_offset = 0
        self.lastseen_offset_counter = 0
        self.block_bit_counter = 0
        self.block_number = 0
        self.wrong_blocks_counter = 0
        self.blocks_counter = 0
        self.group_assembly_started = False
        self.group_good_blocks_counter = 0

        for i in range(4):
            self.group[i] = 0
            self.offset_chars[i] = 0

    def enter_sync(self, sync_block_number):
        """Transition to SYNC state."""
        self.wrong_blocks_counter = 0
        self.blocks_counter = 0
        self.block_bit_counter = 0
        self.block_number = (sync_block_number + 1) % 4
        self.group_assembly_started = False
        self.d_state = 1  # 1 = SYNC

    def process(self, bits):
        """
        Main work function. Processes a continuous array of bits.
        Returns a 2D array of decoded 12-byte groups.
        """
        # RDS Constants
        offset_pos = np.array([0, 1, 2, 3, 2], dtype=np.int32)
        offset_word = np.array([252, 408, 360, 436, 848], dtype=np.uint32)
        syndrome = np.array([383, 14, 303, 663, 748], dtype=np.uint32)

        n_bits = len(bits)
        # Pre-allocate output buffer for the maximum possible number of groups
        max_groups = (n_bits // 104) + 1
        out_groups = np.zeros((max_groups, 12), dtype=np.uint8)
        out_idx = 0

        for i in range(n_bits):
            # Shift in the next bit into the 26-bit register
            self.reg = ((self.reg << 1) | bits[i]) & 0xFFFFFFFF

            if self.d_state == 0:  # NO_SYNC
                reg_syndrome = calc_syndrome(self.reg, 26)
                for j in range(5):
                    if reg_syndrome == syndrome[j]:
                        if not self.presync:
                            self.lastseen_offset = j
                            self.lastseen_offset_counter = self.bit_counter
                            self.presync = True
                        else:
                            bit_distance = self.bit_counter - self.lastseen_offset_counter
                            if offset_pos[self.lastseen_offset] >= offset_pos[j]:
                                block_distance = offset_pos[j] + 4 - offset_pos[self.lastseen_offset]
                            else:
                                block_distance = offset_pos[j] - offset_pos[self.lastseen_offset]

                            if (block_distance * 26) != bit_distance:
                                self.presync = False
                            else:
                                self.enter_sync(j)
                        break  # syndrome found, no more cycles

            else:  # SYNC
                # Wait until 26 bits enter the buffer
                if self.block_bit_counter < 25:
                    self.block_bit_counter += 1
                else:
                    good_block = False
                    dataword = (self.reg >> 10) & 0xFFFF
                    block_calculated_crc = calc_syndrome(dataword, 16)
                    checkword = self.reg & 0x3FF
                    offset_char = 120  # ASCII 'x' (error)

                    # Manage special case of C or C' offset word
                    if self.block_number == 2:
                        block_received_crc = checkword ^ offset_word[2]
                        if block_received_crc == block_calculated_crc:
                            good_block = True
                            offset_char = 67  # 'C'
                        else:
                            block_received_crc = checkword ^ offset_word[4]
                            if block_received_crc == block_calculated_crc:
                                good_block = True
                                offset_char = 99  # 'c'
                            else:
                                self.wrong_blocks_counter += 1
                                good_block = False
                    else:
                        block_received_crc = checkword ^ offset_word[self.block_number]
                        if block_received_crc == block_calculated_crc:
                            good_block = True
                            if self.block_number == 0: offset_char = 65    # 'A'
                            elif self.block_number == 1: offset_char = 66  # 'B'
                            elif self.block_number == 3: offset_char = 68  # 'D'
                        else:
                            self.wrong_blocks_counter += 1
                            good_block = False

                    if self.block_number == 0 and good_block:
                        self.group_assembly_started = True
                        self.group_good_blocks_counter = 1

                    if self.group_assembly_started:
                        if not good_block:
                            self.group_assembly_started = False
                        else:
                            self.group[self.block_number] = dataword
                            self.offset_chars[self.block_number] = offset_char
                            self.group_good_blocks_counter += 1

                        if self.group_good_blocks_counter == 5:
                            # Decode group into 12 bytes
                            out_groups[out_idx, 0] = (self.group[0] >> 8) & 0xFF
                            out_groups[out_idx, 1] = self.group[0] & 0xFF
                            out_groups[out_idx, 2] = (self.group[1] >> 8) & 0xFF
                            out_groups[out_idx, 3] = self.group[1] & 0xFF
                            out_groups[out_idx, 4] = (self.group[2] >> 8) & 0xFF
                            out_groups[out_idx, 5] = self.group[2] & 0xFF
                            out_groups[out_idx, 6] = (self.group[3] >> 8) & 0xFF
                            out_groups[out_idx, 7] = self.group[3] & 0xFF
                            out_groups[out_idx, 8] = self.offset_chars[0]
                            out_groups[out_idx, 9] = self.offset_chars[1]
                            out_groups[out_idx, 10] = self.offset_chars[2]
                            out_groups[out_idx, 11] = self.offset_chars[3]
                            out_idx += 1

                    self.block_bit_counter = 0
                    self.block_number = (self.block_number + 1) % 4
                    self.blocks_counter += 1

                    # 1187.5 bps / 104 bits = 45.7 blocks/sec
                    if self.blocks_counter == 50:
                        if self.wrong_blocks_counter > 35:
                            self.enter_no_sync()
                        else:
                            self.blocks_counter = 0
                            self.wrong_blocks_counter = 0

            self.bit_counter += 1

        return out_groups[:out_idx]
