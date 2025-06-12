# parse_fields.py
import numpy as np

payload = open("results/decoded/20250612_114704/payload.bin","rb").read()
b = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

def slice_bits(bits, start, length):
    return int("".join(str(bits[i]) for i in range(start, start+length)), 2)

length_field = slice_bits(b,  0, 12)
msg_type     = slice_bits(b, 12,  8)
flags        = slice_bits(b, 20,  4)

for shift in range(8):
    b = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    # shift the bitstream
    b_shifted = b[shift:] 
    length = int(''.join(str(x) for x in b_shifted[0:12]), 2)
    msg_type = int(''.join(str(x) for x in b_shifted[12:20]), 2)
    if length < 500:  # arbitrary sane cutoff
        print(f"shift {shift}: Length={length}, Type={msg_type}")


print(f"Length = {length_field}")
print(f"Type   = {msg_type}")
print(f"Flags  = {flags:04b}")
