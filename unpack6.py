# unpack6.py
import numpy as np

data = open("results/decoded/20250612_114704/payload.bin","rb").read()
bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
# Suppose the protocol packs 6-bit characters, MSB first:
chars = []
for i in range(0, len(bits)//6):
    val = int("".join(str(b) for b in bits[i*6:(i+1)*6]), 2)
    # map 0–63 to ASCII: e.g. 1–26→A–Z, 27→space, 28–53→0–9, etc.
    if 1 <= val <= 26:
        chars.append(chr(ord('A') + val - 1))
    elif val == 27:
        chars.append(' ')
    elif 28 <= val <= 37:
        chars.append(chr(ord('0') + val - 28))
    else:
        chars.append('?')
print("".join(chars))
