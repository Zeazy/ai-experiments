import torch


with open("input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(" ".join(chars))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encode - take a string and output a list of integers
decode = lambda s: "".join([itos[c] for c in s])

_str = "hamza is at the library"
# print({ch: i for i, ch in enumerate(_str)})
# print({i: ch for i, ch in enumerate(_str)})
