from safetensors.torch import load_file

# Load everything into a dict of torch.Tensors
state_dict = load_file(
    "/home/fundwotsai/MuseControlLite/checkpoint-13000/model.safetensors",
    device="cpu",       # or "cuda:0" if you want GPU
)

# Print summary: name, shape, dtype
for name, tensor in state_dict.items():
    print(f"{name:40}  shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
