import torch

def inspect_checkpoint(path="checkpoints/best_cartpole_dqn.pt"):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # It’s a plain state_dict
        print(f"Checkpoint '{path}' contains {len(ckpt)} tensors:")
        for k, v in ckpt.items():
            print(f"{k:20s} {tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(ckpt, dict):
        # More complex checkpoint (with optimizer, step, etc.)
        print(f"Checkpoint '{path}' has keys: {list(ckpt.keys())}")
        for k, v in ckpt.items():
            if isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} tensors")
            else:
                print(f"  {k}: {type(v)}")
    else:
        print(f"Unexpected checkpoint type: {type(ckpt)}")


if __name__ == "__main__":
    inspect_checkpoint()
