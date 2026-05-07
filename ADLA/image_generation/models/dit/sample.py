
import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from diffusion import create_diffusion
from download import find_model
from models import DiT_models


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    sample_files = sorted(
        os.path.join(sample_dir, name)
        for name in os.listdir(sample_dir)
        if name.endswith(".png")
    )
    assert len(sample_files) >= num, f"Found {len(sample_files)} samples, but need at least {num} samples."

    samples = []
    for sample_path in tqdm(sample_files[:num], desc="Building .npz file from samples"):
        sample_pil = Image.open(sample_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def get_vae_local_dir(args):
    if args.vae_path:
        return args.vae_path

    repo_id = f"stabilityai/sd-vae-ft-{args.vae}"
    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "local_vae", repo_id.replace("/", "--"))

    cache_dir = args.hf_cache_dir or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return os.path.join(os.path.dirname(cache_dir), "local_vae", repo_id.replace("/", "--"))

    return None


def is_vae_dir_ready(path):
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    weight_names = [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
    ]
    return any(os.path.isfile(os.path.join(path, name)) for name in weight_names)


def load_vae(args, device):
    repo_id = f"stabilityai/sd-vae-ft-{args.vae}"
    local_dir = get_vae_local_dir(args)

    if is_vae_dir_ready(local_dir):
        print(f"Loading VAE from local directory: {local_dir}")
        return AutoencoderKL.from_pretrained(local_dir, local_files_only=True).to(device)

    cache_dir = args.hf_cache_dir or os.environ.get("HUGGINGFACE_HUB_CACHE")
    raise FileNotFoundError(
        f"Local VAE cache not found for {repo_id}. "
        f"Expected a ready directory at {local_dir!r}. "
        f"Please pre-download the VAE or pass --vae_path explicitly. "
        f"Current cache_dir={cache_dir!r}."
    )


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = load_vae(args, device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    if rank == 0:
        folder_name_txt_path = os.path.join(args.sample_dir, "folder_name.txt")
        with open(folder_name_txt_path, "w") as f:
            f.write(folder_name)
        print(f"Saved folder_name to {folder_name_txt_path}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--per_proc_batch_size", type=int, default=32)
    parser.add_argument("--num_fid_samples", type=int, default=50_000)
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--vae_path", type=str, default=None, help="Optional local path to a pre-downloaded VAE.")
    parser.add_argument("--hf_home", type=str, default=None, help="Optional Hugging Face home directory.")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Optional Hugging Face hub cache directory.")
    args = parser.parse_args()
    main(args)
