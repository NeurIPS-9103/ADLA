
"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import wandb
import logging
import argparse
import numpy as np
from PIL import Image
from glob import glob
from time import time, sleep
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from huggingface_hub import snapshot_download

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
 
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def estimated_time(t_start, cur_iter, start_iter, total_iter):
    t_curr = time()
    eta_total = (t_curr - t_start) / (cur_iter + 1 - start_iter) * (total_iter - cur_iter - 1)
    eta_hour = int(eta_total // 3600)
    eta_min = int((eta_total - eta_hour * 3600) // 60)
    eta_sec = int(eta_total - eta_hour * 3600 - eta_min * 60)
    return f'{eta_hour:02d} h {eta_min:02d} m {eta_sec:02d} s', (eta_total / 3600.)


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory.
    Returns the path to the latest checkpoint, or None if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all epoch-based checkpoints
    epoch_ckpts = glob(f"{checkpoint_dir}/epoch_*.pt")
    if epoch_ckpts:
        # Sort by modification time and return the latest
        latest_ckpt = max(epoch_ckpts, key=os.path.getmtime)
        return latest_ckpt
    
    return None


def configure_hf_cache(args):
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.hf_home, "hub"))
    if args.hf_cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def get_vae_local_dir(args, repo_id):
    if args.vae_local_dir:
        return args.vae_local_dir
    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "local_vae", repo_id.replace("/", "--"))
    if args.hf_cache_dir:
        return os.path.join(os.path.dirname(args.hf_cache_dir), "local_vae", repo_id.replace("/", "--"))
    return os.path.join(os.path.expanduser("~/.cache/huggingface"), "local_vae", repo_id.replace("/", "--"))


def is_vae_dir_ready(path):
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    weight_names = [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
    ]
    return any(os.path.isfile(os.path.join(path, name)) for name in weight_names)


def get_hf_endpoints(args):
    endpoints = []
    for endpoint in [args.hf_endpoint, os.environ.get("HF_ENDPOINT"), None]:
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    return endpoints


def resolve_vae_source(args, logger, rank):
    if args.vae_path:
        if not os.path.isdir(args.vae_path):
            raise FileNotFoundError(f"VAE path does not exist or is not a directory: {args.vae_path}")
        return args.vae_path

    repo_id = f"stabilityai/sd-vae-ft-{args.vae}"
    cache_dir = args.hf_cache_dir or os.environ.get("HUGGINGFACE_HUB_CACHE")
    local_dir = get_vae_local_dir(args, repo_id)
    parent_dir = os.path.dirname(local_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    status = {"ok": True, "error": ""}

    if is_vae_dir_ready(local_dir):
        logger.info(f"Using cached VAE from {local_dir}")
    elif rank == 0:
        status = {"ok": False, "error": f"Failed to prepare VAE from {repo_id}"}
        for endpoint in get_hf_endpoints(args):
            endpoint_name = endpoint
            for attempt in range(1, args.hf_download_retries + 1):
                try:
                    logger.info(
                        f"Preparing VAE from {repo_id} into {local_dir} via {endpoint_name} "
                        f"(attempt {attempt}/{args.hf_download_retries})"
                    )
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_dir=local_dir,
                        endpoint=endpoint,
                    )
                    if not is_vae_dir_ready(local_dir):
                        raise RuntimeError(f"Incomplete VAE download in {local_dir}")
                    status = {"ok": True, "error": ""}
                    logger.info(f"VAE is ready in {local_dir}")
                    break
                except Exception as exc:
                    status = {"ok": False, "error": f"{endpoint_name}: {exc}"}
                    logger.warning(
                        f"VAE download failed via {endpoint_name} on attempt "
                        f"{attempt}/{args.hf_download_retries}: {exc}"
                    )
                    if attempt < args.hf_download_retries:
                        sleep(args.hf_download_retry_wait)
            if status["ok"]:
                break

    if dist.is_initialized():
        payload = [status]
        dist.broadcast_object_list(payload, src=0)
        status = payload[0]
        dist.barrier()

    if not status["ok"]:
        raise RuntimeError(status["error"])
    if not is_vae_dir_ready(local_dir):
        raise FileNotFoundError(f"VAE directory is still incomplete: {local_dir}")
    return local_dir


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    configure_hf_cache(args)

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # wandb
    if args.use_wandb and dist.get_rank() == 0:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        else:
            wandb.login()
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            dir=args.results_dir,
            reinit=True,
        )

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        logger = create_logger(experiment_dir)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae_source = resolve_vae_source(args, logger, rank)
    logger.info(f"Loading VAE from {vae_source}")
    vae = AutoencoderKL.from_pretrained(vae_source, local_files_only=True).to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if args.dummy:
        logger.info(f"[WARNING] You are using dummy data, which is only used for debugging!")
        dataset = datasets.FakeData(1281167, (3, args.image_size, args.image_size), 1000, transforms.ToTensor())
    else:
        dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    resume_start_step = 0
    start_time = time()
    global_start_time = time()

    # auto resume is default True
    if True:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            args.resume_ckpt = latest_ckpt
            logger.info(f"Auto resume enabled: found checkpoint {latest_ckpt}")
        else:
            logger.info("Auto resume enabled: no checkpoint found, starting from scratch")

    # Resume
    if (args.resume_ckpt is not None) and (os.path.exists(args.resume_ckpt)):
        logger.info(f'Start resume from {args.resume_ckpt}')
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        args.start_epoch = ckpt['epoch']
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        train_steps = ckpt['train_steps']
        resume_start_step = train_steps
        logger.info(f'Finish resume, from finished epoch {args.start_epoch}.')

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)  # loss_dict['vb'].shape = loss_dict['vb'].shape = loss_dict['vb'].shape = [bs]
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/{(args.epochs * len(loader)):07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Elapsed Time: {((time() - global_start_time) / 3600.):.2f} h, ETA: {estimated_time(global_start_time, train_steps, resume_start_step, args.epochs * len(loader))[0]}")
                # wandb
                if args.use_wandb and dist.get_rank() == 0:
                    wandb.log(
                        {"train_loss": avg_loss, "train_steps_per_sec": steps_per_sec, "used_hours": (time() - global_start_time) / 3600.,
                        "remain_hours": estimated_time(global_start_time, train_steps, resume_start_step, args.epochs * len(loader))[1]},
                        step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # Save DiT checkpoint, epoch-based, can be resume:
        if (epoch + 1) % args.epoch_ckpt_every == 0:
            if rank == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "train_steps": train_steps,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch:03d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae_path", type=str, default=None, help="optional local path to a pre-downloaded VAE")
    parser.add_argument("--vae_local_dir", type=str, default=None, help="optional local dir used for automatic VAE downloads")
    parser.add_argument("--hf_home", type=str, default=None, help="optional Hugging Face home directory")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="optional Hugging Face hub cache directory")
    parser.add_argument("--hf_endpoint", type=str, default=None, help="optional Hugging Face endpoint")
    parser.add_argument("--hf_download_retries", type=int, default=3)
    parser.add_argument("--hf_download_retry_wait", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--epoch_ckpt_every", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, help="ckpt to resume")
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    ## wandb
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_project", type=str, default="dit-training")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    main(args)
