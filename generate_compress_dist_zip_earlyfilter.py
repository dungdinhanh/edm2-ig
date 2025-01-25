"""Script to generate image zip for reproducing quantitative ImageNet-512 results in Table 1."""

import ast
from io import BytesIO
import numpy as np
import os
import PIL.Image
import torch
import time
from typing import Optional
import zipfile

from sampling.sample_images import denoise_latents, \
                                   decode_latents, denoise_latents_compress
from utils import load_networks, count_images_in_zip
import torch.distributed as dist
from torch_utils import dist_util
from torch_utils.file_util import *
import argparse


# @click.command()
# @click.option('--cond_pkl', required=True, type=str, help='Path to conditional network pickle.')
# @click.option('--uncond_pkl', required=True, type=str, help='Path to unconditional network pickle.')
# @click.option('--zip_dir', default='./results', type=str, help='Path to the directory where generated images are saved.')
# @click.option('--num_images', default=50000, type=int, help='Number of images generated to the zip.')
# @click.option('--batch_size', default=64, type=int, help='Batch size for the denoiser network.')
# @click.option('--batch_size_decoder', default=8, type=int, help='Batch size for the decoder network.')
# @click.option('--guidance_scale', default=2.0, type=float, help='Guidance scale. For FID optimal results, use G = 2.0 for guidance interval, G = 1.2 for CFG.')
# @click.option('--guidance_interval', default='[17, 22]', type=str, help='List specifying guidance interval (start, stop) indices. None = CFG, FID: EDM2-XXL = [17, 22], FD_DINOv2: EDM2-XXL = [13, 19].')
# @click.option('--verbose', default=False, type=bool, help='Flag for extra prints.')
# @click.option('--seed', default=0, type=int, help='Random seed for generating images to zip.')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images with conditional and unconditional networks.")

    parser.add_argument('--cond_pkl', required=True, type=str, help='Path to conditional network pickle.')
    parser.add_argument('--uncond_pkl', required=True, type=str, help='Path to unconditional network pickle.')
    parser.add_argument('--zip_dir', default='./results', type=str, help='Path to the directory where generated images are saved.')
    parser.add_argument('--num_images', default=50000, type=int, help='Number of images generated to the zip.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for the denoiser network.')
    parser.add_argument('--batch_size_decoder', default=8, type=int, help='Batch size for the decoder network.')
    parser.add_argument('--guidance_scale', default=2.0, type=float, help='Guidance scale. For FID optimal results, use G = 2.0 for guidance interval, G = 1.2 for CFG.')
    parser.add_argument('--guidance_interval', default='[17, 22]', type=str, help='List specifying guidance interval (start, stop) indices. None = CFG, FID: EDM2-XXL = [17, 22], FD_DINOv2: EDM2-XXL = [13, 19].')
    parser.add_argument('--verbose', default=False, action='store_true', help='Flag for extra prints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for generating images to zip.')
    parser.add_argument("--base_folder", type=str, default="./")
    parser.add_argument("--fix_seed", action="store_true")
    parser.add_argument("--raw_images", action="store_true")
    parser.add_argument("--save-num", default=20000, type=int, help="number of images to save checkpoints")
    parser.add_argument("--skip", default=4, type=int, help="compress rate")
    parser.add_argument("--k", default=4.0, type=float, help="compress k")
    parser.add_argument("--efilter", default=-1, type=int, help="Default early filter is no filter")
    return parser.parse_args()

def main(local_rank) -> None:
    """Reproduces quantitative ImageNet-512 results.

    """
    args = parse_arguments()

    # Create variables with the same names as arguments
    cond_pkl = args.cond_pkl
    uncond_pkl = args.uncond_pkl
    zip_dir = args.zip_dir
    num_images = args.num_images
    batch_size = args.batch_size
    batch_size_decoder = args.batch_size_decoder
    guidance_scale = args.guidance_scale
    guidance_interval = args.guidance_interval
    verbose = args.verbose
    seed = args.seed

    # Example usage of the variables
    if verbose:
        print("Arguments:")
        print(f"cond_pkl: {cond_pkl}")
        print(f"uncond_pkl: {uncond_pkl}")
        print(f"zip_dir: {zip_dir}")
        print(f"num_images: {num_images}")
        print(f"batch_size: {batch_size}")
        print(f"batch_size_decoder: {batch_size_decoder}")
        print(f"guidance_scale: {guidance_scale}")
        print(f"guidance_interval: {guidance_interval}")
        print(f"verbose: {verbose}")
        print(f"seed: {seed}")

    
    guidance_interval = ast.literal_eval(guidance_interval)
    method = 'gi' if guidance_interval is not None else 'cfg'
    # zip_name = f"{os.path.basename(cond_pkl).split('.pkl')[0]}-s{seed}-nimg{num_images}-g{guidance_scale:0.2f}-{method}.zip"
    # zip_path = os.path.join(zip_dir, zip_name)
    # os.makedirs(zip_dir, exist_ok=True)

    # Setup DDP:
    # dist.init_process_group("nccl")
    dist_util.setup_dist(local_rank)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()


    # Create folder for sampling
    base_folder = args.base_folder
    output_folder_path = os.path.join(base_folder, args.zip_dir)
    sample_folder_dir = os.path.join(output_folder_path, f"images")
    reference_dir = os.path.join(output_folder_path, "reference")
    os.makedirs(reference_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    list_png_files, max_index = get_png_files(sample_folder_dir)

    final_file = os.path.join(reference_dir,
                              f"samples_{num_images}x512x512x3.zip")
    if os.path.isfile(final_file):
        dist.barrier()
        print("Sampling complete")
        dist.barrier()
        dist.destroy_process_group()
        return

    checkpoint = os.path.join(sample_folder_dir, "last_samples.zip")

    # load all single image into checkpoint by rank 0 then reload to all processes
    # if rank == 0:
    #     if os.path.isfile(checkpoint):
    #         all_images = list(np.load(checkpoint)['arr_0'])
    #     else:
    #         all_images = []
    #     all_images = compress_images_to_npz(sample_folder_dir, all_images)

    # if os.path.isfile(checkpoint):
    #     all_images = list(np.load(checkpoint)['arr_0'])
    # else:
    #     all_images = []
    # count number of images in last_samples.zip
    if rank == 0:
        if os.path.isfile:
            with zipfile.ZipFile(checkpoint, 'a') as f:
                compress_images_to_zip(sample_folder_dir, f) #todo
        else:
            f =  zipfile.ZipFile(checkpoint, "w")
            f.close()
                
            pass
    dist.barrier()
    no_png_files, max_index = count_images_in_zip_with_max_idx(checkpoint)

    # no_png_files = len(all_images)
    if no_png_files >= num_images:
        if rank == 0:
            print(f"Complete sampling {no_png_files} satisfying >= {num_images}")

            all_images = read_from_zip(sample_folder_dir)
            # create_npz_from_sample_folder(os.path.join(base_folder, args.sample_dir), args.num_fid_samples, args.image_size)
            arr = np.stack(all_images)
            arr = arr[: num_images]
            shape_str = "x".join([str(x) for x in arr.shape])
            reference_dir = os.path.join(output_folder_path, "reference")
            os.makedirs(reference_dir, exist_ok=True)
            out_path = os.path.join(reference_dir, f"samples_{shape_str}.npz")
            # logger.log(f"saving to {out_path}")
            print(f"Saving to {out_path}")
            np.savez(out_path, arr)
            os.remove(checkpoint)
            print("Done")
        dist.barrier()
        dist.destroy_process_group()
        return
    else:
        if rank == 0:
            # remove_prev_npz(args.sample_dir, args.num_fid_samples, args.image_size)
            print("continue sampling")

    total_samples = int(math.ceil((num_images - no_png_files) / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Already sampled {no_png_files}")
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = no_png_files

    if args.fix_seed:
        import random
        # seed = args.seed * dist.get_world_size() + rank
        seed = seed + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        random.seed(seed)
        np.random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)
        seeds = np.arange(seed, seed + dist.get_world_size() * samples_needed_this_gpu, dist.get_world_size(), dtype=int)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        print(f"Starting rank={rank}, random seed, world_size={dist.get_world_size()}.")
        seeds = np.random.randint(0, total, (samples_needed_this_gpu,))

    

    # device = torch.device('cuda')
    # seeds = np.arange(seed, seed + num_images, dtype=int)
    # torch.manual_seed(seed)

    # Define sampler parameters.
    guidance_steps, compression_rate = get_guidance_timesteps_with_weight(n=32, skip=args.skip, k=args.k, filter_early=args.efilter)
    # print(guidance_steps)
    # print(compression_rate)
    # exit(0)
    sampler_kwargs = dict(num_steps=32,
                          sigma_min=0.002,
                          sigma_max=80.0,
                          rho=7,
                          S_churn=0.0,
                          S_min=0.0,
                          S_max=float('inf'),
                          S_noise=1.0)

    # Load networks.
    cond_pkl_path = os.path.join(base_folder, cond_pkl)
    uncond_pkl_path = os.path.join(base_folder, uncond_pkl)
    hub_path = os.path.join(base_folder, "hub")
    cond_net, uncond_net, vae = load_networks(cond_pkl=cond_pkl_path,
                                              uncond_pkl=uncond_pkl_path,
                                              device=device, cache_dir=hub_path)

    # Generate images to zip.
    if rank == 0:
        print(f'Generating {total_samples} images and saving them to "{reference_dir}..."')
    current_samples = 0
    with zipfile.ZipFile(checkpoint, 'a') as f:
        for begin in range(0, samples_needed_this_gpu, batch_size):
            end = min(begin + batch_size, samples_needed_this_gpu)
            total_start = time.time()
            with torch.no_grad():
                den_start = time.time()
                denoised_latents = denoise_latents_compress(cond_net=cond_net,
                                                    uncond_net=uncond_net,
                                                    seeds=seeds[begin:end],
                                                    G=guidance_scale,
                                                    batch_size=batch_size,
                                                    sampler_kwargs=sampler_kwargs,
                                                    guidance_list=guidance_steps,
                                                    compress_rate=compression_rate,
                                                    device=device)
                den_time = time.time() - den_start

                dec_start = time.time()
                images = decode_latents(denoised_latents=denoised_latents,
                                        vae=vae,
                                        batch_size=batch_size_decoder,
                                        device=device)
                dec_time = time.time() - dec_start

            saving_start = time.time()
            for i, image in enumerate(images):
                # zip_fname = f'{img_idx:06d}.png'
                im = PIL.Image.fromarray(image.transpose(1, 2, 0))
                index = i * dist.get_world_size() + rank + max_index
                image_file = f"{sample_folder_dir}/{index:06d}.png"
                im.save(image_file, 'PNG')
                # f.writestr(zip_fname, image_file.getvalue())
            total += global_batch_size
            max_index += global_batch_size
            current_samples += global_batch_size
            saving_time = time.time() - saving_start
            dist.barrier()
            if verbose:
                elapsed_time = time.time() - total_start
                imgs_per_second = (end - begin) / elapsed_time
                print(f'{total}/{num_images} images generated ({imgs_per_second:0.2f} imgs/s)')
                print(f'Batch timing: denoising = {den_time:0.1f}s, decoding = {dec_time:0.1f}s, saving = {saving_time:0.1f}s\n')

            dist.barrier()
            if current_samples >= args.save_num or total >= total_samples:
                if rank == 0:
                    if not args.raw_images:
                        compress_images_to_zip(sample_folder_dir, f=f)
                        # all_images = compress_images_to_npz(sample_folder_dir, all_images)
                        current_samples = 0
                pass
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        if not args.raw_images:
            # create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples)
            print(f"Complete sampling {total} satisfying >= {num_images}")

            all_images = read_from_zip(sample_folder_dir)
            # create_npz_from_sample_folder(os.path.join(base_folder, args.sample_dir), args.num_fid_samples, args.image_size)
            arr = np.stack(all_images)
            arr = arr[: num_images]
            shape_str = "x".join([str(x) for x in arr.shape])
            reference_dir = os.path.join(output_folder_path, "reference")
            os.makedirs(reference_dir, exist_ok=True)
            out_path = os.path.join(reference_dir, f"samples_{shape_str}.npz")
            # logger.log(f"saving to {out_path}")
            print(f"Saving to {out_path}")
            np.savez(out_path, arr)
            os.remove(checkpoint)
        print("Done.")
        # print("Done.")
    dist.barrier()
    dist.destroy_process_group()
        

    # All good.
    if rank == 0:
        print('Done.')


def get_guidance_timesteps_with_weight(n=250, skip=5, k=2.0, filter_early=-1):
    # c * i^2
    if filter_early > 0:
        remove_guidance = filter_early
    else:
        remove_guidance = 0
    T = n - 1 - remove_guidance
    total_steps = n - remove_guidance
    max_steps = int(n/skip)
    c = total_steps/(max_steps**k)
    guidance_timesteps = np.zeros((n,), dtype=int)
    for i in range(max_steps):
        guidance_index = - int(c * (i ** k)) + T
        if 0 <= guidance_index and guidance_index <= T:
            guidance_timesteps[guidance_index] = 1
        else:
            print(f"guidance index: {guidance_index}")
            print(f"constant c: {c}")
            print(f"faulty index: {i}")
            print(f"timesteps {T}")
            print(f"compressd by {skip} times")
            print(f"error in index must larger than 0 or less than {T}")
            exit(0)
    # guidance_timesteps = guidance_timesteps[::-1]
    compression_up = np.zeros_like(guidance_timesteps)
    # cur_index = 0
    accumulate = 0
    for i in range(n):
        if guidance_timesteps[i] != 0:
            compression_up[i] = guidance_timesteps[i] + accumulate
            accumulate = 0
        else:
            accumulate += 1
        pass
    guidance_timesteps = guidance_timesteps[::-1]
    compression_up = compression_up[::-1]
    # print(guidance_timesteps)
    # print(np.sum(guidance_timesteps))
    # print("__________________________________")
    # print(compression_up)
    # exit(0)
    return guidance_timesteps, compression_up



if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
