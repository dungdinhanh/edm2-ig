import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import shutil
import math
import random
from pathlib import Path

def get_png_files(sample_dir):
    list_files = []
    max = 0
    for file in os.listdir(sample_dir):
        if file.endswith(".png"):
            list_files.append(file)
            num_image = int(file.split(".")[0])
            if num_image > max:
                max = num_image
    return list_files, max+1


def create_npz_from_sample_folder(sample_dir, num=50_000, image_size=256, raw_images=False):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    reference_dir = os.path.join(sample_dir, "reference")
    npz_path = os.path.join(reference_dir, f"samples_{num}x{image_size}x{image_size}x3.npz")
    if os.path.isfile(npz_path):
        print(f"Completed sampling _ file found at {npz_path}")
        return npz_path

    images_dir = os.path.join(sample_dir, "images")
    list_png_files, _ = get_png_files(images_dir)
    no_png_files = len(list_png_files)
    os.makedirs(reference_dir, exist_ok=True)
    assert no_png_files >= num, print("not enough images, generate more")
    print("Building .npz file from samples")
    for i in range(num):
        image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
        try:
            # image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
            img = Image.open(image_png_path)
            img.verify()
        except(IOError, SyntaxError) as e:
            print(f'Bad file {image_png_path}')
            print(f'remove {image_png_path}')
            os.remove(image_png_path)
            continue
        sample_pil = Image.open(os.path.join(images_dir, f"{list_png_files[i]}"))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape[1] == image_size, "what the heck?"
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = os.path.join(reference_dir, f"samples_{num}x{samples.shape[1]}x{samples.shape[2]}x3.npz")

    np.savez(npz_path, arr_0=samples)
    if not raw_images:
        shutil.rmtree(images_dir)
        print("Removed all raw images")
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")


    return npz_path

def remove_prev_npz(sample_dir, num=50_000, image_size=256):
    reference_dir = os.path.join(sample_dir, "reference")
    npz_path = os.path.join(reference_dir, f"samples_{num}x{image_size}x{image_size}x3.npz")
    if os.path.isfile(npz_path):
        print(f"Removing {npz_path}")
        os.remove(npz_path)

def get_output_name(sample_dir, num=50000, image_size=256):
    reference_dir = os.path.join(sample_dir, "reference")
    npz_path = os.path.join(reference_dir, f"samples_{num}x{image_size}x{image_size}x3.npz")
    return npz_path

def compress_images_to_npz(sample_folder_dir, all_images=[], remove=True):
    npz_file = os.path.join(sample_folder_dir, "last_samples.npz")
    list_png_files, _ = get_png_files(sample_folder_dir)
    no_png_files = len(list_png_files)
    if no_png_files <= 1:
        return all_images
    for i in range(no_png_files):
        image_png_path = os.path.join(sample_folder_dir, f"{list_png_files[i]}")
        try:
            # image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
            img = Image.open(image_png_path)
            img.verify()
        except(IOError, SyntaxError) as e:
            print(f'Bad file {image_png_path}')
            print(f'remove {image_png_path}')
            os.remove(image_png_path)
            continue
        sample_pil = Image.open(os.path.join(image_png_path))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        all_images.append(sample_np)
        os.remove(image_png_path)
    np_all_images = np.stack(all_images)
    np.savez(npz_file, arr_0=np_all_images)
    return all_images

def compress_images_prompts_to_npz(sample_folder_dir, all_images=[], all_prompts=[]):
    image_folder = os.path.join(sample_folder_dir, "images")
    prompts_folder = os.path.join(sample_folder_dir, "prompts")

    npz_file = os.path.join(sample_folder_dir, "last_samples.npz")
    list_png_files, _ = get_png_files(image_folder)
    no_png_files = len(list_png_files)
    if no_png_files <= 1:
        return all_images, all_prompts
    for i in range(no_png_files):
        image_png_path = os.path.join(image_folder, f"{list_png_files[i]}")
        image_id = Path(image_png_path).stem
        prompt_txt_path = os.path.join(prompts_folder, f"{image_id}.txt")
        try:
            # image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
            img = Image.open(image_png_path)
            img.verify()
        except(IOError, SyntaxError) as e:
            print(f'Bad file {image_png_path}')
            print(f'remove {image_png_path}')
            os.remove(image_png_path)
            print(f"also remove {prompt_txt_path}")
            os.remove(prompt_txt_path)
            continue

        try:
            f = open(prompt_txt_path, "r")
        except(IOError, SyntaxError) as e:
            print(f"Bad file {prompt_txt_path}")
            print(f"remove {prompt_txt_path}")
            os.remove(prompt_txt_path)
            print(f"also remove {image_png_path}")
            os.remove(image_png_path)
            continue
        sample_pil = Image.open(os.path.join(image_png_path))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        all_images.append(sample_np)
        os.remove(image_png_path)

        prompt = f.readlines()[0].strip()
        all_prompts.append(prompt)
        os.remove(prompt_txt_path)
    np_all_images = np.stack(all_images)
    np_all_prompts = np.asarray(all_prompts)
    np.savez(npz_file, np_all_images, np_all_prompts)
    return all_images, all_prompts

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]

    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images