"""Calculate FID/FD_DINOv2 for guidance interval/CFG results in Table 1."""
import argparse
import numpy as np
import os
import torch

from dataset import get_dataloader, get_dataloader_npz
from metrics.embeddings import load_feature_network, load_feature_network_cache
from metrics.frechet_inception_distance import compute_fid
from utils import count_images_in_zip, \
                  load_reference_statistics, \
                  write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Computes evaluation metrics.")
    parser.add_argument('--gen_zip', required=True, type=str,
                        help='Path to zip containing generated images.')
    parser.add_argument('--ref_path', required=True, type=str,
                        help='Path to file containing reference features.')
    parser.add_argument('--results_dir', default='./results', type=str,
                        help='Directory where the results are saved.')
    parser.add_argument('--metric', default='fid', type=str, choices=['fid', 'fd_dinov2'],
                        help='Metric name. One of: fid, fd_dinov2.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for computing features.')
    parser.add_argument('--base_folder', default="./", type=str, help="Default folder")

    args = parser.parse_args()

    """
    python calculate_metrics.py --gen_zip <path-to-gen-zip> --ref_path <path-to-img512.pkl> --metric fid
    """

    gen_zip = args.gen_zip
    ref_path = args.ref_path
    results_dir = args.results_dir
    metric = args.metric
    batch_size = args.batch_size
    base_folder = args.base_folder

    # Ensure valid metric
    assert metric in ['fid', 'fd_dinov2'], f'Invalid metric name: {metric}.'
    os.makedirs(results_dir, exist_ok=True)
    feature_net_name = 'inception_v3' if metric == 'fid' else 'dinov2'
    device = torch.device('cuda')

    # Load reference features.
    print('Loading reference features...')
    mu_ref, sigma_ref = load_reference_statistics(ref_path=ref_path, metric=metric)

    # Load feature network.
    hub_folder = os.path.join(base_folder, "hub")
    print('Loading feature network...')
    feature_net = load_feature_network_cache(name=feature_net_name, hub_folder=hub_folder).to(device)

    # Get dataloader for generated images.
    print('Initializing dataloader...')
    # num_images = count_images_in_zip(gen_zip)
    dataloader = get_dataloader_npz(npz_path=gen_zip,
                                resolution=512,
                                batch_size=batch_size,
                                num_images=50000,
                                num_workers=0)

    # Compute features of generated images.
    print('Computing generated features...')
    gen_features = []
    for gen_images, _ in dataloader:
        with torch.no_grad():
            gen_features.append(feature_net(gen_images.to(device)).cpu().numpy())
    gen_features = np.concatenate(gen_features, axis=0)

    # Compute FID.
    metric_name = 'FID' if metric == 'fid' else 'FD_DINOv2'
    print(f'Computing {metric_name}...')
    fd = compute_fid(gen_features=gen_features,
                     mu_ref=mu_ref,
                     sigma_ref=sigma_ref)
    print(f'{metric_name} = {fd:0.2f}')
    write_jsonl(path=os.path.join(results_dir, f'{metric}.jsonl'),
                data={'gen_zip': gen_zip.split('.zip')[0],
                      metric: fd,
                      'num_gen_images': 50000})
    
    # All good.
    print('Done.')


if __name__ == "__main__":
    main()
