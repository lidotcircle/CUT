from typing import Iterable, List

import torch
from torchvision.transforms import functional as TF
from tqdm import tqdm
from .fid_score import ActivationConvertor, _compute_activations, calculate_activation_statistics, get_activations_from_tensor
from .fid_score import calculate_frechet_distance, calculate_frechet_distance_torch
from .kid_score import polynomial_mmd_averages
from .models.inception import InceptionV3
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import os


def calculate_scores_given_iter(
    iter: Iterable[List[torch.Tensor]],
    device, dims, use_fid_inception=False, torch_svd=False
    ):
    convertor = ActivationConvertor(dims, device, use_fid_inception=use_fid_inception)
    features_array = []
    feat1 = None
    feat2 = None
    for images in tqdm(iter):
        for i in range(len(images)):
            if images[i] is None:
                continue
            img = TF.normalize(images[i], (-1, -1, -1), (2, 2, 2))
            feat = convertor(img)
            if len(features_array) > i:
                features_array[i] = np.concatenate([features_array[i], feat], axis=0)
            else:
                features_array.append(feat)
    
    ans = []
    k1 = features_array[0::2]
    k2 = features_array[1::2]
    assert len(k1) == len(k2)
    for feat1, feat2 in zip(k1, k2):
        # FID
        m1, s1 = calculate_activation_statistics(feat1)
        m2, s2 = calculate_activation_statistics(feat2)
        if torch_svd:
            fid_value = calculate_frechet_distance_torch(m1, s1, m2, s2, device=device)
        else:
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        
        # KID
        kid_values = polynomial_mmd_averages(feat1, feat2, n_subsets=100)
        ans.append((fid_value, kid_values[0].mean(), kid_values[0].std()))
    
    return ans


def calculate_scores_given_paths(paths, batch_size, device, dims, use_fid_inception=False, torch_svd=False):
    """Calculates the KID of two paths"""
    pths = []
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
        if os.path.isdir(p):
            pths.append(p)
        elif p.endswith('.npy'):
            np_imgs = np.load(p)
            if np_imgs.shape[0] > 50000: np_imgs = np_imgs[np.random.permutation(np.arange(np_imgs.shape[0]))][:50000]
            pths.append(np_imgs)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_fid_inception=use_fid_inception).to(device)

    act_true = _compute_activations(pths[0], model, batch_size, dims, device=device)
    pths = pths[1:]
    results = [ [] for i in enumerate(pths) ]
    for j, pth in enumerate(pths):
        store = results[j]
        store.append(paths[j+1])
        print(paths[j+1])

        # KID
        actj = _compute_activations(pth, model, batch_size, dims, device=device)
        kid_values = polynomial_mmd_averages(act_true, actj, n_subsets=100)
        store.append((kid_values[0].mean(), kid_values[0].std()))

        # FID
        m1, s1 = calculate_activation_statistics(act_true)
        m2, s2 = calculate_activation_statistics(actj)
        if torch_svd:
            fid_value = calculate_frechet_distance_torch(m1, s1, m2, s2, device=device)
        else:
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        store.append(fid_value)

    return results


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--true', type=str, required=True,
                        help=('Path to the true images'))
    parser.add_argument('--fake', type=str, nargs='+', required=True,
                        help=('Path to the generated images'))
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--device', default='cpu', type=str,
                        help='gpu (cuda) or cpu')
    parser.add_argument('--torch_svd', action='store_true', required=False, 
                        help=('compute frechet distance with pytorch (quicker, but final result is different with original method'))
    parser.add_argument('--use_fid_inception', action='store_true', required=False, 
                        help=('uses the pretrained Inception model used in Tensorflow\'s FID implementation'))
    args = parser.parse_args()
    print(args)
    paths = [args.true] + args.fake

    results = calculate_scores_given_paths(paths, args.batch_size, device=args.device, dims=args.dims,
                                           torch_svd=args.torch_svd,
                                           use_fid_inception=args.use_fid_inception)
    for p, m, s in results:
        print('[KID: %.5f (%.5f), FID: %.2f](%s)' % (m[0], m[1], s, p))
