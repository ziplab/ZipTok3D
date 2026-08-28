import torch
import numpy as np
import mcubes
import trimesh
from scipy.spatial import cKDTree as KDTree

from cod.models.vae.base import BaseAutoencoder


def chunked_reconstruct(model: BaseAutoencoder, pc: torch.Tensor, query_points: torch.Tensor, chunk_size: int,
                        threshold: float = 0., active_num_latents=None,
                        num_decode_loops=None, sample_posterior=True):
    logits = []
    z = model.encode(
        pc,
        active_num_latents=active_num_latents,
        sample_posterior=sample_posterior,
    )[0]
    recon = model.decode(
        z,
        active_num_latents=active_num_latents,
        num_decode_loops=num_decode_loops,
    )[0]
    for i in range(0, query_points.size(1), chunk_size):
        logit = model.decode_queries(recon, query_points[:, i:i + chunk_size])
        logits.append(logit)

    logits = torch.cat(logits, dim=1)
    preds = (logits > threshold).float()

    return logits, preds


def occupancy_to_mesh(preds, r, threshold=0):
    preds = preds.detach()
    gap = 2. / (r - 1)
    volume = preds.view(r, r, r).permute(1, 0, 2).cpu().numpy()
    verts, faces = mcubes.marching_cubes(volume, threshold)
    verts *= gap
    verts -= 1

    return trimesh.Trimesh(verts, faces)


def create_grid_queries(density):
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    z = np.linspace(-1, 1, density)

    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
    grid = grid.view(3, -1).transpose(0, 1)[None]
    return grid

def compute_cd_of_mesh(mesh, surface, threshold=0.02):
    if mesh is None:
        ## assume all points are at the origin when mesh is missing
        return np.mean(surface) * 2, 0

    pred = mesh.sample(100000)

    tree = KDTree(pred)
    dist, _ = tree.query(surface)
    d1 = dist
    gt_to_gen_chamfer = np.mean(dist)
    # gt_to_gen_chamfer = np.mean(np.square(dist))

    tree = KDTree(surface)
    dist, _ = tree.query(pred)
    d2 = dist
    gen_to_gt_chamfer = np.mean(dist)
    # gen_to_gt_chamfer = np.mean(np.square(gen_to_gt_chamfer))
    cd = gt_to_gen_chamfer + gen_to_gt_chamfer
    th = threshold
    recall = (d2 < th).mean()
    precision = (d1 < th).mean()

    if recall + precision > 0:
        fscore = (2 * recall * precision) / (recall + precision)
    else:
        fscore = 0

    return cd, fscore
