import numpy as np


def two_stage_sampling(points_list, num_samples, chunk_size: int = 2000, oversample_ratio: int = 10):
    """
    The two-stage sampling technique to load query points in an IO-efficient manner.
    :param points_list:
    :param num_samples:
    :return:
    """
    chunk_size = chunk_size
    num_total_blocks = points_list[0].shape[0] // chunk_size
    num_blocks = ((num_samples * oversample_ratio) // chunk_size) + 1
    block_indices = np.random.choice(num_total_blocks, num_blocks, replace=False)
    block_indices = np.sort(block_indices)
    sampled_points_list = []

    point_indices = None
    for points in points_list:
        blocks = []
        for block_idx in block_indices:
            start = block_idx * chunk_size
            end = min((block_idx + 1) * chunk_size, points_list[0].shape[0])
            blocks.append(points[start:end])

        points = np.concatenate(blocks, axis=0)
        if point_indices is None:
            point_indices = np.random.choice(points.shape[0], num_samples, replace=False)
        sampled_points_list.append(points[point_indices])

    return sampled_points_list
