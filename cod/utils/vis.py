from typing import Union

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def tensor_to_img(x):
    if isinstance(x, torch.Tensor):
        x = x.float().detach().cpu().numpy()

    return (x * 255).astype(np.uint8)


def visualize_point_clouds(points):
    import open3d as o3d

    points = _tensor_to_numpy(points)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='point cloud visualizer')
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().show_coordinate_frame = True
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().change_field_of_view(step=-90)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[..., :3])
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points[..., :3]))
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def points_to_img(points, colors='blue', zdir='z', view_angle=(30, -45),
                  points_range=None,
                  point_size=3, size=(2, 2)) -> np.ndarray:
    matplotlib.use('Agg')
    points = _tensor_to_numpy(points)
    if points.shape[0] == 0:
        return np.ones((*[x * 100 for x in size], 3), dtype=np.uint8) * 255.
    if isinstance(colors, np.ndarray) or isinstance(colors, torch.Tensor):
        colors = _tensor_to_numpy(colors)

    fig = plt.figure(figsize=size)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    ax = fig.add_subplot(projection='3d', adjustable='box')
    ax.view_init(view_angle[0], view_angle[1])
    if points_range is not None:
        min_val, max_val = points_range
    else:
        min_val, max_val = np.min(points), np.max(points)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.set_axis_off()

    ax.scatter(x, y, z, zdir=zdir, c=colors, s=point_size, depthshade=False)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def _tensor_to_numpy(x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
