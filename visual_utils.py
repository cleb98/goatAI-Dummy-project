from typing import List, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist


DEFAULT10 = [
    (196, 229, 56),
    (253, 167, 223),
    (247, 159, 31),
    (18, 137, 167),
    (181, 52, 113),
    (0, 148, 50),
    (153, 128, 250),
    (234, 32, 39),
    (76, 108, 252),
    (220, 76, 252),
]


def get_random_colors(n_colors):
    # type: (int) -> List[Tuple[int, int, int]]
    """
    Generate a list of `n_colors` random bright colors in RGB format.
    Colors are different from each other (as much as possible).

    :param n_colors: number of colors to generate
    :return: list of RGB colors, where each color is a tuple of (R, G, B)
        and each channel is in the range [0, 255]
    """

    # generate a pool of random bright colors
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 256, size=(n_colors * 128, 3), dtype=int)

    # select colors iteratively to maximize minimum distance
    selected_colors = DEFAULT10[:n_colors]
    for _ in range(len(selected_colors), n_colors):
        distance_matrix = cdist(np.array(selected_colors), colors)
        min_distances = np.min(distance_matrix, axis=0)
        next_color_index = np.argmax(min_distances)
        selected_colors.append(colors[next_color_index])

    return [tuple(color) for color in selected_colors]


def apply_mask_overlay(img, masks):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Applies colored overlays to an image based on binary masks.

    :param img: The input image to which the masks will be applied.
        ->> shape: (3, H, W)
        ->> values in range float[0, 1]
    :param masks: Binary masks indicating regions to overlay.
        ->> shape: (N_masks, H, W)
        ->> values in range bin{0, 1}
    :return: The image with colored mask overlays applied.
        ->> shape: (3, H, W)
        ->> values in range float[0, 1]
    """
    colors = get_random_colors(masks.shape[0])
    overlay = img.clone() * 0.4

    for class_idx in range(masks.shape[0]):
        # m0: binary mask for class `idx`; shape: (3, H, W)
        m0 = (masks[class_idx][None, ...] > 0).repeat((3, 1, 1))

        # m1: colored mask for class `idx`; shape: (3, H, W)
        m1 = masks[class_idx][None, ...].repeat((3, 1, 1))
        m1 *= torch.tensor(colors[class_idx]).reshape(-1, 1, 1) / 255.

        overlay[m0] = 0.25 * img[m0] + 0.75 * m1[m0]

    return overlay


def tensor_to_cv2(x):
    # type: (torch.Tensor) -> np.ndarray
    """
    Convert a torch tensor representing an image to a numpy array in
    the format expected by OpenCV.

    :param x: torch tensor representing an image
        ->> shape: (3, H, W)
        ->> values in range [0, 1]
    :return: image as a numpy array (OpenCV format)
        ->> shape: (H, W, 3)
        ->> values in range [0, 255] (np.uint8)
    """
    img = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img

def decode_and_apply_mask_overlay(img, masks):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Decodes binary masks into RGB images and applies colored overlays to a batch of images.

    :param img: Input images to which the masks will be applied.
        ->> shape: (B, 3, H, W)
        ->> values in range float[0, 1]
    :param masks: Binary masks indicating regions to overlay.
        ->> shape: (B, N_masks, H, W)
        ->> values in range bin{0, 1}
    :return: Images with colored mask overlays applied.
        ->> shape: (B, 3, H, W)
        ->> values in range float[0, 1]
    """
    B, _, H, W = img.shape
    _, N_masks, _, _ = masks.shape

    # Generate a colormap
    colormap = get_random_colors(N_masks)
    class_colors = torch.tensor(colormap, dtype=torch.uint8, device=img.device)

    # Decode masks into RGB
    background_color = torch.tensor([0, 0, 0], dtype=torch.uint8, device=img.device).unsqueeze(0)
    rgb_colors = torch.cat([background_color, class_colors], dim=0)

    # Background pixels (B, H, W)
    background_mask = torch.all(masks == 0, dim=1)
    class_map = masks.argmax(dim=1)  # (B, H, W)
    class_map[background_mask] = -1  # Set background to -1

    # Decode into RGB (B, H, W, 3) -> (B, 3, H, W)
    decoded_mask = rgb_colors[class_map + 1].permute(0, 3, 1, 2).float() / 255.0

    # Apply overlay
    overlay = img.clone() * 0.4 + decoded_mask * 0.6

    return overlay



def demo():
    # generate a grid of random colors
    n_rows = 5
    n_cols = 8
    colors = get_random_colors(n_rows * n_cols)
    grid = np.zeros((n_rows * 100, n_cols * 100, 3), dtype=np.uint8)
    for i in range(n_rows):
        for j in range(8):
            color = colors[i * n_cols + j]
            grid[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = color

    # display the grid
    cv2.imshow('grid', grid[..., ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    demo()
