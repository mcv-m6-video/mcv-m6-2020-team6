from itertools import product

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.util.shape import view_as_windows


def find_match_offset(template, img):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return np.array(max_loc),np.array(max_val)


def crop_for_blocksize(frame, blocksize):
    new_h = frame.shape[0] - (frame.shape[0] % blocksize)
    new_w = frame.shape[1] - (frame.shape[1] % blocksize)
    return frame[:new_h, :new_w]


def block_matching(src, dst, blocksize, searcharea=4, compensation="Forward"):
    """

    Args:
        src (numpy.ndarray): The source frame in grayscale
        dst (numpy.ndarray): The destination frame in grayscale
        blocksize (int): Size of the block to match
        searcharea (int, optional): The window size in which to search for a match.
            Defaults to 4.
        compensation (str, optional): Whether to perform "Backward" or
            "Forward" compensation. Defaults to "Forward".

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): offset vectors
    """

    if compensation == "Backward":
        src, dst = dst, src

    src, dst = crop_for_blocksize(src, blocksize), crop_for_blocksize(dst, blocksize)
    blocks = view_as_windows(src, blocksize, blocksize)
    last_shape = blocks.shape[:2]
    blocks = blocks.reshape(-1, blocksize, blocksize)
    padded_dst = np.pad(dst, searcharea, mode='constant')
    search_size = 2 * searcharea + blocksize
    searchareas = view_as_windows(padded_dst, search_size, blocksize).reshape(
        -1, search_size, search_size
    )
    assert len(blocks) == len(searchareas)
    if compensation== 'Backward':
        offsets = np.vstack(
            [find_match_offset(*item)[0]- searcharea if(find_match_offset(*item)[1]>0.99) else [0,0] for item in zip(blocks, searchareas)]
        ).reshape(*last_shape, 2)
    else:
        offsets = np.vstack(
            [searcharea-find_match_offset(*item)[0] if(find_match_offset(*item)[1]>0.99) else [0,0] for item in zip(blocks, searchareas)]
        ).reshape(*last_shape, 2)

    return offsets, blocks


def plot_offsets(img, offsets):
    img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(img)
    assert offsets.ndim == 3
    step_h, step_w = img.size[0] // offsets.shape[1], img.size[1] // offsets.shape[0]
    gen_h = range(step_h // 2, img.size[0], step_h)
    gen_w = range(step_w // 2, img.size[1], step_w)
    offsets = offsets.reshape(-1, 2)
    for (i, j), offset in zip(product(gen_h, gen_w), offsets):
        if (offset != [0, 0]).any():
            draw.line([i, j, i + offset[0], j + offset[1]], fill="red")
    return img


def threshold_dim(off, vals):
    off = off.copy()
    bins = np.arange(off.max() - off.min()) + off.min()
    hist, _ = np.histogram(off, bins=bins)
    most_popular = np.argsort(hist) + off.min()
    m = np.ones_like(off)
    for i in range(vals):
        m = np.logical_and(m, off != most_popular[-i])
    off[m] = 0
    return off
