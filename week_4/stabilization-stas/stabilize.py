from itertools import chain, islice

import click
import imageio
import numpy as np
from tqdm.auto import tqdm

from compute_offsets import compute_and_save


def fit_frame_in_stab(frame, stab, px, py):
    qx, qy = frame.shape[0] + px, frame.shape[1] + py
    if px > stab.shape[0] or py > stab.shape[1] or min(qx, qy) < 0:
        return stab
    if px < 0:
        frame = frame[-px:, :]
    if py < 0:
        frame = frame[:, -py:]
    px, py = max(px, 0), max(py, 0)
    if qx > stab.shape[0]:
        diff = qx - stab.shape[0]
        frame = frame[: frame.shape[0] - diff, :]
    if qy > stab.shape[1]:
        diff = qy - stab.shape[1]
        frame = frame[:, : frame.shape[1] - diff]
    qx, qy = min(qx, stab.shape[0]), min(qy, stab.shape[1])
    stab[px:qx, py:qy] = frame
    return stab


def stabilize(frames, offsets, out_file="out.gif", verbose=False, fps=5):
    offsets[:, [0, 1]] = offsets[:, [1, 0]]
    if verbose:
        print("Stabilizing video")
        if hasattr(frames, "count_frames"):
            frames = tqdm(frames, total=frames.count_frames())
        elif type(verbose) == int:
            frames = tqdm(frames, total=verbose)
        else:
            frames = tqdm(frames)

    writer = imageio.get_writer(out_file, fps=fps)
    max_offset = np.abs(offsets).max()

    px, py = max_offset, max_offset

    offsets = chain([np.zeros(2).astype(int)], offsets)
    for frame, offset in zip(frames, offsets):
        out_shape = np.array(frame.shape[:2]) + 2 * max_offset
        stab_frame = np.zeros((*out_shape, 3), dtype=np.uint8)

        px, py = offset + (px, py)
        fit_frame_in_stab(frame, stab_frame, px, py)
        writer.append_data(stab_frame)
    writer.close()


def reduce_offsets(offsets):
    # TODO Find more fancy ways for better results

    def most_popular(off, amount=1):
        bins = np.arange(off.max() - off.min()) + off.min()
        hist, _ = np.histogram(off, bins=bins)
        most_popular = np.argsort(hist) + off.min()
        if amount == 1:
            return most_popular[-1]
        return most_popular[-amount:]

    def get_offset_vector(off):
        off = off.reshape(-1, 2)
        return -most_popular(off[:, 0]), -most_popular(off[:, 1])

    # return np.array([get_offset_vector(offset) for offset in offsets])
    return (
        np.stack([-offset.mean(axis=0).mean(axis=0).astype(int) for offset in offsets])
        * 8
    )


@click.command()
@click.option("-v", "--video_path", type=str, default="/data/stab/stabilization.mp4")
@click.option(
    "--pickle_file",
    "--pkl",
    help="Read/Write offsets from/to pickle file. -1 to disable pickle",
)
@click.option(
    "-o", "--output_file", type=str, default="out.gif", help="Can also be .mp4"
)
@click.option("--fps", type=int, default=3, help="Output frames per second")
@click.option(
    "-b", "--block_size", default=8, type=int, help="Block size for block matching"
)
@click.option(
    "-s", "--search_area", default=None, type=int, help="Defaults to block_size"
)
@click.option(
    "-p",
    "--begin",
    default=None,
    type=int,
    help="Begin computing at frame",
)
@click.option(
    "-q",
    "--end",
    default=None,
    type=int,
    help="Finish computation at frame",
)
@click.option("--verbose", is_flag=True)
def stabilize_command(
    video_path,
    pickle_file,
    output_file,
    fps,
    block_size,
    search_area,
    begin,
    end,
    verbose,
):

    offsets = compute_and_save(
        block_size=block_size,
        search_area=search_area,
        video_file=video_path,
        pickle_file=pickle_file,
        begin=begin,
        end=end,
        verbose=verbose,
    )
    stab_offsets = reduce_offsets(offsets)

    frames = imageio.get_reader(video_path)
    begin, end = begin or 0, end or frames.count_frames()
    frames = islice(frames, begin, end)

    stabilize(frames, stab_offsets, output_file, end - begin, fps)


if __name__ == "__main__":
    stabilize_command()
