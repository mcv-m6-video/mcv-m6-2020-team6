import os
import pickle
from pathlib import Path

import click
import cv2
import imageio
from tqdm.auto import tqdm

from matching import block_matching


def spit_two(gen, start=None, stop=None):
    it = iter(gen)
    if start is not None:
        while start != 0:
            _ = next(it)
            start -= 1
    old = next(it)
    while stop is None or stop != 0:
        try:
            new = next(it)
        except StopIteration:
            return
        yield old, new
        old = new
        if stop:
            stop -= 1


def compute_offsets(
    frames, block_size, search_area, start=None, stop=None, progress=False
):
    if progress:
        print("Computing offsets")
        start, stop = start or 0, stop or frames.count_frames()
        frames = tqdm(frames, total=stop - start)
    for i, (src, dst) in enumerate(spit_two(frames, start=start, stop=stop)):
        src, dst = (
            cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY),
        )
        offs, _ = block_matching(src, dst, block_size, search_area)
        yield offs


def compute_and_save(
    block_size=8,
    search_area=None,
    video_file="/data/stab/stabilization.mp4",
    pickle_file=None,
    begin=None,
    end=None,
    verbose=False,
):

    if search_area is None:
        search_area = block_size

    video_file = Path(video_file)
    frames = imageio.get_reader(video_file)
    begin, end = begin or 0, end or frames.count_frames()

    if pickle_file is None:
        pickle_file = Path(
            f"computed/offsets_{video_file.stem}_{block_size}_{search_area}.pkl"
        )
    elif pickle_file != "-1":
        pickle_file = Path(pickle_file)

    offs = None
    if pickle_file != "-1" and pickle_file.exists():
        with open(pickle_file, "rb") as ofs:
            try:
                offs = pickle.load(ofs)
            except EOFError:
                pass

    if offs is not None:
        return offs[begin:end]

    if offs is None:
        offs = list(
            compute_offsets(frames, block_size, search_area, begin, end, verbose)
        )
        if pickle_file != "-1" and begin == 0 and end == frames.count_frames():
            os.makedirs(pickle_file.parent, exist_ok=True)
            with open(pickle_file, "wb") as ofs:
                pickle.dump(offs, ofs)
    return offs


@click.command()
@click.option(
    "-b", "--block_size", default=8, type=int, help="Block size for block matching"
)
@click.option(
    "-s", "--search_area", default=None, type=int, help="Defaults to block_size"
)
@click.option(
    "-v", "--video_file", default="/data/stab/stabilization.mp4", help="File to run on"
)
@click.option(
    "-o",
    "--pickle_file",
    default=None,
    type=str,
    help=(
        "File for pickles."
        "By default 'computed/offsets_{video_name}_{block_size}_{search_area}.pkl'."
        " Use -1 for not saving the results at all."
    ),
)
@click.option("-p", "--begin", default=None, help="Begin computing at frame")
@click.option("-q", "--end", default=None, help="Finish computation at frame")
@click.option("-v", "--verbose", is_flag=True)
def compute_and_save_command(
    block_size, search_area, video_file, pickle_file, begin, end, verbose
):
    compute_and_save(
        block_size, search_area, video_file, pickle_file, begin, end, verbose
    )


if __name__ == "__main__":
    compute_and_save_command()
