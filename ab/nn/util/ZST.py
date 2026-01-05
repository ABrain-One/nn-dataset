import zstandard as zstd

from ab.nn.util.hf.HF import *


def compress(input_path: Path, output_path: Path, remove: bool = False):
    compressor = zstd.ZstdCompressor(
        level=22,  # strongest standard compression
        threads=0  # use all CPU cores
    )
    with open(input_path, "rb") as fin, open(output_path, "wb") as f:
        compressor.copy_stream(fin, f)
    if remove: os.remove(input_path)


def decompress(input_path: Path, output_path: Path, remove: bool = False):
    decompressor = zstd.ZstdDecompressor()

    with open(input_path, "rb") as fin, open(output_path, "wb") as f:
        decompressor.copy_stream(fin, f)
    if remove: os.remove(input_path)
