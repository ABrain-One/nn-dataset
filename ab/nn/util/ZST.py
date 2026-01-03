import os

import zstandard as zstd

from ab.nn.util.Const import *


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


def clean_gen_folders():
    pass


def compress_db():
    compress(db_file, zst_db_file, True)
    clean_gen_folders()


def decompress_db():
    decompress(zst_db_file, db_file, True)


if __name__ == "__main__":
    compress_db()
    decompress_db()
