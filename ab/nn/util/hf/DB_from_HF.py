import shutil

from ab.nn.util.Const import *
from ab.nn.util.ZST import *

def clean_gen_folders():
    for gen_folder in gen_folders:
        shutil.rmtree(gen_folder)


repo_id = f'{HF_NN}/LEMUR_DB'

def db_from_hf():
    from ab.nn.util.hf.HF import download
    download(repo_id, zst_db_file, db_dir)
    decompress(zst_db_file, db_file, True)

if __name__ == "__main__":
    db_from_hf()