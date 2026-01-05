import argparse

from ab.nn.util.Const import *
from ab.nn.util.ZST import *
from ab.nn.util.db.Write import init_population
from ab.nn.util.hf.HF import *
from ab.nn.util.hf.DB_from_HF import repo_id


def clean_gen_folders():
    for gen_folder in gen_folders:
        shutil.rmtree(gen_folder)


def db2hf(remove_gen_folders: bool = False):
    init_population()
    compress(db_file, zst_db_file, True)
    upload_file(repo_id, zst_db_file, zst_db_file, True)
    if remove_gen_folders: clean_gen_folders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_gen_folders', type=bool, default=False)
    a = parser.parse_args()
    db2hf(remove_gen_folders=a.remove_gen_folders)
