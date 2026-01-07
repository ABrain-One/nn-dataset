import argparse
import os
import shutil

from ab.nn.util.Const import *
from ab.nn.util.ZST import *
from ab.nn.util.db.Write import init_population
from ab.nn.util.hf.DB_from_HF import repo_id


def clean_gen_folders():
    for gen_folder in gen_folders:
        shutil.rmtree(gen_folder)


def db2hf(remove_gen_folders=False, hf_token=None):
    init_population()
    compress(db_file, zst_db_file, True)
    from ab.nn.util.hf.HF import upload_file
    upload_file(repo_id, zst_db_file, zst_db_file, True, hf_token=hf_token)
    if remove_gen_folders: clean_gen_folders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_gen_folders', action=argparse.BooleanOptionalAction)
    parser.add_argument('--HF_TOKEN', type=str, default=None)
    a = parser.parse_args()
    if a.HF_TOKEN: os.environ["MY_API_KEY"] = a.HF_TOKEN
    db2hf(remove_gen_folders=a.remove_gen_folders, hf_token=a.HF_TOKEN)
