import sys, shutil
from os import remove, scandir
from ab.nn.util.Const import model_script, stat_train_dir, db_dir


def main(nn_name):
    model_path = model_script(nn_name)
    remove(model_path)
    print(f'Model deleted: {model_path}')
    with scandir(stat_train_dir) as it:
        for entry in it:
            if entry.name.endswith(f'_{nn_name}'):
                shutil.rmtree(entry.path, ignore_errors=True)
                print(f'Stat deleted: {entry.path}')
    shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ab.nn.util.DeleteNN <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    main(model_name)