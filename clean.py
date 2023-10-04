import os
import shutil

main_dir = os.path.dirname(os.path.abspath(__file__))


def clean_data():
    dir = os.path.join(main_dir, "static/data/input")
    for folder in os.listdir(dir):
        folder = os.path.join(dir, folder)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

    dir2 = os.path.join(main_dir, "static\\result")
    for folder in os.listdir(dir2):
        folder = os.path.join(dir2, folder)
        shutil.rmtree(folder)


clean_data()
