import sys
sys.path.append('./')
import os
import ssl

import wget

from pysrc.auxiliary.aux_tool.FileTool import FileTool


def bar(current, total, width):
    progress = current / total
    wid = 30
    bar_style = '#' * (int(wid * progress) + 1) + '-' * (wid - int(wid * progress) - 1)

    print(f'\r|{bar_style}| {"%.2f" % (current / total * 100)}%', end='')


def demo():
    """download and unzip auxiliary and validation data from https://zenodo.org/records/14087017."""

    """prepare"""
    for folder in ("temp", "data", "validation"):
        local_path = FileTool.get_project_dir(f"{folder}/")
        if not local_path.exists():
            local_path.mkdir(parents=True)

    filenames = ("aux_data", "basin_mask", "products", "validation")
    for filename in filenames:
        """download data"""
        url = f"https://zenodo.org/records/14087017/files/{filename}.zip?download=1"
        local_path = FileTool.get_project_dir(f"temp/{filename}.zip")

        print(f"downloading: {url}")

        try:
            wget.download(url, str(local_path), bar=bar)
        except Exception as e:
            if e.args[0].errno == 1:
                ssl._create_default_https_context = ssl._create_unverified_context
                wget.download(url, str(local_path), bar=bar)
            else:
                raise e
        else:
            pass


        print()

        print(f"unzipping: {local_path} ...")
        """unzip"""
        if filename in ("aux_data", "products"):
            local_path_unzip = FileTool.get_project_dir("data")
        elif filename in ("basin_mask",):
            local_path_unzip = FileTool.get_project_dir("data/basin_mask")
        elif filename in ("validation",):
            local_path_unzip = FileTool.get_project_dir("validation")
        else:
            assert False

        FileTool.un_zip(local_path, local_path_unzip)
        os.remove(local_path)
        print("done!\n")


if __name__ == '__main__':
    demo()
