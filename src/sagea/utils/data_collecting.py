#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/13 18:26 
# @File    : data_collecting.py

import os
import pathlib
import ssl

import wget
from sagea.utils.FileTool import FileTool


def bar(current, total, width):
    progress = current / total
    wid = 30
    bar_style = '#' * (int(wid * progress) + 1) + '-' * (wid - int(wid * progress) - 1)

    print(f'\r|{bar_style}| {"%.2f" % (current / total * 100)}%', end='')


def download_ddk_data(loc_path_temp, local_path_unzip):
    """download and unzip ddk data from https://zenodo.org/records/15679042."""

    """prepare"""

    """download data"""
    url = f"https://zenodo.org/records/15679042/files/ddk_data.zip?download=1"

    print(f"downloading: {url}")


    try:
        wget.download(url, str(loc_path_temp), bar=bar)
    except Exception as e:
        if e.args[0].errno == 1:
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(url, str(loc_path_temp), bar=bar)
        else:
            raise e
    else:
        pass
    print()

    print(f"unzipping: {loc_path_temp} ...")
    """unzip"""

    FileTool.un_zip(loc_path_temp, local_path_unzip)
    os.remove(loc_path_temp)
    print("done!\n")


if __name__ == '__main__':
    download_ddk_data(
        pathlib.Path("/Users/shuhao/PycharmProjects/SaGEA/src/sagea/data/temp.zip"),
        pathlib.Path("/Users/shuhao/PycharmProjects/SaGEA/src/sagea/data/ddk2"),
    )
