import json
import pathlib
from pathlib import Path

import requests

from pysrc.auxiliary.aux_tool.FileTool import FileTool


def _download_onefile_with_token(url, token, save_path):
    headers = {
        'Authorization': f'Bearer {token}'
    }

    with requests.Session() as session:
        session.headers.update(headers)
        response = session.get(url)

        if response.status_code == 401:
            raise Exception("Authentication failed, please check your API token")
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)


def _download_files_with_token(file_list, token, save_dir, rewrite=False, log=True):
    with open(file_list) as f:
        url_list = f.readlines()

    while True:
        try:
            skip_for_all = True
            for i in range(len(url_list)):
                this_url = str(url_list[i]).replace("\n", "")
                filename = Path(this_url).name

                save_path = Path(save_dir) / filename
                if save_path.exists() and not rewrite:
                    if log:
                        print(f"path {save_path} exists, skip for this.")
                    continue
                if not save_path.parent.exists():
                    save_path.parent.mkdir()

                if log:
                    print(f"downloading {this_url}")
                _download_onefile_with_token(this_url, token, save_path)
                if log:
                    print(f"downloading successes：{save_path}")

                skip_for_all = False

            if skip_for_all:
                return 0

        except:
            print("meet error, retrying...")


class CollectGLDASConfig:
    def __init__(self):
        self.__token = None
        self.__files_dir = None
        self.__save_dir = None

    def from_json(self, json_path):
        assert isinstance(json_path, Path) or type(json_path) in (dict, str)

        if isinstance(json_path, Path) or type(json_path) in (str,):
            with open(json_path) as f:
                config_dict = json.load(f)
        else:
            config_dict = json_path

        assert set(config_dict.keys()) >= {"filelist", "token", "save_dir"}

        self.set_token(config_dict["token"])
        self.set_files(config_dict["filelist"])
        self.set_save_dir(config_dict["save_dir"])

    def set_token(self, v: str):
        self.__token = v
        return self

    def get_token(self):
        return self.__token

    def set_files(self, v: str):
        p = pathlib.Path(v)

        if p.is_absolute():
            self.__files_dir = p
        else:
            self.__files_dir = FileTool.get_project_dir(p)
        return self

    def get_files(self):
        return self.__files_dir

    def set_save_dir(self, v: str):
        p = pathlib.Path(v)

        if p.is_absolute():
            self.__save_dir = p
        else:
            self.__save_dir = FileTool.get_project_dir(p)
        return self

    def get_save_dir(self):
        return self.__save_dir


class CollectGLDAS:
    def __init__(self, config=None):
        self.configuration = CollectGLDASConfig()

        if config is not None:
            self.configuration.from_json(config)

    def run(self, rewrite=False, log=True):
        file_list = self.configuration.get_files()
        token = self.configuration.get_token()
        save = self.configuration.get_save_dir()

        _download_files_with_token(file_list, token, save, rewrite=rewrite, log=log)
