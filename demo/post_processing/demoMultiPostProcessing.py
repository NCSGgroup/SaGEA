import itertools
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from demo.post_processing.demoPostProcessing import PostProcessing
from pysrc.auxiliary.aux_tool.FileTool import FileTool


def get_one_process_json_file(filepath: pathlib.WindowsPath):
    with open(filepath, 'r') as f:
        dict_from_jason = json.load(f)

    for key in dict_from_jason.keys():
        print(key, dict_from_jason[key])

    begin_date = dict_from_jason["begin_date"]
    end_date = dict_from_jason["GRACE_institute"]
    basin = dict_from_jason["basin"]

    institute_list = dict_from_jason["GRACE_institute"]
    institute_indexes = range(len(institute_list))

    release_list = dict_from_jason["GRACE_release"]
    release_indexes = range(len(release_list))

    lmax_list = dict_from_jason["lmax"]
    lmax_indexes = range(len(lmax_list))

    replace_low_degree_list = dict_from_jason["replace_low_degree"]
    replace_low_degree_indexes = range(len(replace_low_degree_list))

    low_degree_degree1_file_id = dict_from_jason["low_degree_degree1_file_id"]  # not a list

    low_degree_c20_file_id_list = dict_from_jason["low_degree_c20_file_id"]
    low_degree_c20_file_id_indexes = range(len(low_degree_c20_file_id_list))

    low_degree_c30_file_id = dict_from_jason["low_degree_c30_file_id"]  # not a list

    de_correlation_list = dict_from_jason["de_correlation"]
    de_correlation_sliding_window_type_list = dict_from_jason["de_correlation_sliding_window_type"]
    de_correlation_params_list = dict_from_jason["de_correlation_params"]
    assert len(de_correlation_list) == len(de_correlation_sliding_window_type_list) == len(de_correlation_params_list)
    de_correlation_indexes = range(len(de_correlation_list))

    shc_filter_list = dict_from_jason["shc_filter"]
    shc_filter_params_list = dict_from_jason["shc_filter_params"]
    assert len(shc_filter_list) == len(shc_filter_params_list)
    shc_filter_indexes = range(len(shc_filter_list))

    leakage_method_list = dict_from_jason["leakage_method"]
    leakage_method_indexes = range(len(leakage_method_list))

    GIA_model_list = dict_from_jason["GIA_model"]
    GIA_model_indexes = range(len(GIA_model_list))

    combination = itertools.product(
        institute_indexes, release_indexes, lmax_indexes, replace_low_degree_indexes, low_degree_c20_file_id_indexes,
        de_correlation_indexes, shc_filter_indexes, leakage_method_indexes, GIA_model_indexes
    )

    combination_list = list(combination)

    config_dict_list = []
    for combine_results in combination_list:
        this_pp_config_dict = dict(
            begin_date=begin_date,
            end_date=end_date,
            basin=basin,
            GRACE_institute=institute_list[combine_results[0]],
            GRACE_release=release_list[combine_results[1]],
            lmax=lmax_list[combine_results[2]],
            replace_low_degree=replace_low_degree_list[combine_results[3]],
            low_degree_degree1_file_id=low_degree_degree1_file_id,
            low_degree_c20_file_id=low_degree_c20_file_id_list[combine_results[4]],
            low_degree_c30_file_id=low_degree_c30_file_id,
            de_correlation=de_correlation_list[combine_results[5]],
            de_correlation_sliding_window_type=de_correlation_sliding_window_type_list[combine_results[5]],
            de_correlation_params=de_correlation_params_list[combine_results[5]],
            shc_filter=shc_filter_list[combine_results[6]],
            shc_filter_params=shc_filter_params_list[combine_results[6]],
            leakage_method=leakage_method_list[combine_results[7]],
            GIA_model=GIA_model_list[combine_results[8]],
        )

        config_dict_list.append(this_pp_config_dict)

    return config_dict_list


def demo():
    config_list = get_one_process_json_file(
        FileTool.get_project_dir() / 'setting/post_processing/MultiPostProcessing.json')

    for i in trange(len(config_list)):
        path = FileTool.get_project_dir() / f'results/multi_post_processing/{i+1728}'
        if path.exists():
            if {"params.txt", "results.npz"} <= set([list(path.iterdir())[i].name for i in range(len(list(path.iterdir())))]):
                continue

        else:
            pathlib.Path.mkdir(path, parents=True)

        this_config = config_list[i]

        pp = PostProcessing()
        pp.configuration.set_from_json(this_config)

        pp.prepare()
        pp.load_files()  # load GRACE SHC, and replace low-degrees
        pp.correct_gia()  # GIA correction
        # pp.de_correlation()  # de-correlation filter
        pp.filter()  # low-pass filter
        pp.shc_to_grid()  # synthesis harmonic to (EWH) grid
        pp.correct_leakage()  # leakage correction

        config_dict = pp.configuration.__dict__

        with open(path / 'params.txt', 'w+') as f:
            for key in config_dict.keys():
                f.write(f'{key}\t{config_dict[key]}\n')

        times = pp.get_year_fraction()
        values = pp.get_ewh()
        np.savez(path / 'results.npz', times=times, values=values)


if __name__ == '__main__':
    demo()
