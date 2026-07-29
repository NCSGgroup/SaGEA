import warnings

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from pysrc.data_class.SHC import SHC


def plot_rms(
        shc: SHC = None,
        rmss=None,
        save=None,
        plot_from_degree=None,
        labels: list = None,
        legend_param: dict = None,
        line_param: dict = None,
        frame_param: dict = None,
        get_fig_only=False
):
    config = {
        "font.family": 'serif',
        "font.size": 16,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    matplotlib.rcParams.update(config)

    assert (shc is None and rmss is not None) or (shc is not None and rmss is None)

    if rmss is None:
        rms_list = list(shc.get_degree_rms())
        lmax = shc.get_lmax()
    else:
        rms_list = list(rmss)
        lmax = np.shape(rms_list)[1] - 1

    for i in range(len(rms_list)):
        assert len(rms_list[i]) - 1 == lmax

    '''line parameters'''
    if line_param is not None and "style" in line_param.keys():
        assert len(rms_list) == len(line_param["style"])
        line_style_list = line_param["style"]
    else:
        line_style_list = [None] * len(rms_list)

    if line_param is not None and "color" in line_param.keys() is not None:
        assert len(rms_list) == len(line_param["color"])
        line_color_list = line_param["color"]
    else:
        line_color_list = [None] * len(rms_list)

    if line_param is not None and "width" in line_param.keys() is not None:
        assert len(rms_list) == len(line_param["width"])
        line_width_list = line_param["width"]
    else:
        line_width_list = [None] * len(rms_list)

    if line_param is not None and "alpha" in line_param.keys() is not None:
        assert len(rms_list) == len(line_param["alpha"])
        line_alpha_list = line_param["alpha"]
    else:
        line_alpha_list = [None] * len(rms_list)
    '''end of line parameters'''

    '''label parameters'''
    if labels is not None:
        assert len(rms_list) == len(labels)
    else:
        labels = [None] * len(rms_list)

    if legend_param is not None and "loc" in legend_param.keys():
        legend_loc = legend_param["loc"]
    else:
        legend_loc = "best"

    if legend_param is not None and "hide" in legend_param.keys():
        legend_hide = legend_param["hide"]
    else:
        legend_hide = False

    if legend_param is not None and "fontsize" in legend_param.keys():
        legend_fontsize = legend_param["fontsize"]
    else:
        legend_fontsize = None
    '''end of label parameters'''

    '''frame parameters'''
    if frame_param is not None and "ylim" in frame_param.keys():
        set_ylim = frame_param["ylim"]
    else:
        set_ylim = None

    if frame_param is not None and "ylabel" in frame_param.keys():
        set_ylabel = frame_param["ylabel"]
    else:
        set_ylabel = None

    if frame_param is not None and "xlabel" in frame_param.keys():
        set_xlabel = frame_param["xlabel"]
    else:
        set_xlabel = None

    if frame_param is not None and "yscale" in frame_param.keys():
        set_yscale = frame_param["yscale"]
    else:
        set_yscale = None

    '''end of frame parameters'''

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0.20, 0.15, 0.60, 0.80])

    ax.set_xlim(0, lmax + 1)
    if set_ylim is not None:
        ax.set_ylim(*set_ylim)

    if set_yscale is not None:
        ax.set_yscale(set_yscale)

    degrees = np.arange(0, lmax + 1, 1)
    if plot_from_degree is None:
        plot_from_degree = 1
    for i in range(len(rms_list)):
        rms = rms_list[i]
        ax.plot(
            degrees[plot_from_degree:], rms[plot_from_degree:],
            label=labels[i] if labels is not None else None,
            linestyle=line_style_list[i],
            color=line_color_list[i],
            alpha=line_alpha_list[i],
            linewidth=line_width_list[i]
        )

    if labels is not None:
        if legend_hide:
            warnings.warn("To show legend, please set legend_param['hide'] as False")

        else:
            ax.legend(
                loc=legend_loc,
                fontsize=legend_fontsize
            )

    ax.grid(ls="--")

    if set_xlabel is not None:
        ax.set_xlabel(set_xlabel, fontsize=18)

    if set_ylabel is not None:
        ax.set_ylabel(set_ylabel, fontsize=18)

    if get_fig_only:
        if save is not None:
            warnings.warn("To save or show figure, please set param get_fig_only as False")

        return fig, ax

    else:
        if save is not None:
            plt.savefig(save, dpi=200, transparent=True)

        plt.show()
        plt.close()

    pass
