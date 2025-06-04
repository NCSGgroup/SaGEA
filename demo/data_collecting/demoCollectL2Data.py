"""
This demo is a validation for downloading GRACE Level-2 products
"""
import sys
sys.path.append('./')

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.data_collection.collect_GRACE_L2.CollectL2SH import CollectL2SH
from pysrc.data_collection.collect_GRACE_L2.CollectL2LowDeg import CollectL2LowDeg

from pysrc.auxiliary.preference.EnumClasses import *


def demo_collect_L2_SH():
    col = CollectL2SH()

    json_path = FileTool.get_project_dir() / 'setting/data_collection/CollectL2Data.json'
    col.config(preset=json_path)

    col.configuration.set_max_relink_times(999)
    col.run()


def demo_collect_L2_low_degrees():
    col = CollectL2LowDeg()

    col.configuration.set_file_id(file_id=L2LowDegreeFileID.TN11)
    col.run()

    col.configuration.set_file_id(file_id=L2LowDegreeFileID.TN13)
    col.run()

    col.configuration.set_file_id(file_id=L2LowDegreeFileID.TN14)
    col.run()


if __name__ == '__main__':
    demo_collect_L2_SH()
