from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.data_collection.collect_GLDAS.CollectGLDAS import CollectGLDAS


def demo():
    col = CollectGLDAS(config=FileTool.get_project_dir("setting/data_collection/CollectGLDAS.json"))

    col.run(rewrite=False, log=True)


if __name__ == '__main__':
    demo()
