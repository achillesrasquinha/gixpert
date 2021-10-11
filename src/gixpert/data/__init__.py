import os.path as osp

from gixpert.config import PATH
from gixpert import __name__ as NAME

from bpyutils.util.environ import getenv
from bpyutils.util.system  import makedirs

import deeply.datasets as dd

_PREFIX  = NAME.upper()

DATASETS = ("hyper_kvasir_segmented",)

def get_data_dir(data_dir = None):
    data_dir = data_dir \
        or getenv("DATA_DIR", prefix = _PREFIX) \
        or osp.join(PATH["CACHE"], "data")

    makedirs(data_dir, exist_ok = True)

    return data_dir

def get_data(data_dir = None):
    data_dir = get_data_dir(data_dir)
    dd.load(*DATASETS, data_dir = data_dir)

def preprocess_data(data_dir = None):
    data_dir = get_data_dir(data_dir)
    datasets = dd.load(*DATASETS, data_dir = data_dir)