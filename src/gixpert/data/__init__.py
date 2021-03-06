import os.path as osp

import numpy as np
import imgaug.augmenters as iaa
import tqdm as tq

from bpyutils.util.environ import getenv
from bpyutils.util.array   import sequencify
from bpyutils.util.types   import lmap, build_fn
from bpyutils.util.ml      import get_data_dir, get_dataset_tag
from bpyutils._compat      import iteritems
from bpyutils.log import get_logger
from bpyutils import parallel

import deeply.datasets as dd
from   deeply.datasets.util import SPLIT_TYPES, split as split_datasets, length as ds_length
from   deeply.util.image    import augment as augment_images
import deeply.img.augmenters as dia
import deeply

from gixpert.config import PATH
from gixpert import __name__ as NAME, dops, settings

_PREFIX  = NAME.upper()
logger   = get_logger(name = NAME)

DATASETS = eval(settings.get("datasets"))
DEFAULT_DATASET = DATASETS[0]

TAG = get_dataset_tag(NAME)

IMAGE_WIDTH     = settings.get("image_width")
IMAGE_HEIGHT    = settings.get("image_height")

AUGMENTER       = iaa.Sequential([
    dia.Combination([
        iaa.Fliplr(1.0),
        iaa.Flipud(0.5),
        iaa.TranslateX(percent = (-0.15, 0.15)),
        iaa.TranslateY(percent = (-0.15, 0.15)),
        iaa.ScaleX((0.9, 1.2)),
        iaa.ScaleY((0.9, 1.2)),
        iaa.Rotate(rotate = (-30, 30)),
        iaa.ShearX((-30, 30)),
        iaa.ShearY((-30, 30)),
        iaa.ElasticTransformation(alpha = (0, 30), sigma = 5.0)
    ]),
    iaa.Resize({ "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT })
])

def _augment_image(data, split_type, *args, **kwargs):
    data_dir    = get_data_dir(NAME, data_dir = kwargs.get("data_dir"))

    image, mask = np.copy(data[0]), np.copy(data[1])

    dir_path    = osp.join(data_dir, split_type)

    images_dir  = osp.join(dir_path, "images")
    masks_dir   = osp.join(dir_path, "masks")

    augmenter   = AUGMENTER

    augment_images(augmenter, images = image, masks = mask,
        dir_images = images_dir, dir_masks = masks_dir)

def _augment_split(split_type, *args, **kwargs):
    jobs  = kwargs.get("jobs", settings.get("jobs"))
    check = kwargs.get("check", False)
    data_dir = get_data_dir(NAME, data_dir = kwargs.get("data_dir"))

    dataset_name = kwargs.get("dataset_name")
    
    dataset = dd.load(dataset_name, data_dir = data_dir, shuffle_files = False)
    dataset = dataset["data"]
    
    dataset = split_datasets(dataset, splits = (.8, .1, .1))
    groups  = dict(zip(SPLIT_TYPES, dataset))

    split   = groups[split_type]

    if check:
        split = split.take(1)

    logger.info("Augmenting dataset %s for type %s..." % (dataset_name, split_type))

    with parallel.no_daemon_pool(processes = jobs) as pool:
        length    = ds_length(split)

        function_ = build_fn(_augment_image, split_type = split_type, *args, **kwargs)

        list(tq.tqdm(pool.imap(function_, split.batch(1)), total = length))

def _augment_dataset(dataset_name, *args, **kwargs):
    jobs = kwargs.get("jobs", settings.get("jobs"))

    with parallel.no_daemon_pool(processes = jobs) as pool:
        function_ = build_fn(_augment_split, dataset_name = dataset_name, *args, **kwargs)
        list(pool.imap(function_, SPLIT_TYPES))

def get_datasets(check = False):
    dataset_names = DATASETS

    if check:
        dataset_names = [DEFAULT_DATASET]

    return dataset_names

def get_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(NAME, data_dir = data_dir)
    dataset_names = get_datasets(check = check)

    try:
        dops.download('dataset:%s' % TAG, target_dir = data_dir)
    except deeply.exception.OpsError:
        logger.warn("No data object found. Building...")

        datasets = dd.load(*dataset_names, data_dir = data_dir)

def preprocess_data(data_dir = None, check = False, *args, **kwargs):
    jobs = kwargs.get("jobs", settings.get("jobs"))
    data_dir = get_data_dir(NAME, data_dir = data_dir)

    try:
        dops.download('dataset:%s' % TAG, target_dir = data_dir)
    except deeply.exception.OpsError:
        logger.warn("No data object found. Building...")

        dataset_names = get_datasets(check = check)

        with parallel.no_daemon_pool(processes = jobs) as pool:
            function_ = build_fn(_augment_dataset, check = check, *args, **kwargs)
            list(pool.imap(function_, dataset_names))

        config = [
            { "source": osp.join(data_dir, split_type), "destination": split_type }
                for split_type in SPLIT_TYPES
        ]

        dops.upload(*config, name = 'dataset:%s' % TAG)