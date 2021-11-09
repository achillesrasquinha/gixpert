import os.path as osp

import numpy as np
import imgaug.augmenters as iaa
import tqdm as tq

from bpyutils.util.environ import getenv
from bpyutils.util.system  import makedirs
from bpyutils.util.array   import sequencify
from bpyutils.util.types   import lmap
from bpyutils.util.string  import get_random_str
from bpyutils._compat      import iteritems
from bpyutils.log import get_logger

import deeply.datasets as dd
from   deeply.datasets.util import SPLIT_TYPES, split as split_datasets
from   deeply.util.image    import augment as augment_images
import deeply.img.augmenters as dia
import deeply

from gixpert.config import PATH, DEFAULT
from gixpert import __name__ as NAME, dops, settings

_PREFIX  = NAME.upper()
logger   = get_logger(name = NAME)

DATASETS = (
    "cvc_clinic_db",
    "etis_larib",
    "kvasir_segmented",
    "hyper_kvasir_segmented"
)

def get_data_dir(data_dir = None):
    data_dir = data_dir \
        or getenv("DATA_DIR", prefix = _PREFIX) \
        or osp.join(PATH["CACHE"], "data")

    makedirs(data_dir, exist_ok = True)

    return data_dir

def get_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(data_dir)

    try:
        dops.download('dataset:latest', target_dir = data_dir)
    except deeply.exception.OpsError:
        logger.warn("No data object found. Building...")

        datasets = dd.load(*DATASETS, data_dir = data_dir)

def preprocess_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(data_dir)

    try:
        dops.download('dataset:latest', target_dir = data_dir)
    except deeply.exception.OpsError:
        logger.warn("No data object found. Building...")

        datasets = lmap(
            lambda x: x["data"],
            sequencify(dd.load(*DATASETS, data_dir = data_dir, shuffle_files = True))
        )

        width, height  = settings.get("image_size")

        base_augmentor = iaa.Sequential([
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
            iaa.Resize({ "width": width, "height": height })
        ])

        image_augmentor = base_augmentor # TODO: Add Image Enhancers.

        mask_augmentor  = iaa.Sequential([
            base_augmentor,
            dia.Dilate(kernel = np.ones((10, 10)))
        ])

        for i, dataset in enumerate(datasets):
            dataset = split_datasets(dataset, splits = (.8, .1, .1))
            groups  = dict(zip(SPLIT_TYPES, dataset))

            for split_type, split in iteritems(groups):
                dir_path   = osp.join(data_dir, split_type)

                images_dir = osp.join(dir_path, "images")
                masks_dir  = osp.join(dir_path, "masks")

                if check:
                    split = split.take(3)

                logger.info("Augmenting dataset %s for type %s..." % (DATASETS[i], split_type))

                for data in tq.tqdm(split.batch(1)):
                    prefix = get_random_str()

                    image, mask = data["image"].numpy(), data["mask"].numpy()

                    augment_images(image_augmentor, images = image, dir_path = images_dir, prefix = prefix)
                    augment_images(mask_augmentor,  images = mask,  dir_path = masks_dir,  prefix = prefix)

        config = [
            { "source": osp.join(data_dir, split_type), "destination": split_type }
                for split_type in SPLIT_TYPES
        ]
        dops.upload(*config, name = 'dataset')