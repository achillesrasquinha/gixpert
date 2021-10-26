import os.path as osp

import numpy as np
import imgaug.augmenters as iaa
import tqdm as tq

from bpyutils.util.environ import getenv
from bpyutils.util.system  import makedirs
from bpyutils.util.array   import sequencify
from bpyutils.util.types   import lmap
from bpyutils.util._dict   import merge_dict
from bpyutils._compat      import iteritems

import deeply.datasets as dd
from   deeply.datasets.util import SPLIT_TYPES, split as split_datasets
from   deeply.util.image    import augment as augment_images
import deeply.img.augmenters as dia

from gixpert.config import PATH
from gixpert.const  import IMAGE_SIZE
from gixpert import __name__ as NAME, dops

_PREFIX  = NAME.upper()

DATASETS = (
    "cvc_clinic_db",
    # "etis_larib",
    # "kvasir_segmented",
    # "hyper_kvasir_segmented"
)

def get_data_dir(data_dir = None):
    data_dir = data_dir \
        or getenv("DATA_DIR", prefix = _PREFIX) \
        or osp.join(PATH["CACHE"], "data")

    makedirs(data_dir, exist_ok = True)

    return data_dir

def get_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(data_dir)
    datasets = dd.load(*DATASETS, data_dir = data_dir)

def preprocess_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(data_dir)
    datasets = lmap(
        lambda x: x["data"],
        sequencify(dd.load(*DATASETS, data_dir = data_dir, shuffle_files = True))
    )

    width, height   = IMAGE_SIZE
    # TODO: Add Image Enhancers.
    image_augmentor = iaa.Sequential([
        iaa.Resize({ "width": width, "height": height })
    ])

    mask_augmentor  = iaa.Sequential([
        iaa.Resize({ "width": width, "height": height }),
        dia.Dilate(kernel = np.ones((15, 15)))
    ])
    
    for dataset in datasets:
        dataset = split_datasets(dataset)
        groups  = dict(zip(SPLIT_TYPES, dataset))

        for split_type, split in iteritems(groups):
            dir_path = osp.join(data_dir, split_type)

            if check:
                split = split.take(3)

            for i, data in enumerate(tq.tqdm(split.batch(1))):
                augment_images(image_augmentor, images = data["image"].numpy(), filename = osp.join(dir_path, "images", "%s.jpg" % i))
                augment_images(mask_augmentor,  images = data["mask"].numpy(),  filename = osp.join(dir_path, "masks",  "%s.jpg" % i))

    # config = [
    #     { "source": osp.join(data_dir, split_type), "destination": split_type }
    #         for split_type in SPLIT_TYPES
    # ]
    # dops.upload(*config)