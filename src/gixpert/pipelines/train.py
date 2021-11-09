import os.path as osp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics    import binary_accuracy

from deeply.model.unet import (
    UNet,
    Trainer
)

from deeply.datasets.util import SPLIT_TYPES
from deeply.generators    import ImageMaskGenerator
from deeply.losses        import dice_loss

from gixpert.data   import get_data_dir
from gixpert.config import DEFAULT
from gixpert import dops, settings

def build_model(artifacts_path = None):
    dropout_rate  = settings.get("dropout_rate")
    batch_norm    = settings.get("batch_norm")

    width, height = settings.get("image_size")

    unet = UNet(x = width, y = height, n_classes = 1,
        final_activation = "sigmoid", batch_norm = batch_norm, 
        dropout_rate = dropout_rate, padding = "same")
    
    if artifacts_path:
        path_plot = osp.join(artifacts_path, "model.png")
        unet.plot(to_file = path_plot)

    return unet

def train(check = False, data_dir = None, artifacts_path = None, *args, **kwargs):
    batch_size    = 1 if check else settings.get("batch_size")
    learning_rate = settings.get("learning_rate")
    epochs        = settings.get("epochs")
    
    model = build_model()
    model.compile(optimizer = Adam(learning_rate = learning_rate),
        loss = dice_loss, metrics = [binary_accuracy])

    dops.watch(model)

    output_shape = model.output_shape[1:-1]

    data_dir = get_data_dir(data_dir)

    path_img = osp.join(data_dir, "%s", "images")
    path_msk = osp.join(data_dir, "%s", "masks")

    args = dict(
        batch_size = batch_size,
        color_mode = "grayscale",
        image_size = DEFAULT["image_size"],
        mask_size  = output_shape,
        shuffle    = True
    )

    train_, val, test = [
        ImageMaskGenerator(path_img % type_, path_msk % type_, **args)
            for type_ in SPLIT_TYPES
    ]

    trainer = Trainer(artifacts_path = artifacts_path)
    history = trainer.fit(model, train_, val = val, epochs = epochs, batch_size = batch_size)