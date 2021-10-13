import os.path as osp

from tensorflow.keras.optimizers import Adam

from deeply.model.unet import (
    UNet,
    Trainer
)

from deeply.datasets.util import SPLIT_TYPES
from deeply.generators    import ImageMaskGenerator
from deeply.losses        import dice_loss

from gixpert.data import get_data_dir

IMAGE_SIZE = (256, 256)

def build_model():
    width, height = IMAGE_SIZE
    unet = UNet(x = width, y = height, n_classes = 1,
        final_activation = "sigmoid", batch_norm = False, padding = "same")
    
    # unet.plot()

    return unet

def train(batch_size = 1, learning_rate = 1e-5, epochs = 10, data_dir = None, *args, **kwargs):
    model = build_model()
    model.compile(optimizer = Adam(learning_rate = learning_rate),
        loss = dice_loss)

    output_shape = model.output_shape[1:-1]

    data_dir = get_data_dir(data_dir)

    path_img = osp.join(data_dir, "%s", "images")
    path_msk = osp.join(data_dir, "%s", "masks")

    args = dict(
        batch_size = batch_size,
        color_mode = "grayscale",
        image_size = IMAGE_SIZE,
        mask_size  = output_shape,
        shuffle    = True
    )

    train_, val, test = [
        ImageMaskGenerator(path_img % type_, path_msk % type_, **args) for type_ in SPLIT_TYPES
    ]
    
    trainer = Trainer()
    history = trainer.fit(model, train_, val = val, epochs = epochs)