from deeply.model.unet import (
    UNet
)

from gixpert.data import get_data_dir

TARGET_IMAGE_SIZE = (256, 256)

def build_model():
    width, height = TARGET_IMAGE_SIZE
    unet = UNet(x = width, y = height, n_classes = 1,
        final_activation = "sigmoid", batch_norm = False, padding = "same")
    
    # unet.plot()

    return unet

def train(data_dir = None):
    data_dir = get_data_dir(data_dir)
    model    = build_model()
