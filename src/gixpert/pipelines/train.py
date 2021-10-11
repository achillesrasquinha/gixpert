from deeply.model.unet import (
    UNet
)

TARGET_IMAGE_SIZE = (256, 256)

def build_model():
    width, height = TARGET_IMAGE_SIZE
    unet = UNet(x = width, y = height, n_classes = 1,
        final_activation = "sigmoid", batch_norm = False, padding = "same")

def train():
    pass