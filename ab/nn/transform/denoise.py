import torchvision.transforms as T

def transform(norm: tuple = None) -> T.Compose:
    return T.Compose([
        T.ToTensor() # Simply convert the already-cropped PIL image to a Tensor
    ])
