import torchvision.transforms as T

def transform(norm=None):
    # The framework calls this. If it returns None, you get your error.
    # We return a simple identity transform to stay safe.
    return T.Compose([
        T.ToTensor()
    ])
