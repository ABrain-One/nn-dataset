import torchvision.transforms as transforms


def transform(norm):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
