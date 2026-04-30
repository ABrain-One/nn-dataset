from torchvision import transforms

def transform(norm):
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
