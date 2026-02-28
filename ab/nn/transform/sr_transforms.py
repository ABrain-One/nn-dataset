
import random
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = F.hflip(target)
        return img, target

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.vflip(img)
            target = F.vflip(target)
        return img, target

class RandomRotation:
    def __init__(self, degrees=90):
        self.degrees = [0, 90, 180, 270]

    def __call__(self, img, target):
        angle = random.choice(self.degrees)
        if angle != 0:
            img = F.rotate(img, angle)
            target = F.rotate(target, angle)
        return img, target

# Default transform instance exposed for Loader.py
transform = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation()
])
