import random
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomCrop:
    def __init__(self, size=192, scale=3):
        self.size = size
        self.scale = scale

    def __call__(self, img, target):
        lr_size = self.size // self.scale
        _, lr_h, lr_w = img.shape
        
        top = random.randint(0, max(0, lr_h - lr_size))
        left = random.randint(0, max(0, lr_w - lr_size))
        
        hr_top = top * self.scale
        hr_left = left * self.scale

        img = F.crop(img, top, left, lr_size, lr_size)
        target = F.crop(target, hr_top, hr_left, self.size, self.size)
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

def transform(norm=None):
    return Compose([
        RandomCrop(size=192, scale=3),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation()
    ])
