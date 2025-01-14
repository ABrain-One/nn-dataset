import os
from os import makedirs, mkdir
from os.path import join, exists
from typing import Sequence

import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from pycocotools import mask as cocmask
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from ab.nn.util.Const import data_dir

coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
coco_img_url = "http://images.cocodataset.org/zips/{}2017.zip"

# Reduce COCOS classes:
# MIN_CLASS_LIST = [0, 1, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 5, 64, 20, 63, 7, 72]
# MIN_CLASS_LIST = [0, 1, 16, 9, 44, 6, 18, 5, 64, 20, 63, 7]
MIN_CLASS_LIST = [0, 1, 2, 3, 4, 5]
MIN_CLASS_N = len(MIN_CLASS_LIST)

def class_n ():
    return MIN_CLASS_N

def get_class_list():
    return MIN_CLASS_LIST

__norm_mean = (104.01362025, 114.03422265, 119.9165958)
__norm_dev = (73.6027665 , 69.89082075, 70.9150767)
__minimum_accuracy = 0.11 # todo: Required correct value for minimum accuracy provided by the untrained NN model due to random output generation

def loader(transform_fn):
    path = join(data_dir, 'coco')
    transform = transform_fn((__norm_mean, __norm_dev))
    resize = None
    for i in transform.transforms:
        if i.__class__ == transforms.Resize:
            if isinstance(i.size, Sequence):
                h, w = i.size
                resize = (w, h)
    if not resize: raise Exception("The transformer must include a resize function for the image segmentation task.")
    train_set = COCOSegDataset(transform=transform, root=path, spilt="train", resize=resize, preprocess=True)
    val_set = COCOSegDataset(transform=transform, root=path, spilt="val", resize=resize, preprocess=True)
    return (class_n(),), __minimum_accuracy, train_set, val_set


class COCOSegDataset(torch.utils.data.Dataset):
    def __init__(self, transform, resize, root:os.path, spilt="val", class_limit = None, num_limit = None, preprocess=True, least_pix=1000):
        """Read datas from COCOS and generate 2D masks.

        Parameters
        ----------
        path : Path towards coco root directory. It should be structured as default.
        spilt : str `"train"` or `"val"`.
        transform : transform towards the image. If transform.Resize is specified, 
          it will overwrite `resize` parameter, BUT ONLY accepts Sequence (w,h)
        class_limit : Limit class index from 0 to the value. Set to `None` for no limit.
        num_limit : Limit maximum number of images to use. Only works with `preprocess`.
        resize : tuple (w,h) to resize the image and its mask. Uses Image from PIL to avoid
          artifacts on the mask. Will be ignored if transforms.Resize is defined in transform.
        preprocess : Set true to allow preprocess that filter out all images with mask that
          have lesser than `least_pix` pixels.
        least_pix : filter out thersold of preprocess.
        """

        class_list = get_class_list() # Limit class index within the list. Set to `None` for no limit,
          # It will ignore the `class_limit` parameter.
        valid_split = ["train","val"]
        self.root = root
        if spilt in valid_split:
            try:
                self.coco = COCO(join(root, "annotations", f"instances_{spilt}2017.json"))
            except:
                # annFile doesn't exist, download the annotation file to root.
                print("Annotation file doesn't exists! Downloading")
                makedirs(root,exist_ok=True)
                download_and_extract_archive(coco_ann_url, self.root, filename="annotations_trainval2017.zip")
                print("Annotation file preparation complete")
                self.coco = COCO(join(root, "annotations", f"instances_{spilt}2017.json"))
        else:
            raise Exception(f"Invalid spilt: {spilt}")
        if not exists(self.root):
            mkdir(self.root)
        
        self.num_classes = len(self.coco.getCatIds())
        self.masks = []
        self.resize = resize
        ## Resizing the mask should be handled seperately, or there might be significant artifacts.
        lst_tsf = []
        for i in transform.transforms:
            if i.__class__==transforms.Resize:
                if isinstance(i.size, Sequence):
                    h,w = i.size
                    self.resize=(w,h)
                else:
                    raise TypeError(f"Size in transform should be only sequence. Got {type(i.size)}")
            else:
                lst_tsf.append(i)
        self.transform = transforms.Compose(lst_tsf)
        self.limit_classes = class_limit
        self.num_limit = num_limit
        self.class_list = class_list
        self.no_missing_img = False
        lst = []
        for i in transform.transforms:
            if i.__class__!=transforms.Resize and i.__class__!=transforms.Normalize:
                lst.append(i)
        self.mask_transform = transforms.Compose(lst)
        if preprocess:
            self.ids = []
            self.__preprocess__(list(self.coco.imgs.keys()),least_pix=least_pix,spilt=spilt)
        else:
            self.ids = list(self.coco.imgs.keys())
        self.root = join(root, spilt + "2017")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        image_info = coco.loadImgs(img_id)[0]
        file_Path = join(self.root, image_info['file_name'])
        try:
            image = Image.open(file_Path).convert('RGB') # Read RGB image
        except:
            ## Image doesn't exist, download from coco_url
            ## This process is quite time-consuming and should be avoided later
            if self.no_missing_img:
                print("Failed to read image(s). Download will be performed thereafter, but it can significantly slowdown processing.")
                self.no_missing_img= True
            response = requests.get(image_info["coco_url"])
            if response.status_code == 200:
                with open(file_Path, 'wb') as file:
                    file.write(response.content)
                image = Image.open(file_Path).convert('RGB')
            else:
                raise RuntimeError(f"No image found for {image_info['file_name']} and failed to download")
            
        
        # Create mask
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((image_info["height"], image_info["width"]))
        for ann in anns:
            rle = cocmask.frPyObjects(ann['segmentation'], image_info["height"], image_info["width"])
            m = cocmask.decode(rle)
            cat = ann['category_id']
            if self.class_list==None and self.limit_classes==None:
                c = cat
            elif not self.class_list==None and cat in self.class_list:
                c = self.class_list.index(cat)
            elif not self.limit_classes==None and ann['category_id'] <= self.limit_classes:
                c = cat
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        image = image.resize(size=self.resize, resample=Image.BILINEAR)
        ## Bilinear will cause problem when resizing the Mask, but is commonly used when resizing image
        mask = Image.fromarray(mask).resize(size=self.resize, resample=Image.NEAREST)
        image = self.transform(image)
        mask = self.mask_transform(mask).long()
        return image, mask[0]

    def __preprocess__(self,ids,least_pix:int=1000,spilt='val'): #Filter out bad samples
        coco = self.coco
        ## Test on first image to figure out if dataset itself exists
        first_image_info = coco.loadImgs(ids[0])[0]
        first_file_Path = join(self.root, spilt + "2017", first_image_info['file_name'])
        list_file_path = join(self.root, spilt + "2017.list")
        if not exists(first_file_Path):
            print("Image dataset doesn't exists! Downloading...")
            download_and_extract_archive(coco_img_url.format(spilt), self.root, filename=f"{spilt}2017.zip") ## Download using torchvision download API
            print("Image dataset preparation complete")
        ## Check whether the configuration matches or not.
        no_mismatch = False
        if exists(list_file_path):
            print("List file found, loading...")
            no_mismatch = True
            length = 0
            with open(list_file_path,"r") as f:
                for line in f:
                    if line.startswith("#"):
                        para = line.replace("#","").split(";")
                        ## Check parameter `num_limit`
                        if (int(para[0]) != self.num_limit and self.num_limit!=None) or (int(para[0]) != 0 and self.num_limit==None):
                            no_mismatch = False
                            print("num_limit not matched!")
                        ## Check parameter `class_limit`
                        if (int(para[1]) != self.limit_classes and self.limit_classes!=None) or (int(para[1]) != 0 and self.limit_classes==None):
                            no_mismatch = False
                            print("num_limit not matched!")
                        ## Check `class_list`
                        classes = para[2].split(",")
                        if len(classes)==len(self.class_list):
                            for it in classes:
                                if not (int(it) in self.class_list):
                                    no_mismatch = False
                                    print(f"class_list not matched! None-existing class:{it}")
                        else:
                            no_mismatch = False
                            print("class_list not matched!")
                        ## Check `least_pix`
                        if int(para[3])!=least_pix:
                            no_mismatch = False
                            print("least_pix not matched!")
                        if no_mismatch:
                            print("All configuration matches!")
                        length = int(para[4])
                    else:
                        if no_mismatch:
                            self.ids.append(int(line))
                        else:
                            print("Configuration not matched! Overwrite...")
                            break
                
            if no_mismatch:
                if(length==len(self.ids)):
                    print("Loaded from file:"+list_file_path)
                    return
                else:
                    print(f"Mismatched length({length}) of ids({len(self.ids)}). List file ignored")
                    no_mismatch=False
                    ## Clear self.ids
                    self.ids = []
        print("Perform preprocess from scratch, this will take a while...")
        tbar = tqdm.trange(len(ids))
        for i in tbar:
            if not self.num_limit==None and len(self.ids)+1>self.num_limit:
                print("num_limit exceeded, abort preprocess")
                break
            img_id = ids[i]
            image_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            mask = np.zeros((image_info["height"], image_info["width"]))
            for ann in anns:
                rle = cocmask.frPyObjects(ann['segmentation'], image_info["height"], image_info["width"])
                m = cocmask.decode(rle)
                cat = ann['category_id']
                if self.class_list==None and self.limit_classes==None:
                    c = cat
                elif not self.class_list==None and cat in self.class_list:
                    c = self.class_list.index(cat)
                elif not self.limit_classes==None and ann['category_id'] <= self.limit_classes:
                    c = cat
                else:
                    continue
                if len(m.shape) < 3:
                    mask[:, :] += (mask == 0) * (m * c)
                else:
                    mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
            if (mask > 0).sum() > least_pix:
                self.ids.append(img_id)
            tbar.set_description(f"Accepted {len(self.ids)} of {i+1} images.")
        ## Create a list file to store the information
        print("Creating list file...")
        with open(list_file_path,"w") as f:
            list_str = ""
            if not(self.class_list == None):
                for it in self.class_list:
                    list_str += str(it)+","
                list_str = list_str[:-1]
            f.write(f"#{0 if self.num_limit==None else self.num_limit};{0 if self.limit_classes==None else self.limit_classes};"+list_str+f";{least_pix};{len(self.ids)}\n")
            for it in self.ids:
                f.write(f"{it}\n")
        print("Preprocess Complete!")

    def __len__(self):
        return len(self.ids)