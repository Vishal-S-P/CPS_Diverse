# Source code adapted from - https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py
import torch
from PIL import Image
# import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import torchvision
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from functools import partial
from torch.utils.data import Subset
from .celeba import CelebA
from .lsun import LSUN
from .utils_imagenet import ClassImageFolder

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )
    
def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    print("Number of files found : ", len(all_files))
    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def get_val_dataset(dataset_pth, config):
    
    if config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(dataset_pth),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(dataset_pth),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(dataset_pth),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "CelebA_HQ" or config.data.dataset == 'FFHQ' or config.data.dataset == 'AFHQ':
        if config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(dataset_pth),
                transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(dataset_pth),#os.path.join(args.exp, "datasets", "celeba_hq"),
                transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                              transforms.ToTensor()])
            )
            num_items = len(dataset)
            print("Number of images :", num_items)

            test_dataset = dataset
            # indices = list(range(num_items))
            # random_state = np.random.get_state()
            # np.random.seed(2019)
            # np.random.shuffle(indices)
            # np.random.set_state(random_state)
            # train_indices, test_indices = (
            #     indices[: int(num_items * 0.)],
            #     indices[int(num_items * 0.) :],
            # )
            # test_dataset = Subset(dataset, test_indices)

    elif config.data.dataset == 'ImageNet':
        # only use validation dataset here
        
        if config.data.subset_1k:
            from datasets.imagenet_subset import ImageDataset
            dataset = ImageDataset(os.path.join(dataset_pth),
                     os.path.join(dataset_pth, 'imagenet_val_1k.txt'),
                     image_size=config.data.image_size,
                     normalize=False)
            test_dataset = dataset
        elif config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(dataset_pth, 'datasets', 'ood'),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            import pickle
            with open(os.path.join(dataset_pth,'class_to_idx.pkl'), 'rb') as f:
                class_to_idx = pickle.load(f)
            classes = list(class_to_idx.keys())
            dataset = ClassImageFolder(
                os.path.join(dataset_pth),#os.path.join(args.exp, "datasets", "celeba_hq"),
                transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                              transforms.ToTensor()]),
                allow_empty=True,
                classes=classes,
                class_to_idx=class_to_idx,
            )
            num_items = len(dataset)
            # dataset = torchvision.datasets.ImageNet(
            #     os.path.join(dataset_pth, 'datasets', 'imagenet'), split='val',
            #     transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
            #     transforms.ToTensor()])
            # )
            test_dataset = dataset

    elif config.data.dataset == 'CIFAR10':
        # only use validation dataset here
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(dataset_pth),
            transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                          transforms.ToTensor()])
        )
        num_items = len(dataset)
        test_dataset = dataset

    elif config.data.dataset == "lsun":
        if config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(dataset_pth, 'datasets', "ood_{}".format(config.data.category)),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            train_folder = "{}_train".format(config.data.category)
            val_folder = "{}_val".format(config.data.category)
            test_dataset = LSUN(
                root=dataset_pth,#os.path.join(dataset_pth, "datasets", "lsun"),
                classes=[val_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                )
            )
            dataset = test_dataset

    elif config.data.dataset == "lsun_cat":
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(dataset_pth),
            transform=transforms.Compose([transforms.CenterCrop([config.data.image_size, config.data.image_size]),
                                          transforms.ToTensor()])
        )
        test_dataset = dataset

    else:
        raise NotImplementedError("Unknown Dataset....")
        

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(X):
    X = 2 * X - 1.0
    return X

def inverse_data_transform(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)