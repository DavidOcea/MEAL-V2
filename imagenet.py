"""Dataset class for loading imagenet data."""

import os

from torch.utils import data as data_utils
from torchvision import datasets as torch_datasets
from torchvision import transforms
from folder import FileListLabeledDataset
from transforms import RandomResizedCrop, Compose, Resize, CenterCrop, ToTensor, \
    Normalize, RandomHorizontalFlip, ColorJitter, Lighting

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_data_list = '/data/dingrui/code/poseidon/jh_cls2_train.txt'
train_data_root = '/data/dingrui/data/JH_cls3_20191030'
val_data_list = '/data/dingrui/code/poseidon/jh_cls2_test.txt'
val_data_root = '/data/dingrui/data/JH_cls3_20191030'
normalize = Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])

def get_train_loader(imagenet_path, batch_size, num_workers, image_size):
    # train_dataset = ImageNet(imagenet_path, image_size, is_train=True)
    train_dataset = FileListLabeledDataset(train_data_list, train_data_root,
                                            Compose([
                                                RandomResizedCrop(224,
                                                                scale=(0.5, 1.2),  
                                                                ratio=(0.75, 1.25)),
                                                RandomHorizontalFlip(),
                                                ColorJitter(brightness=0.4, contrast=0.,
                                                            saturation=0.4, hue=0.4),
                                                ToTensor(),
                                                Lighting(1, [0.2175, 0.0188, 0.0045], [[-0.5675,  0.7192,  0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948,  0.4203]]), 
                                                normalize,
                                            ]))
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)


def get_val_loader(imagenet_path, batch_size, num_workers, image_size):
    # val_dataset = ImageNet(imagenet_path, image_size, is_train=False)
    val_dataset = FileListLabeledDataset(val_data_list,val_data_root,
                                        Compose([
                                            Resize([256,256]),
                                            CenterCrop(224),
                                            ToTensor(),
                                            normalize
                                        ]))
    return data_utils.DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)


class ImageNet(torch_datasets.ImageFolder):
    """Dataset class for ImageNet dataset.

    Arguments:
        root_dir (str): Path to the dataset root directory, which must contain
            train/ and val/ directories.
        is_train (bool): Whether to read training or validation images.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, im_size, is_train):
        if is_train:
            root_dir = os.path.join(root_dir, 'train')
            transform = transforms.Compose([
                transforms.RandomResizedCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        else:
            root_dir = os.path.join(root_dir, 'test') #val
            transform = transforms.Compose([
                transforms.Resize(int(256/224*im_size)),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        super().__init__(root_dir, transform=transform)
