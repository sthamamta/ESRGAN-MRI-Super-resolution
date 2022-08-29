
"""Realize the function of dataset preparation."""
import os
# import queue
# import threading
from xmlrpc.client import Boolean

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, upscale_factor: int, mode: str,mri=True) -> None:
        super(TrainValidImageDataset, self).__init__()
        # Get all image file names in folder
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor
        # Load training dataset or test dataset
        self.mode = mode
        self.mri = mri

    def __getitem__(self, batch_index: int):
        # Read a batch of image data
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "Train":
            hr_image = imgproc.random_crop(image, self.image_size)
            hr_image = imgproc.random_rotate(hr_image, angles=[0, 90, 180, 270])
            hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
            hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)
        elif self.mode == "Valid":
            hr_image = imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)
        lr_image = cv2.resize(lr_image, (self.image_size,self.image_size), interpolation = cv2.INTER_LINEAR)
        # if self.mri:
        #      # BGR convert to RGB
        #     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY)
        #     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
        # else:
        #     # BGR convert to RGB
        #     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        #     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)
        
        # print(lr_tensor.shape)
        # print(hr_tensor.shape)
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_lr_image_dir (str): Test dataset address for low resolution image dir.
        test_hr_image_dir (str): Test dataset address for high resolution image dir.
        mri(Boolean): if true convert the image bgrtogray else bgrtorgb
    """

    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str,mri=True) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.lr_image_file_names = [os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir)]
        self.hr_image_file_names = [os.path.join(test_hr_image_dir, x) for x in os.listdir(test_hr_image_dir)]
        self.mri = mri

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        lr_image = cv2.imread(self.lr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        hr_image = cv2.imread(self.hr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        

        # if self.mri:
        #      # BGR convert to RGB
        #     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY)
        #     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
        # else:
        #     # BGR convert to RGB
        #     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        #     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)


        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)


# class PrefetchGenerator(threading.Thread):
#     """A fast data prefetch generator.

#     Args:
#         generator: Data generator.
#         num_data_prefetch_queue (int): How many early data load queues.
#     """

#     def __init__(self, generator, num_data_prefetch_queue: int) -> None:
#         threading.Thread.__init__(self)
#         self.queue = queue.Queue(num_data_prefetch_queue)
#         self.generator = generator
#         self.daemon = True
#         self.start()

#     def run(self) -> None:
#         for item in self.generator:
#             self.queue.put(item)
#         self.queue.put(None)

#     def __next__(self):
#         next_item = self.queue.get()
#         if next_item is None:
#             raise StopIteration
#         return next_item

#     def __iter__(self):
#         return self


# class PrefetchDataLoader(DataLoader):
#     """A fast data prefetch dataloader.

#     Args:
#         num_data_prefetch_queue (int): How many early data load queues.
#         kwargs (dict): Other extended parameters.
#     """

#     def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
#         self.num_data_prefetch_queue = num_data_prefetch_queue
#         super(PrefetchDataLoader, self).__init__(**kwargs)

#     def __iter__(self):
#         return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


# class CPUPrefetcher:
#     """Use the CPU side to accelerate data reading.

#     Args:
#         dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
#     """

#     def __init__(self, dataloader: DataLoader) -> None:
#         self.original_dataloader = dataloader
#         self.data = iter(dataloader)

#     def next(self):
#         try:
#             return next(self.data)
#         except StopIteration:
#             return None

#     def reset(self):
#         self.data = iter(self.original_dataloader)

#     def __len__(self) -> int:
#         return len(self.original_dataloader)


# class CUDAPrefetcher:
#     """Use the CUDA side to accelerate data reading.

#     Args:
#         dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
#         device (torch.device): Specify running device.
#     """

#     def __init__(self, dataloader: DataLoader, device: torch.device):
#         self.batch_data = None
#         self.original_dataloader = dataloader
#         self.device = device

#         self.data = iter(dataloader)
#         self.stream = torch.cuda.Stream()
#         self.preload()

#     def preload(self):
#         try:
#             self.batch_data = next(self.data)
#         except StopIteration:
#             self.batch_data = None
#             return None

#         with torch.cuda.stream(self.stream):
#             for k, v in self.batch_data.items():
#                 if torch.is_tensor(v):
#                     self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         batch_data = self.batch_data
#         self.preload()
#         return batch_data

#     def reset(self):
#         self.data = iter(self.original_dataloader)
#         self.preload()

#     def __len__(self) -> int:
#         return len(self.original_dataloader)
