
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Image magnification factor
upscale_factor = 2
# Current configuration parameter method
# mode = "train_rrdbnet"
# mode = "train_esrgan"
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "RRDBNet"



if mode == "train_rrdbnet":
    # Dataset address
    train_image_dir = "./data/MRI/ESRGAN_MRI/train"
    valid_image_dir = "./data/MRI/ESRGAN_MRI/valid"
    # test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
    test_lr_image_dir = f"./data/MRI_test/lr_{upscale_factor}"
    test_hr_image_dir = f"./data/MRI_test/hr"

    image_size = 96
    batch_size = 32
    num_workers = 4

    # Incremental training and migration training
    # resume = "samples/RRDBNet_x4-DFO2K-2e2a91f4.pth.tar"
    resume = None

    # Total num epochs
    epochs = 2005

    project_name='ESRGAN_Pytorch'
    exp_name='RRDBNET'

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5
    only_test_y_channel = False

    # How many iterations to print the training result
    print_frequency = 500  #batch id to print
    save_frequency = 500 # epoch no to save

if mode == "train_esrgan":
    # Dataset address
    train_image_dir = "./data/MRI/ESRGAN_MRI/train"
    valid_image_dir = "./data/MRI/ESRGAN_MRI/valid"
    # test_lr_image_dir = f"./data/Set5/LRbicx{upscale_factor}"
    test_lr_image_dir = f"./data/MRI_test/lr_{upscale_factor}"
    test_hr_image_dir = f"./data/MRI_test/hr"

    image_size = 96
    batch_size = 32
    num_workers = 4

    # Incremental training and migration training
    resume = "./results/RRDBNET/g_last.pth.tar"
    resume_d = ""
    resume_g = ""

    project_name='ESRGAN_Pytorch'
    exp_name='ESRGAN'
    only_test_y_channel = False
    # Total num epochs
    epochs = 5005

    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.34"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    adversarial_weight = 0.005

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    print_frequency = 200

    save_frequency = 500 # epoch no to save

if mode == "test":
    # Test data address
    lr_dir = f"./data/MRI_test/lr_{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"  #directory provided to save the output of the model
    hr_dir = f"./data/MRI_test/hr"
    only_test_y_channel = False
    model_path = "./results/ESRGAN/g_best.pth.tar"
