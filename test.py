
"""File description: Realize the verification function after model training."""
import os

import cv2
import torch
from natsort import natsorted

import config
import imgproc
from image_quality_assessment import PSNR, SSIM
from model import Generator


def main() -> None:
    # Initialize the super-resolution model
    model = Generator().to(device=config.device, memory_format=torch.channels_last)
    print("Build ESRGAN model successfully.")

    # print(config);quit();

    # print(model)
    # rand_tensor = torch.randn(1,3,128,128).to('cuda')
    # output_tensor = model(rand_tensor)
    # print('input shape:', rand_tensor.shape)
    # print('output shape: ',output_tensor.shape)
    # import sys
    # sys.exit()
    
    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load ESRGAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image and HR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)  # lr_image shape is (360,256)
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED)

        # Convert BGR channel image format data to RGB channel image format data
        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY)
        # hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)

        print('before image 2 tensor')
        print('range of lr image:', lr_image.min(),lr_image.max())
        print('range of hr image:', hr_image.min(),hr_image.max())
        print('shape of image',lr_image.shape)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=True).unsqueeze_(0)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=True).unsqueeze_(0)

        # print('after image 2 tensor')
        # print('range of lr image:', lr_tensor.min(),lr_tensor.max())
        # print('range of hr image:', hr_tensor.min(),hr_tensor.max())
        # print('shape of tensor',hr_tensor.shape)

        # Transfer Tensor channel image format data to CUDA device
        lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr_tensor = hr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)


        # quit();
        # lr_tensor = lr_tensor.unsqueeze(0)
        # hr_tensor = hr_tensor.unsqueeze(0)
        print('lr shape',lr_tensor.shape)
        print('hr shape',hr_tensor.shape)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = model(lr_tensor)


        print('range of sr tensor', sr_tensor.min(),sr_tensor.max())
        # Save image
        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_tensor, hr_tensor).item()
        ssim_metrics += ssim(sr_tensor, hr_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} dB\n"
          f"SSIM: {avg_ssim:4.4f} u")


if __name__ == "__main__":
    main()
