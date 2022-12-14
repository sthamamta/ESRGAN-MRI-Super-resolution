
import os
import shutil
import time
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from model import Generator
import wandb

def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")


    # data_iter = iter(train_prefetcher)
    # data = data_iter.next()

    # lr = data["lr"]
    # lr =lr.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    # print('the shape of the data is',lr.shape)

    model = build_model()
    print("Build RRDBNet model successfully.")

    wandb.init(
    project= config.project_name,
            name = config.exp_name )

    wandb.watch(model,log="all",log_freq=1)

    # model_output = model(lr)
    # print('model output shape is',model_output.shape)
    # quit();

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        print('inside resume')
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(config.resume,map_location=lambda storage,loc:storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the optimizer scheduler
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    # writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    for epoch in range(start_epoch, config.epochs):
        train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, wandb)
        _, _ = validate(model, valid_prefetcher, epoch, wandb, psnr_model, ssim_model, "Valid")
        psnr, ssim = validate(model, test_prefetcher, epoch, wandb, psnr_model, ssim_model, "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        if epoch % config.save_frequency == 0:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                    os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_last.pth.tar"))


    wandb.unwatch(model)
    wandb.finish()

# def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
def load_dataset():
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "Train")
    valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "Valid")
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False)


    return train_dataloader,valid_dataloader,test_dataloader


def build_model() -> nn.Module:
    model = Generator()
    model = model.to(device=config.device, memory_format=torch.channels_last)

    return model


def define_loss() -> nn.L1Loss:
    pixel_criterion = nn.L1Loss()
    pixel_criterion = pixel_criterion.to(device=config.device, memory_format=torch.channels_last)

    return pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)

    return scheduler


def train(model: nn.Module,
          train_prefetcher,
          pixel_criterion: nn.L1Loss,
          optimizer: optim.Adam,
          epoch: int,
          scaler: amp.GradScaler,
          wandb) -> None:
    """Training main program

    Args:
        model (nn.Module): the generator model in the generative network
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): Calculate the pixel difference between real and fake samples
        optimizer (optim.Adam): optimizer for optimizing generator models in generative networks
        epoch (int): number of training epochs during training the generative network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0


    # Get the initialization training time
    end = time.time()

    for batch_index,batch_data in enumerate(train_prefetcher):
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)

            loss = pixel_criterion(sr, hr)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()


        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            # Record loss during training and output to file
            # writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            wandb.log({"Train/Loss" : loss.item() })
            progress.display(batch_index)



def validate(model: nn.Module,
             data_prefetcher,
             epoch: int,
             wandb,
             psnr_model: nn.Module,
             ssim_model: nn.Module,
             mode: str):
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        psnr_model (nn.Module): The model used to calculate the PSNR function
        ssim_model (nn.Module): The model used to compute the SSIM function
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    model.eval()


    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        for batch_index,batch_data in enumerate(data_prefetcher):
            # Transfer the in-memory data to the CUDA device to speed up the test
            lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)


    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        # writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        # writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
        wandb.log({"{mode}/PSNR" : psnres.avg,
            "{mode}/SSIM":ssimes.avg,
            })
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
