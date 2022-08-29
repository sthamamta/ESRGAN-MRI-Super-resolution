This project is the implementation of ESRGAN for super-resolution of medical images.

In this experiment, we will train rrdbnet for 2005 epochs on patch data from single axis and again train esrgan for 5005 epochs.
The patch size is 96 created with the stride of 65 so that total of 15 patches are create from single image.
Therefore, total dataset size is:
Train set: 213 images* 15 patches = 3,195 patches
Val set: 64 images * 15 patches= 960 pateches
Test set: 7 images * 15 patches = 105 patches


