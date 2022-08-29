This project is the implementation of ESRGAN for super-resolution of medical images.

In this experiment, we will train rrdbnet for 2005 epochs on patch data from z-axis and again train esrgan for 5005 epoch.
The patch size is 96 create with the stride value 65 so that total of 15 patches are create from single image.
therefore, total dataset size is:
train: 213 images* 15 patches = 3,195 patches
val: 64 images * 15 patches= 960 pateches
test: 7 images * 15 patches = 105 patches


