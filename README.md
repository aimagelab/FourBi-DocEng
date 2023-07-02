# FourBi

We provide the checkpoint for the model:
 - [Mirror 1](https://github.com/aimagelab/FourBi-DocEng/releases/download/checkpoint/2616.pth)
 - [Mirror 2](https://drive.google.com/file/d/1qv5f8bC5c73ud2zmIK8eSWYSDRpz9EWP/view?usp=sharing)

## Setup
To run this project, we used `python 3.8.3` and `pytorch 2.0` 
```bash
# Windows
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
# Linux
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

## Binarize
To run the model on a folder with images, run the following command:
```
python binarize.py <path to checkpoint> --src <path to the test images folder> 
--dst <path to the output folder> --batch_size <batch size>
```
The default batch size is 4. Consider increasing the batch size to speed up the binarization process.

To run the model on the CPU, just change the `--device` argument (default `cuda`)
```
python binarize.py <path to checkpoint> --src <path to the test images folder> 
--dst <path to the output folder> --batch_size <batch size> --device cpu
```

