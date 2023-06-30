# FourBi

We provide the checkpoint for the [model](https://drive.google.com/file/d/1qv5f8bC5c73ud2zmIK8eSWYSDRpz9EWP/view?usp=sharing)
 
To run the model on a folder with images, run the following command:
```
python binarize.py <path to checkpoint> --src <path to the test images folder> 
--dst <path to the output folder> --batch_size <batch size>
```
The default batch size is 4