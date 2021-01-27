# CSRGAN

Official pyTorch implementaion of the paper [CSRGAN - Image Super-Resolution and Colorization in a single Generative Adversarial Network](PDFS/CSRGAN.pdf)

## Requirments 

```
conda env create -f requirments.yml
```

## Dataset 
In this paper we used [Stanford University dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)<b
This dataset contains:  
Number of dog breeds: 120
Number of images: 20,580
**The images are in different sizes thus we used resize in the preprocess of the data**

We used small portion of this dataset ~30 different dog breeds ~6K images. 
We splited our dataset to train and test with respect to the different dog breeds.
Train: ~5400 images
Test: ~560 images
