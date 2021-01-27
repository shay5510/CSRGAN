# CSRGAN

Official pyTorch implementaion of the paper [CSRGAN - Image Super-Resolution and Colorization in a single Generative Adversarial Network](PDFS/CSRGAN.pdf)

## Requirments 

```
conda env create -f requirments.yml
```

## Dataset 
In this paper we used [Stanford University dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)<br/>
This dataset contains:<br/>
Number of dog breeds: 120<br/>
Number of images: 20,580<br/>
**The images are in different sizes thus we add resize(256X256) to the preprocess**<br/><br/>
We used small portion of this dataset ~30 different dog breeds ~6K images.<br/> 
We splited our dataset to train and test with respect to the different dog breeds.<br/>
Train: ~5400 images<br/>
Test: ~560 images<br/>

## Training Details
We used google cloud services for training our model - using:<br/>
- 1 GPU - Tesla K80 
- 2 CPU - 13GB RAM

We trained our mode for 21 epochs, approximately 8hr.<br/>

## Training Models 
- [Generator_feature_extractor](code/Generator_feature_extractor.py)
- [Generator](code/Generator.py)
- [Generator_break_63_plus_1](code/Generator_break_63_plus_1.py)
- [Generator_RRDB](code/Generator_RRDB.py)

