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
- [GeneratorFeatures](code/Generator_feature_extractor.py)
- [Generator](code/Generator.py)
- [GeneratorBroken63](code/Generator_break_63_plus_1.py)
- [GeneratorRRDB](code/Generator_RRDB.py)<br/>
**For each Generator one can use regular GAN or Conditional GAN(example downwards)**

## Command line 
Input arguments to [train.py](train.py):
```
--epochs                | default=21   
--saveparams_freq_batch | default=5    
--saveimg_freq_step     | default=100  
--lrG                   | default=1e-4 
--lrD                   | default=1e-5 
--train_path            | default='./data/6k_data/train'
--test_path             | default='./data/6k_data/test'
--type_of_dataset       | default="10_dogs"
--fname                 | default=""
--generator             | default="GeneratorFeatures"
--discriminator         | default= "Discriminator"
--batch_size            | default= 16,type=int
```
- command line for our default model(regular GAN - GeneratorFeatures)
```
python train.py
```
- command line for different model(Conditional GAN - GeneratorBroken63) 
```
python train.py --generator GeneratorBroken63 --discriminator C
```
