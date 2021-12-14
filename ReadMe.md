# CS 194-26: Final Proposed Project

### Semantic Segmentation to SPADE Pipeline

Website URL: https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/projFinalProposed/cs194-26-abe \\
Personal Website URL: https://normankarr.com/computational-photography/final-project/

Code overview:\\

Code regarding semantic segmentation model and training is under the segmentation folder \\ 
Pretrained segmentation models should be placed in a folder named "models" under the segmentation folder \\

Code regarding SPADE is all under the SPADE folder. Most of this code is pulled straight from the SPADE repository: https://github.com/NVlabs/SPADE \\

The majority of code used for this project is in the Google Colab notebook. The cityscapes datasets are expected to placed under a datasets folder in the corresponding drive. If this is done, then the notebook should be able to just be run all the way through without error until the SPADE visualization. Once at the SPADE visualization, the output contained in results.zip has to be reconfigured to have the same directory structure as Cityscapes.