# Require
* Tensorflow >= 2.3.1
* Keras 
# Structure
` data_processing.py ` : load face and label from folder dataset and load trio of images (anchor-positive-nagative)  
` training.py ` : creat model with triplet loss and training model with input is trio of images  
` test.py ` : file test model  
` main.py ` : if you want you can using it, I have built the necessary functionality
` Inception_RestnetV1.py ` : network Inception ResNet V1
# Dataset
You must create a folder that contains human face folders:  At least there must be three people  
**Example**  
*train_dataset*  
Kevin  
Peter  
Alisa  
Jackson  
Williams  
Hanks  
Messi  
...  
Each person has a separate folder containing their face photo.    
**You can use MTCNN to crop faces.**
