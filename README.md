# HOG-SVM-ImageClassifier
### The HOG-SVM Image Classifier is a method employed for image recognition and object detection. To build a model, one gathers a dataset and uses HOG (Histogram of Oriented Gradients) to extract features from the images. These features are then input into the SVM (Support Vector Machine) algorithm to train the model. Once trained, the model can classify new images. The evaluation process involves using a separate set of images to test the model's accuracy. The HOG-SVM Image Classifier is an effective technique for accurately identifying objects in various contexts.



# Workflow
+ We have 2 dataset training and testing datasets.
+ First we need to go the path of the training images
+ Get each image & Extract features from it using Histogram of Oriented Gradients
Map each image to its class. Folder name is class of the image
Now we have features (extracted using HOG) and target for each image.
Create SVM model & train it using training dataset we created.
Create Testing dataset in the same way you create training dataset.
Finally test the model using testing dataset
