import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

training_dataset = 'C:\\Users\\user\\PycharmProjects\\ml_Assignment3\\train'
test_dataset = 'C:\\Users\\user\\PycharmProjects\\ml_Assignment3\\test'
img = []
data = ['accordian', 'dollar_bill', 'motorbike', 'soccer_ball']
train_data = []
train_labels = []

def Hog(image_Location):
    image = cv2.imread(image_Location)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (128, 64))
    fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd, hog_image

for item in data:
    for img_name in os.listdir(os.path.join(training_dataset, item)):
        img_loc = os.path.join(training_dataset, item, img_name)
        ext_features, hog_img = Hog(img_loc)
        train_data.append(ext_features)
        train_labels.append(item)

#Plot all train set
        
fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(20, 10))
for i in range(4):
    for j in range(14):
        index = i * 5 + j
        image_name = os.listdir(os.path.join(training_dataset, data[i]))[j]
        image_path = os.path.join(training_dataset, data[i], image_name)
        features, hog_image = Hog(image_path)
        axs[i,j].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        axs[i,j].set_title(data[i])
        axs[i,j].axis('off')
plt.tight_layout()
plt.show()


#Plott all hog images of train set

fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(20, 10))
for i in range(4):
    for j in range(14):
        index = i * 5 + j
        image_name = os.listdir(os.path.join(training_dataset, categories[i]))[j]
        image_path = os.path.join(training_dataset, data[i], image_name)
        features, hog_image = Hog(image_path)
        axs[i,j].imshow(hog_image)
        axs[i,j].set_title(data[i])
        axs[i,j].axis('off')
plt.tight_layout()
plt.show()

test_data = []
test_labels = []
for item in data:
    for img_name in os.listdir(os.path.join(test_dataset, item)):
        img_path = os.path.join(test_dataset, item, img_name)
        ext_features, hog_img = Hog(img_path)
        test_data.append(ext_features)
        test_labels.append(item)
        img.append(cv2.imread(img_path))


linear_svm_model = svm.SVC(kernel='linear', C=1)
linear_svm_model.fit(train_data, train_labels)
pred_labels_train_linear = linear_svm_model.predict(train_data)
accuracy_train_linear = accuracy_score(train_labels, pred_labels_train_linear)
print("-----------------------------------")
print("Linear SVM Train Accuracy:", accuracy_train_linear)
print("-----------------------------------")

pred_labels_test_linear = linear_svm_model.predict(test_data)
accuracy_test_linear = accuracy_score(test_labels, pred_labels_test_linear)
print("Linear SVM Test Accuracy:", accuracy_test_linear)
print("-----------------------------------")
print("-----------------------------------")


poly_svm_model = svm.SVC(kernel='poly', degree=2, C=1)
poly_svm_model.fit(train_data, train_labels)
pred_labels_train_poly = poly_svm_model.predict(train_data)
accuracy_train_poly = accuracy_score(train_labels, pred_labels_train_poly)
print("Polynomial SVM Train Accuracy:", accuracy_train_poly)

pred_labels_test_poly = poly_svm_model.predict(test_data)
accuracy_test_poly = accuracy_score(test_labels, pred_labels_test_poly)
print("-----------------------------------")
print("Polynomial SVM Test Accuracy:", accuracy_test_poly)
print("-----------------------------------")

print("-----------------------------------")
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(train_data, train_labels)
pred_labels_train_rbf = svm_model.predict(train_data)
accuracy_train = accuracy_score(train_labels, pred_labels_train_rbf)
print("RBF Train Accuracy:", accuracy_train)

pred_labels_test_rbf = svm_model.predict(test_data)
accuracy_test = accuracy_score(test_labels, pred_labels_test_rbf)
print("-----------------------------------")
print("RBF Test Accuracy:", accuracy_test)
print("-----------------------------------")

# plot of the prediction
fig,axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 9))
for i, axis in enumerate(axs.flat):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
    pred = pred_labels_test_linear[i]
    actual = test_labels[i]
    if (pred == actual):
        axis.set_title(actual, color='green')
    else:
        axis.set_title("Predicted: {} \n Actual: ".format(pred, actual), color='red')
plt.show()