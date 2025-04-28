# cs760-project-1--covid-19-diagnosis-based-on-ct-scans-solved
**TO GET THIS SOLUTION VISIT:** [CS760 Project 1- COVID-19 Diagnosis based on CT Scans Solved](https://www.ankitcodinghub.com/product/cs-760-machine-learning-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;117462&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS760 Project 1- COVID-19 Diagnosis based on CT Scans Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
COVID-19 Diagnosis based on CT Scans

We applied several machine learning methods on the scanned CT chest images to explore the. relationship between positiveness of COVID-19 and chest CT.

We considered several competitive machine learning approaches, including Logistic Regression, K Nearest Neighbors, Naive Bayes, Decision Tree, Random Forest, Support Vector Machine and Convolutional Neural Network. We trained these models with raw image data and make predictions. We implemented the transfer learning on our own dataset. We used ResNet-18 architecture and personalized our own fully connection layer at the end. We froze the parameters in previous layers] only trained the personalized layers. In our result, we have SVM achieve the best performance with test accuracy 89%. The ResNet-18 has the second high test accuracy of 81.33%.

To combine the feature extraction advantage of convolutional neural networks and the interpretation of other machine learning methods, we used simple CNN to process images and transfer them into vectors representing the image characteristics. Then we applied several machine learning models to the image features. The advantage of this idea is instead of using raw image pixel value for classification, we implemented our machine learning algorithms on feature vectors extracted by pre-trained Convolutional Neural Network layers. This can reduce the feature dimension significantly, hence reduce the time and space complexity, free up more resources for more analysis. We aim to improve the accuracy of prediction and explore the explanation and interpretation as well.

We make use of the high accuracy from neural networks along with the interpretation from other machine learning models to better help make decisions and explain the relationship between COVID results and scanned images from a different perspective. We hope our model provides some insights into potential applications of machine learning methods in this specific COVID-19 situation.

1 Introduction

1

(a) The CT scan of a (b) The CT scan of a (c) The CT scan of a (d) The CT scan of a COVID-19 patient. COVID-19 patient. healthy person. healthy person.

Figure 1: Examples of CT scans from COVID-CT dataset.

This work is to develop a classifier for accurate diagnosis of COVID-19 based on the CT scans. We train the classifier based on three popular machine learning methods: random forest, SVM and Neural network. Since neural network is the most commonly used method for image processing, we train two Neural network models: simple convolutional neural network and ResNet-18 to perform COVID-19 CT classification. Out of consideration of computation complexity, we use pre-trained ResNet-18 to improve the efficiency of ResNet18. Finally, we perform numerical evaluations of the selected methods on the CT dataset, comparing and contrasting their performance of both accuracy and efficiency.

The highest accuracy we obtain is 89%, which is achieved by SVM classifier. The accuracy of our SVM classifier is higher than the baseline method self-Tran+DenseNet-169 which is proposed by He et al.[3].

Contributions of this paper are summarized as follows. The rest of the paper is organized as follows. Section 2 discusses the related work on COVID-19 diagnosis based on CT scans. Section 3 introduce the dataset we analyze in details. We present the basic ideas of the selected methods Section 4, describe the experiment setting, report and discuss the empirical results in Section 5. We summarize the conclusions of this work, and discuss interesting future follow-up work in Section 6. The implementation details are provided in Appendix.

2 Related work

As coronavirus caused a global epidemic problem that spread quickly, there has been a fair amount of work in diagnosis of COVID-19[3, 4, 5, 6, 7, 1]. Diagnosis of COVID-19 is typically associated with symptoms of pneumonia, computed tomography (CT) scans[3, 7], and chest X-ray[4, 5, 6]. Our work is closely related to He et al.[3], Xu et al.[1] and Singh et al.[7] which also used CT scans to perform diagnosis of COVID-19. Specifically, we use the same dataset[8] as He et al.[3].

He et al.[3] combined many popular models in deep learning with transfer learning to train the data. He et al. proposed Self-Trans, a self-supervised transfer learning approach where contrastive self-supervised learning is integrated into transfer learning process to adjust the network weights pretrained on source data. The highest accuracy 86% is achieved by combining Self-Tran and DenseNet-160.

Xu et al.[1] evaluated two convolutional neural networks (CNN) three-dimensional classification models. Ont was ResNet-based network and another model was based on ResNet-based network structure by concatenating the location attention mechanism in the full-connection layer. It achieves an overall accuracy of 86.7%.

Singh et al.[7] used a CNN whose initial parameters of CNN are tuned using multiobjective differential evolution (MODE). The highest accuracy of the proposed approach achieves above 92%.

Ardakani et al.[9] implemented several well-known CNN architectures such as AlexNet, VGG, ResNet to predict on CT scans. The best performance is achieved by ResNet101. They demonstrated in their study that the ResNet-101 can be considered as a promising model to characterize and diagnose COVID-19 infections.

In contrast, our work is not restricted to deep learning methods, but also consider other competitive machine learning approaches. We consider other benchmark machine learning methods: Random Forest and Support Vector Machine. We try to improved the accuracy of prediction and find explanation and interpretaion as well. We compare these methods in the Section 5.

3 Dataset

Data provided by Zhao et al.[8] has 746 CT images, containing clinical findings of COVID-19 from 216 patients. The images in the dataset consist of 349 CT scans that are positive for COVID-19 and 397 CT scans are negative for COVID-19. Figure 1 contains some sample images from the dataset. The size of images are different. The minimum width is 115 pixels, and the minimum height is 61 pixels.

4 Approach

In this section, we will describe which approaches we utilize to train our data and make classification.

4.1 Random forest

Random forests or random decision forests are one of the most widely used machine learning algorithms for regression and classification. Random forests are essentially a bootstrap, or bagging of decision trees.

Suppose the data matrix X . A decision tree is a flowchart-like tree structure where the branch represents a decision rule, and each leaf node represents the outcome. A decision tree makes predictions based on series of questions. In practice, we train the decision tree by choosing the most informative feature via mutual information or other criterion in each internal node, and split the data according to this feature.

However, decision trees is sensitive to the data. Thus they can be quite biased, and tend to overfit. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Random forests generally outperform decision trees.

4.2 Neural network

The idea of neural networks comes from the biological brains and neurons. Each neuron receives information from several previous neurons. In the human body, eyes and ears receive information, then the neurons transfer the information to the brain to make decisions. A node without an activation function is known as a perceptron.

One layer consists of multiple nodes. A standard neural network contains several layers: one input layer, one output layer, and several hidden layers. The input layer receives the feature vectors from each observation, while the output layer outputs the prediction probability of each class of label, and the hidden layer contains the intermediate results. The number of nodes in the input is the number of training samples in each batch, the number of nodes in the output layer is the number of labels, the number of nodes in hidden layers and the number of hidden layers is determined by users based on the empirical performance. During the training period, the weights and bias are updated to minimize the selected objective function such as cross entropy via stochastic gradient descent.

A convolutional neural network adds several other layers especially for image processing. An image contains pixels represented by value. A convolutional layer can extract certain features in an image, such as eyes, mouth, ears of a dog. The word convolution comes from the functional analysis, it has an integral expression that can be used for computing the sum of two random variables. The convolution in CNN conducts the similiar steps, it has several feature detectors, each detector walk through the image and compute the sum of the product of values stored in corresponding pixels.

We train the CT datasets with two CNN models: simple CNN and ResNet.

4.2.1 Simple CNN

We build one simple CNN model with three convolutional layers to train the data. To avoid the large parameter dominate the updates, we use the batch normalization in two convolutional layers. The maxpooling layers are introduced to help with local invariance. They introduce no parameters. The architecture of this model is shown as Figure 2.

Figure 2: Architecture of the simple CNN we trained.

4.2.2 Transfer Learning + ResNets

4.3 Support vector machines (SVMs)

Let X denote the sample matrix, where x1,…,xn are individual samples. y =

(y1,…,yn) are the labels, where yi ∈ {0,1} for i = 1,…,n. The training of a SVM classifier involves finding a hyperplane

Hβ,β0 = {i ∈ {1,…,n} : yi(hxi,βi + β0) ≥ 1}, kβk &lt; m,

to separate the training samples from two different class with the largest margin m [12]. To maximize the margin m:

m = min dist(xi,H(β˜,β˜0)).

i=1,…,n

However, the requirement that the data is linear separable have been too strong, leading to many possible relaxations, e.g. soft-margin SVM. Soft-margin SVM state a preference for margins that classify the training data correctly, but soften the constraints to allow for non-separable data with a penalty (ε1,…,εn) ∈ Rn proportional to the amount by which the example is misclassified. Thus the optimization problem becomes: s.t.

In practice, C is determined by cross validation.

A more advanced tool is kernel, which transforms the data to be linear separable. Suppose the transformation ψ : Rd → Hilbert space H transforms the data x1,…,xn to linear separable data ψ(x1),…,ψ(xn), then the primal optimization problem becomes:

s.t.

The dual problem is

s.t. i = 1,…,n.

With kernel K = hψ(·),ψ(·)iH, the dual problem can be represented as

s.t. i = 1,…,n.

Actually, given K, ψ is unnecessary for us to solve the dual problem. In practice, we select the kernel K and then solve the dual problem to get SVM classifier.

4.4 Simple CNN as feature extractor

We also consider using simple CNN to process images and transfer them into vectors representing the image characteristics, then we use other machine learning methods in previous sections on the feature vectors. The main idea is similar to transfer learning, the convolutional layer is suitable for extracting features in images. Ideally, applying previous machine learning methods on these vectors can give higher accuracy and better interpretation than applying them on raw images.

We use the CNN architecture in 4.2.1 as feature extractor, we add one layer with 10 nodes right before the output layer. Once we finish the training, we feed all images to the architecture and get the intermediate result outputted by the second to last layer. Then we have the new dataset with each observation has one corresponding feature vector.

5 Experiments

5.1 Main result on raw images

We resize all images to size 115×61, and divide them into training set and testing set at a ratio of 9:1. The packages we use are listed below:

• file reading: pandas, skimage, shutil, os;

• data processing: pandas, numpy, random, sklearn, PIL;

• benchmarking: time;

• plot: matplotlib;

• classifer: sklearn, torch, torchvision.

We provide more details of how we use these packages in Appendix. We train the methods we introduced in last section, and the settings are described below.

Random Forest: We flatten the image arrays to vectors of length 7015. Our random forests generate 100 decision trees, whose sample size is the same as the original sample size but the samples are drawn with replacement. Our criteria of choosing features in each internal node is Gini impurity. Gini impurity is also known as the total decrease in node impurity. This is how much the model fit or accuracy decreases when you drop a variable. The test accuracy of our random forest is around 80%.

Simple CNN: The learning rate, batch size and number of epochs we select empirically are 0.05, 8 and 20, respectively. The objective function we use here is cross entropy. All the activation functions except the output layer are ReLU. The test accuracy we obtain from simple CNN is 70.67%.

Transfer Learning+ResNet: We use a pre-trained ResNet-18[13] and only train the last hidden layer with activation function ReLu and output layer. To avoid overfitting, we drop feature maps with a probability of 0.5. The accuracy of the training set and test set are 82.116% and 81.33%, respectively, indicating that there is almost no overfitting.

Support Vector Machine: We flatten the image matrices to vectors of length 115 × 61 = 7015. We use 3-fold cross-validated grid search to determine the kernel from two alternative kernels, radial basis function kernel exp and linear kernel x&gt;x0, the γ ∈ {0.001,0.0001} in radial basis function kernel and the regularization coefficient C ∈ {1,10,100,1000}. Therefore, we choose radial basis function kernel with γ = 0.001, C = 10. The hyperparameters we obtain are C = 10,γ = 0.001, and selected kernel is radial basis function kernel. The test accuracy our SVM classifier we obtain is 89%.

We summarize the empirical performance of the selected approaches as Table 2. Both SVM and Random Forest achieves 100% accuracy on training set. However, the test accuracy of random forest is only 77%, which is far less than 100%, thus we can conclude that the random forest suffers from serious over-fitting. The highest accuracy is achieved by SVM, which is 89%, higher than the baseline method He et al.[3]. SVM also stand out for its computation efficiency. The execution time of SVM is only 0.87 minutes, far less than another competitive model: ResNet-18, which takes 12.58 minutes to train the model.

Model Train Accuracy Test Accuracy Execution time (min)

SVM 100% 89% 0.87

Random Forest 100% 77% 0.04

Simple CNN 81.222% 70.67% 4.61

ResNet-18 82.116% 81.33% 12.58

He et al.[3] – 86% –

Table 1: Comparison of SVM, random forest, simple CNN, ResNet-18 and the baseline method He et al.[3].

5.2 Results on extracted feature by simple CNN

We use the idea in 4.4, we apply several machine learning methods on the extracted feature vectors, we also search for the best parameters for each model and make predictions. We divided the datasets to 5 fold and compute the mean cross validation accuracy. The results are shown below:

Model Train Accuracy Test Accuracy K-fold accuracy

Logistic Regression 99% 59% 97.52%

Decision Tree 98.14% 60% 97.37%

Random Forest 98.45% 57% 97.83%

K Nearest Neighbors 98.61% 58% 98.45%

Naive Bayes 98.60% 57% 98.45%

SVM 98.45% 63% 98.76%

Table 2: Comparison of the baseline method on extracted features.

The SVM gave the best test accuracy so we draw the hyperplane on the 2 dimension plot to show the region of data in 2 classes:

Figure 3: SVM with different γ in Gaussian kernel

The plot shows a fairly clear hyperplane to separate the two regions. We conclude the SVM can separate the data well on a higher dimensional space.

6 Conclusions and future work

We utilize the high accuracy from neural network along with the interpretation from machine learning models to better help make decisions and explain the relationship between COVID results and scanned images from a different perspective.

To summarize, SVM is the best model among the selected models to diagnose COVID-19, with the highest test accuracy of 89% and high computation efficiency. The ResNet architecture has the second high test accuracy, which can be improved if we tuned the layers and nodes to make them more suitable for the data.

In this work, the methods with raw dataset have generally better performance than those on extracted features. In our future work, we can explore more on how to use CNN architecture extract features well and combine them with our other machine learning models.

We will also focus on the feature importance generated by machine learning methods. We will explore which feature can best represent the CT image and help distinguish the Covid and Non-Covid results.

References

[4] S¸aban Oztu¨rk, Umut¨ Ozkaya, and Mu¨cahid Barstu˘gan. Classification of coronavirus (covid-19) from¨ x-ray and ct images using shrunken features. International Journal of Imaging Systems and Technology.

[11] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015.

[12] Vladimir N. Vapnik. The Nature of Statistical Learning Theory. Springer-Verlag, Berlin, Heidelberg, 1995.

A Appendix: Data processing Python source code

# pandas: plot the accuracy results, read files, save the data, etc import pandas as pd

# os: make sure the code can be executed for all operating systems import os

# random: draw random samples

1

2

3

4

5

6

7

8 import random

9

# imread: read and display images from skimage.io import imread

# train_test_split: create validation set from sklearn.model_selection import train_test_split

# pyplot: plot import matplotlib.pyplot as plt

# Image: process images such as resize/clip the images from PIL import Image

# shutil: remove an entire directory import shutil

# load the images’ names and label, and save them into a dataframe # use filter to skip the flies such as DS_Store and other caches covidImgs = list(filter(lambda x: len(x.split(“.”)) &gt; 1, os.listdir(os.path.join(“dataset”,

“CT”, “CT_COVID”)))) healthImgs = list(filter(lambda x: len(x.split(“.”)) &gt; 1, os.listdir(os.path.join(“dataset”, “CT”, “CT_NonCOVID”)))) # 1: True; 0: False ctLabel = pd.DataFrame({“Image”: covidImgs + healthImgs, “Covid”: [1]*len(covidImgs) + [0]* len(healthImgs)})

ctLabel

# create the training set and test set train_name, test_name, train_y, test_y = train_test_split(ctLabel.Image, ctLabel.Covid, test_size = 0.1) # create the validation set train_name, val_name, train_y, val_y = train_test_split(train_name, train_y, test_size =

0.1) def resizeImage(folder):

“””

Resize all the images in the folder to (115, 61).

Parameter ——–folder: the folder to save the resized images: “train”, “test” and “validation”.

Return —–a list of image arrays

“””

if folder == ’train’: x, y = train_name, train_y

elif folder == ’test’:

x, y = test_name, test_y else:

x, y = val_name, val_y

imgs = []

new_dir = os.path.join(“dataset”, “CT”, folder)

# remove previous data, recreate the train/val/test dataset

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63 if os.path.exists(new_dir) and os.path.isdir(new_dir):

64 shutil.rmtree(new_dir)

65 os.makedirs(new_dir)

66

for name, covid in zip(x, y):

original_folder = “CT_COVID” if covid else “CT_NonCOVID” img_path = os.path.join(“dataset”, “CT”, original_folder, name) img = Image.open(img_path).convert(’RGB’).resize((115, 61)).convert(’L’) new_path = os.path.join(new_dir, name) img.save(new_path, “JPEG”, optimize=True) img.close()

#################################################################### ###### read the image arrays (not necessary for this function) ##### ###### can be removed ############################################## #################################################################### img = imread(new_path, as_gray=True) # convert the type of pixel to float32 img = img.astype(’float32’) # normalize the pixel values img /= 255.0

# append the image into the list imgs.append(img) return imgs

train_x, test_x, val_x = resizeImage(“train”), resizeImage(“test”), resizeImage(”

validation”)

# display the CT images for i in random.choices(range(len(train_x)), k=4):

plt.imshow(train_x[i], cmap=’gray’) plt.show()

# write the data into csv files pd.DataFrame({“name”:train_name , “covid”:train_y}).to_csv(’train.csv’, index=False) pd.DataFrame({“name”:test_name , “covid”:test_y}).to_csv(’test.csv’, index=False) pd.DataFrame({“name”:val_name , “covid”:val_y}).to_csv(’validation.csv’, index=False)

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

B Appendix: Random forest implementation with Python

# os: make sure the code related to path can work for all operating systems

import os

# time: benchmarking import time

# sklearn: report the classification accuracy import sklearn

# RandomForestClassifier: the random forest classifier from sklearn.ensemble import RandomForestClassifier

# GridSearchCV: select the hyperparameter from sklearn.model_selection import GridSearchCV

# pandas: plot the accuracy results, read files, save the import pandas as pd

# numpy: process arrays import numpy as np

# os: make sure the code can be executed for all operating data, etc

systems

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23 import os

24

# random: draw random samples import random

# imread: read and display images from skimage.io import imread

# train_test_split: create validation set from sklearn.model_selection import train_test_split

# pyplot: plot import matplotlib.pyplot as plt

# Image: process images such as resize/clip the images from PIL import Image

# shutil: remove an entire directory import shutil

# time: benchmarking import time

DIMENSION = (115, 61)

IMG_PATH = os.path.join(“dataset”, “CT”)

# Importing Train and Test datasets train_data = pd.read_csv(“train.csv”) test_data = pd.read_csv(“test.csv”) val_data = pd.read_csv(“validation.csv”)

def readImage(folder): “””

Get the image data array of all images in one folder.

Parameters ———folder: the name of the folder: “train”, “test”, “validation”.

Returns

——-

A list of image arrays and the labels.

“””

data = pd.read_csv(folder + “.csv”)

x, y = data.name, data.covid

imgs = [] new_dir = os.path.join(IMG_PATH, folder)

for name, covid in zip(x, y):

img_path = os.path.join(new_dir, name) img = imread(img_path, as_gray=True).flatten()

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76 # convert the type of pixel to float32

77 img = img.astype(’float32’)

78 # normalize the pixel values

79 img /= 255.0

80 # append the image into the list

81 imgs.append(img)

82

83 return imgs, y

84

85 # load the datasets

86 train_x, train_y = readImage(“train”)

87 test_x, test_y = readImage(“test”)

88 val_x, val_y = readImage(“validation”)

89

90 # ================== Using Random Forest without hyper paramter tuning and clustering

=================== start = time.time() rf = RandomForestClassifier(n_estimators = 100) rf.fit(train_x+val_x, np.concatenate((train_y.values, val_y.values)))

# rf.fit(train_x, train_y)

print(“Training data metrics:”) print(sklearn.metrics.classification_report(y_true = train_y, y_pred = rf.predict(train_x)))

print(“Validation data metrics:”) print(sklearn.metrics.classification_report(y_true = val_y, y_pred = rf.predict(val_x)))

# Predictions on testset

# test data metrics print(“Test data metrics:”) print(sklearn.metrics.classification_report(y_true = test_y, y_pred = rf.predict(test_x))) end = time.time() print(“Time elapsed: %.2f min” % ((end-start)/60))

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

C Appendix: Simple CNN implementation with Python

# In[1]:

# pandas: plot the accuracy results, read files, save the data, etc import pandas as pd

# os: make sure the code can be executed for all operating systems import os

# random: draw random samples import random

# imread: read and display images from skimage.io import imread

# train_test_split: create validation set from sklearn.model_selection import train_test_split

# pyplot: plot import matplotlib.pyplot as plt

# Image: process images such as resize/clip the images from PIL import Image

# shutil: remove an entire directory import shutil

# time: benchmarking import time

# pytorch libraries and modules import torch

# numpy: process the data import numpy as np

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37 from torch.autograd import Variable

38 from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module,

Softmax, BatchNorm2d, Dropout from torch.utils.data import Dataset from torchvision import transforms from torch.utils.data import DataLoader import torch.nn.functional as F from torch.optim import Adam, SGD # In[2]:

#————————### SETTINGS

#————————-

# Hyperparameters

RANDOM_SEED = 1

LEARNING_RATE = 0.05

BATCH_SIZE = 8

NUM_EPOCHS = 20

# Architecture

NUM_CLASSES = 2

DIMENSION = (115, 61)

# Other

DEVICE = torch.device(’cuda:0’ if torch.cuda.is_available() else ’cpu’)

IMG_PATH = os.path.join(“dataset”, “CT”) NUM_WORKERS = 0 # In[3]:

def readImage(folder): “””

Get the image data array of all images in one folder.

Parameters ———folder: the name of the folder: “train”, “test”, “validation”.

Returns

——-

A list of image arrays and the labels.

“””

data = pd.read_csv(folder + “.csv”)

x, y = data.name, data.covid imgs = []

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89 new_dir = os.path.join(IMG_PATH, folder)

90

91 for name, covid in zip(x, y):

92 img_path = os.path.join(new_dir, name) 93 img = imread(img_path, as_gray=True)

94 # convert the type of pixel to float32

95 img = img.astype(’float32’)

96 # normalize the pixel values

97 img /= 255.0

98 # append the image into the list

99 imgs.append(img)

100

101 return imgs, y

102

103 # load the datasets

train_x, train_y = readImage(“train”) test_x, test_y = readImage(“test”) val_x, val_y = readImage(“validation”) # In[4]:

plt.imshow(train_x[random.randrange(len(train_x))], cmap=’gray’) plt.show() # In[5]:

class CTDataset(Dataset):

“””Custom Dataset for loading CT images””” def __init__(self, csv_path, img_dir, transform=None):

df = pd.read_csv(csv_path) self.img_dir = [img_dir] * len(df[’name’].values) self.img_names = df[’name’].values self.y = df[’covid’].values self.transform = transform

def __getitem__(self, index):

img = Image.open(os.path.join(self.img_dir[index],

self.img_names[index]))

# img = imread(os.path.join(self.img_dir[index],

# self.img_names[index]), as_gray=True).astype(’

float32’) if self.transform is not None:

img = self.transform(img)

label = self.y[index] return img, label

def __len__(self):

return self.y.shape[0]

# concatenate two datasets def __add__(self, newDataset):

self.img_names = np.concatenate((self.img_names, newDataset.img_names)) self.img_dir += newDataset.img_dir self.y = np.concatenate((self.y, newDataset.y)) return self

# In[6]:

# Note that transforms.ToTensor()

# already divides pixels by 255. internally

custom_transform = transforms.Compose([#transforms.Lambda(lambda x: x/255.),

transforms.ToTensor()])

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

162 train_dataset = CTDataset(csv_path=’train.csv’,

163 img_dir=os.path.join(“dataset”, “CT”, “train”), 164 transform=custom_transform)

165

166 valid_dataset = CTDataset(csv_path=’validation.csv’,

img_dir=os.path.join(“dataset”, “CT”, “validation”), transform=custom_transform)

# this method don’t need validation dataset

train_loader = DataLoader(dataset= train_dataset+valid_dataset,

batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

valid_loader = DataLoader(dataset=valid_dataset,

batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

test_dataset = CTDataset(csv_path=’test.csv’, img_dir=os.path.join(“dataset”, “CT”, “test”), transform=custom_transform)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# In[7]:

# Checking the dataset for images, labels in train_loader:

print(’Image batch dimensions:’, images.shape) print(’Image label dimensions:’, labels.shape) break

# Checking the dataset for images, labels in train_loader:

print(’Image batch dimensions:’, images.shape) print(’Image label dimensions:’, labels.shape) break

# In[8]:

# This cell just checks if the dataset can be loaded correctly.

torch.manual_seed(0)

167

168

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

185

186

187

188

189

190

191

192

193

194

195

196

197

198

199

200

201

202

203

204

205

206

207

208

209

210

211

212

213

214

215

216 num_epochs = 2

217 for epoch in range(num_epochs):

218

219 for batch_idx, (x, y) in enumerate(train_loader):

220

221 print(’Epoch:’, epoch+1, end=’’)

222 print(’ | Batch index:’, batch_idx, end=’’)

223 print(’ | Batch size:’, y.size()[0])

224

225 x = x.to(DEVICE)

226 y = y.to(DEVICE)

227

228 print(’break minibatch for-loop’)

229 break

230

# # Multilayer Perceptron Model # In[9]:

###############################

### NO NEED TO CHANGE THIS CELL ###############################

def compute_epoch_loss(model, data_loader):

model.eval() curr_loss, num_examples = 0., 0 with torch.no_grad():

for features, targets in data_loader: features = features.to(DEVICE) targets = targets.to(DEVICE) logits, probas = model(features) loss = F.cross_entropy(logits, targets, reduction=’sum’) num_examples += targets.size(0) curr_loss += loss

curr_loss = curr_loss / num_examples return curr_loss

def compute_accuracy(model, data_loader, device):

model.eval() correct_pred, num_examples = 0, 0 for i, (features, targets) in enumerate(data_loader):

features = features.to(device) targets = targets.to(device)

logits, probas = model(features)

_, predicted_labels = torch.max(probas, 1) num_examples += targets.size(0) correct_pred += (predicted_labels == targets).sum()

return correct_pred.float()/num_examples * 100

# In[10]:

class ConvNet(torch.nn.Module):

def __init__(self, num_classes): super(ConvNet, self).__init__() self.num_classes = num_classes

### Layers: ADD ADDITIONAL LAYERS BELOW IF YOU LIKE

# 115*61*1 =&gt; 28*28*8

self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(6,6),

stride=(2,4), padding=(0,1))

# 28*28*8 =&gt; 14*14*8 self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

231

232

233

234

235

236

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

264

265

266

267

268

269

270

271

272

273

274

275

276

277

278

279

280

281

282

283

284

285

286

287

288

289

290 # 14*14*8 =&gt; 14*14*16

291 self.conv_2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3),

stride=(1,1), padding=1)

# 14*14*16 =&gt; 7*7*16 self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0) self.bn_2 = torch.nn.BatchNorm2d(16)

# 7*7*16 =&gt; 7*7*32 self.conv_3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride

=1, padding=2) self.bn_3 = torch.nn.BatchNorm2d(32)

# Multilayer perceptron self.linear_1 = torch.nn.Linear(7*7*32, 7*7*64) self.bn_l1 = torch.nn.BatchNorm1d(7*7*64) self.linear_2 = torch.nn.Linear(7*7*64, 7*7*64) self.bn_l2 = torch.nn.BatchNorm1d(7*7*64) self.linear_out = torch.nn.Linear(7*7*64, num_classes)

def forward(self, x):

### MAKE SURE YOU CONNECT THE LAYERS PROPERLY IF YOU CHANGED

### ANYTHNG IN THE __init__ METHOD ABOVE out = self.conv_1(x) out = F.relu(out) out = self.pool_1(out)

out = self.conv_2(out) out = self.bn_2(out)

# out = F.dropout(out, p=0.2, training=self.training) out = F.relu(out) out = self.pool_2(out)

out = self.conv_3(out) out = self.bn_3(out)

# out = F.dropout(out, p=0.2, training=self.training) out = F.relu(out)

out = self.linear_1(out.view(-1, 7*7*32)) out = self.bn_l1(out) out = F.relu(out)

# out = F.dropout(out, p=0.2, training=self.training)

out = self.linear_2(out) out = self.bn_l2(out) out = F.relu(out)

# out = F.dropout(out, p=0.2, training=self.training)

logits = self.linear_out(out) probas = F.softmax(logits, dim=1) return logits, probas

################################# ### Model Initialization ###

#################################

# the random seed makes sure that the random weight initialization

292

293

294

295

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

313

314

315

316

317

318

319

320

321

322

323

324

325

326

327

328

329

330

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

347

348

349

350

351 # in the model is always the same.

353 # to try different random seeds. In this homework, this is not

# necessary. torch.manual_seed(RANDOM_SEED)

### IF YOU CHANGED THE ARCHITECTURE ABOVE, MAKE SURE YOU

### ACCOUNT FOR IT VIA THE PARAMETERS BELOW. I.e., if you

model = ConvNet(NUM_CLASSES) model = model.to(DEVICE)

### For this homework, do not change the optimizer. However, you ### likely want to experiment with the learning rate! optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) # In[11]:

###############################

### NO NEED TO CHANGE THIS CELL ############################### def train(model, train_loader, test_loader):

minibatch_cost, epoch_cost = [], [] start_time = time.time() for epoch in range(NUM_EPOCHS):

model.train() for batch_idx, (features, targets) in enumerate(train_loader):

features = features.to(DEVICE) targets = targets.to(DEVICE)

### FORWARD AND BACK PROP logits, probas = model(features) cost = F.cross_entropy(logits, targets) optimizer.zero_grad()

cost.backward() minibatch_cost.append(cost)

### UPDATE MODEL PARAMETERS optimizer.step()

### LOGGING if not batch_idx % 150:

print (’Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f’

%(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost))

with torch.set_grad_enabled(False): # save memory during inference print(’Epoch: %03d/%03d | Train: %.3f%%’ % ( epoch+1, NUM_EPOCHS, compute_accuracy(model, train_loader, device=DEVICE)))

354

355

356

357

358

359

360

361

362

363

364

365

366

367

368

369

370

371

372

373

374

375

376

377

378

379

380

381

382

383

384

385

386

387

388

389

390

391

392

393

394

395

396

397

398

399

400

401

402

403

404

405

406

407

408

409

410

411

412

413

414 cost = compute_epoch_loss(model, train_loader)

415 epoch_cost.append(cost)

416

417 print(’Time elapsed: %.2f min’ % ((time.time() – start_time)/60))

print(’Total Training Time: %.2f min’ % ((time.time() – start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference print(’Test accuracy: %.2f%%’ % (compute_accuracy(model, test_loader, device=DEVICE)

)) print(’Total Time: %.2f min’ % ((time.time() – start_time)/60)) return minibatch_cost, epoch_cost

minibatch_cost, epoch_cost = train(model, train_loader, test_loader)

plt.plot(range(len(minibatch_cost)), minibatch_cost) plt.ylabel(’Cross Entropy’) plt.xlabel(’Minibatch’) plt.show()

plt.plot(range(len(epoch_cost)), epoch_cost) plt.ylabel(’Cross Entropy’) plt.xlabel(’Epoch’) plt.show()

418

419

420

421

422

423

424

425

426

427

428

429

430

431

432

433

434

435

436

437

438

439

440

441

D Appendix: ResNet implementation with Python

# In[1]:

# pandas: plot the accuracy results, read files, save the data, etc import pandas as pd

# os: make sure the code can be executed for all operating systems import os

# random: draw random samples import random

# imread: read and display images from skimage.io import imread

# train_test_split: create validation set from sklearn.model_selection import train_test_split

# pyplot: plot import matplotlib.pyplot as plt

# Image: process images such as resize/clip the images from PIL import Image

# shutil: remove an entire directory import shutil

# time: benchmarking import time

# pytorch libraries and modules

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32 import torch

33

34 # numpy: process the data

35 import numpy as np

36

from torch.autograd import Variable

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module,

Softmax, BatchNorm2d, Dropout from torch.utils.data import Dataset from torchvision import transforms from torch.utils.data import DataLoader import torch.nn.functional as F from torch.optim import Adam, SGD # In[2]:

#————————### SETTINGS

#————————-

# Hyperparameters

RANDOM_SEED = 1

LEARNING_RATE = 0.05

BATCH_SIZE = 8

NUM_EPOCHS = 20

# Architecture

NUM_FEATURES = 32*32

NUM_CLASSES = 2

DIMENSION = (115, 61)

# Other

DEVICE = torch.device(’cuda:0’ if torch.cuda.is_available() else ’cpu’)

IMG_PATH = os.path.join(“dataset”, “CT”) NUM_WORKERS = 0

# # Loading the data # In[3]:

class CTDataset(Dataset):

“””Custom Dataset for loading CT images””” def __init__(self, csv_path, img_dir, transform=None):

df = pd.read_csv(csv_path)

self.img_dir = [img_dir] * len(df[’name’].values) self.img_names = df[’name’].values self.y = df[’covid’].values self.transform = transform

def __getitem__(self, index):

img = Image.open(os.path.join(self.img_dir[index],

self.img_names[index]))

# img = imread(os.path.join(self.img_dir[index],

# self.img_names[index]), as_gray=True).astype(’

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

float32’)

91 if self.transform is not None: 92 img = self.transform(img)

93

94 label = self.y[index]

95 return img, label

96

97 def __len__(self):

98 return self.y.shape[0]

# concatenate two datasets def __add__(self, newDataset):

self.img_names = np.concatenate((self.img_names, newDataset.img_names)) self.img_dir += newDataset.img_dir self.y = np.concatenate((self.y, newDataset.y)) return self

# In[4]:

# Note that transforms.ToTensor()

# already divides pixels by 255. internally

custom_transform = transforms.Compose([#transforms.Lambda(lambda x: x/255.), transforms.ToTensor()])

train_dataset = CTDataset(csv_path=’train.csv’, img_dir=os.path.join(“dataset”, “CT”, “train”), transform=custom_transform)

valid_dataset = CTDataset(csv_path=’validation.csv’, img_dir=os.path.join(“dataset”, “CT”, “validation”), transform=custom_transform)

# this method don’t need validation dataset

train_loader = DataLoader(dataset= train_dataset+valid_dataset,

batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

valid_loader = DataLoader(dataset=valid_dataset,

batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

test_dataset = CTDataset(csv_path=’test.csv’, img_dir=os.path.join(“dataset”, “CT”, “test”), transform=custom_transform)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# In[5]:

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152 # Checking the dataset

153 for images, labels in train_loader:

154 print(’Image batch dimensions:’, images.shape)

155 print(’Image label dimensions:’, labels.shape)

156 break

157

158 # Checking the dataset

159 for images, labels in train_loader:

160 print(’Image batch dimensions:’, images.shape)

161 print(’Image label dimensions:’, labels.shape)

162 break

# In[6]:

# This cell just checks if the dataset can be loaded correctly.

torch.manual_seed(0)

num_epochs = 2 for epoch in range(num_epochs):

for batch_idx, (x, y) in enumerate(train_loader):

print(’Epoch:’, epoch+1, end=’’) print(’ | Batch index:’, batch_idx, end=’’) print(’ | Batch size:’, y.size()[0])

x = x.to(DEVICE) y = y.to(DEVICE)

print(’break minibatch for-loop’) break

# # ResNet # In[7]:

def compute_epoch_loss(model, data_loader):

model.eval() curr_loss, num_examples = 0., 0 with torch.no_grad():

for features, targets in data_loader: features = features.to(DEVICE) targets = targets.to(DEVICE) logits = model(features)

loss = F.cross_entropy(logits, targets, reduction=’sum’) num_examples += targets.size(0) curr_loss += loss

curr_loss = curr_loss / num_examples return curr_loss

def compute_accuracy(model, data_loader): model.eval() correct_pred, num_examples = 0, 0 for i, (features, targets) in enumerate(data_loader):

features = features.to(DEVICE) targets = targets.to(DEVICE)

163

164

165

166

167

168

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

185

186

187

188

189

190

191

192

193

194

195

196

197

198

199

200

201

202

203

204

205

206

207

208

209

210

211

212

213

214

215

216 logits = model(features)

217 _, predicted_labels = torch.max(logits, 1)

218 num_examples += targets.size(0)

219 correct_pred += (predicted_labels == targets).sum()

220 return correct_pred.float()/num_examples * 100

221

222

223

224 # In[8]:

225

226

model = torch.hub.load(’pytorch/vision:v0.6.0’, ’resnet18’, pretrained=True)

# or any of these variants

# model = torch.hub.load(’pytorch/vision:v0.6.0’, ’resnet34’, pretrained=True)

# model = torch.hub.load(’pytorch/vision:v0.6.0’, ’resnet50’, pretrained=True)

# model = torch.hub.load(’pytorch/vision:v0.6.0’, ’resnet101’, pretrained=True)

# model = torch.hub.load(’pytorch/vision:v0.6.0’, ’resnet152’, pretrained=True) model.eval() # In[9]:

# keep the pretrained layers, don’t update them for parameter in model.parameters():

parameter.requires_grad = False

# In[10]:

model.conv1 = torch.nn.Conv2d(1,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=

False) model.fc = torch.nn.Sequential( torch.nn.Linear(in_features=512, out_features=100, bias=True), torch.nn.ReLU(inplace=True), torch.nn.Dropout(p=0.5, inplace=False), torch.nn.Linear(in_features=100, out_features= NUM_CLASSES, bias=True))

# In[15]:

# instances where you want to save and load your neural networks across different devices. model = model.to(DEVICE)

# torch.optim is a package implementing various optimization algorithms.

optimizer = torch.optim.Adam([

{’params’: model.conv1.parameters()},

{’params’: model.fc.parameters()} ])

# In[16]:

def train(model, train_loader, test_loader, NUM_EPOCHS):

minibatch_cost, epoch_cost = [], [] start_time = time.time() for epoch in range(NUM_EPOCHS):

model.train() for batch_idx, (features, targets) in enumerate(train_loader):

227

228

229

230

231

232

233

234

235

236

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

264

265

266

267

268

269

270

271

272

273

274

275

276

277

278

279

280 features = features.to(DEVICE)

281 targets = targets.to(DEVICE)

282

283 ### FORWARD AND BACK PROP

284 logits = model(features)

285 cost = F.cross_entropy(logits, targets)

286 optimizer.zero_grad()

287

288 cost.backward()

289 minibatch_cost.append(cost)

### UPDATE MODEL PARAMETERS optimizer.step()

### LOGGING if not batch_idx % 150:

print (’Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f’

%(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost))

with torch.set_grad_enabled(False): # save memory during inference print(’Epoch: %03d/%03d | Train: %.3f%%’ % ( epoch+1, NUM_EPOCHS, compute_accuracy(model, train_loader)))

cost = compute_epoch_loss(model, train_loader) epoch_cost.append(cost) print(’Time elapsed: %.2f min’ % ((time.time() – start_time)/60)) print(’Total Training Time: %.2f min’ % ((time.time() – start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference print(’Test accuracy: %.2f%%’ % (compute_accuracy(model, test_loader))) print(’Total Time: %.2f min’ % ((time.time() – start_time)/60)) return minibatch_cost, epoch_cost

# In[17]:

minibatch_cost, epoch_cost = train(model, train_loader, test_loader, NUM_EPOCHS = NUM_EPOCHS

)

plt.plot(range(len(minibatch_cost)), minibatch_cost) plt.ylabel(’Cross Entropy’) plt.xlabel(’Minibatch’) plt.show()

plt.plot(range(len(epoch_cost)), epoch_cost) plt.ylabel(’Cross Entropy’) plt.xlabel(’Epoch’) plt.show()

290

291

292

293

294

295

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

313

314

315

316

317

318

319

320

321

322

323

324

325

326

327

328

329

330

331

332

333

334

335

E Appendix: SVM implementation with Python

# Path: list the files in the directory from pathlib import Path

# os: make sure code can be executed in all import os operating systems

1

2

3

4

5

6

# time: benchmarking import time

# plt: draw pictures import matplotlib.pyplot as plt

# svm: svm classifier from sklearn import svm, metrics

# Bunch: container object for datasets from sklearn.utils import Bunch

# numpy: process the image arrays import numpy as np

# pandas: process the image arrays import pandas as pd

# GridSearchCV: Exhaustive search over specified parameter values for an estimator # train_test_split: split the full datasets into train and test dataset, respectively from sklearn.model_selection import GridSearchCV, train_test_split

# imread: read the images from skimage.io import imread

# resize: resize the images from skimage.transform import resize

# plot_decision_regions: visualize the SVM hyperplane from mlxtend.plotting import plot_decision_regions

DIMENSION = (115, 61)

IMG_PATH = os.path.join(“dataset”, “CT”)

def load_image_files(container_path, dimension=DIMENSION):

“””

Load image files with categories as subfolder names which performs like scikit-learn sample dataset

Parameters ———-

container_path : string or unicode

Path to the main folder holding one subfolder per category dimension : tuple size to which image are adjusted to

Returns

——Bunch “””

image_dir = Path(container_path)

folders = [directory for directory in image_dir.iterdir() if (directory.is_dir() and ” COVID” in directory.name)] categories = [fo.name for fo in folders]

descr = “A image classification dataset” images = [] flat_data = [] target = [] for i, direc in enumerate(folders):

for file in direc.iterdir():

if len(file.name.split(“.”)) == 1 or file.name[0] == ’.’:

continue

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

img = imread(file, as_gray = True) img_resized = resize(img, dimension, anti_aliasing=True, mode=’reflect’) flat_data.append(img_resized.flatten()) images.append(img_resized) target.append(i)

flat_data = np.array(flat_data) target = np.array(target) images = np.array(images)

return Bunch(data=flat_data,

target=target, target_names=categories, images=images, DESCR=descr)

image_dataset = load_image_files(IMG_PATH)

X_train, X_test, y_train, y_test = train_test_split( image_dataset.data, image_dataset.target, test_size=0.1,random_state=109)

param_grid = [

{’C’: [1, 10, 100, 1000], ’kernel’: [’linear’]},

{’C’: [1, 10, 100, 1000], ’gamma’: [0.001, 0.0001], ’kernel’: [’rbf’]},

]

svc = svm.SVC()

start = time.time() clf1 = GridSearchCV(svc, param_grid, cv = 3) clf1.fit(X_train, y_train) y_pred = clf1.predict(X_test) print(“Classification report for – {}: {} “.format( clf1, metrics.classification_report(y_test, y_pred)))

print(“Time elasped: %.2f min” % ((time.time()-start)/60)) clf1.best_params_

param_grid = [

{’C’: [10, 50], ’gamma’: [0.002, 0.001, 0.0005], ’kernel’: [’rbf’]},

]

svc = svm.SVC()

start = time.time()

clf2 = GridSearchCV(svc, param_grid, cv = 3) clf2.fit(X_train, y_train) y_pred = clf2.predict(X_test) print(“Classification report for – {}: {} “.format( clf2, metrics.classification_report(y_test, y_pred)))

print(“Time elasped: %.2f min” % ((time.time()-start)/60)) clf2.best_params_

# train accuracy y_pred = clf2.predict(X_train) print(“Classification report for – {}: {} “.format( clf1, metrics.classification_report(y_train, y_pred)))

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124
