# Introduction-to-Machine-Learning
Project codes in Introduction to Machine Learning

## Task 1a
In this task, we are required to report 5 ten-fold cross validation errors for 5 different penalty parameters in a ridge regression task.

Reading data and suffling: I first read the data from the provided dataset, then I suffled the data randomly to avoid bias.

Data split, training, and validation: I selected a row of data everay ten rows to put into the validation set to raise randomness. The rest rows are put into the training set. I then performed ridge regression on the training set. With the trained coefficients and intercepts, the x-validation errors were calculated and stored in a list for the average error.

Perform training and validation on the sets with resulting errors averaged: I performed the procedure above by increasing the index to ten(choosing different rows for the validation set), with ten errors calculated. Finally, the average error on the validation set is calculated and reported.

## Task 1b
In this project, we are asked to used hand crafted features for a regression task.

With the data read from the file, I performed the required feature design by manipulating columns.

I then used the LinearRegression class provided by sklearn to perform the training. Because the last feature is a constantand the method will also returen the intercept after training, I dropped the last feature during training. After training, I added the intercept back into the trained parameter for the constant feature and report the corresponding coefficients.

# Task 2
This project mainly consists of data manipulation and model selection and training. All the testing codes are included in the latter part of the submitted code, indicated by the comment cell '''Production Code during testing'''. Run the cells until this cell can reproduce the prediction.

I averaged the twelve tests for one patient and used KNN for data imputation. This algorithm tries to impute the missing data based on k other samples with a value for the feature and the nearest Euclidean distance. The lost data is computed by averaging the neighbor's existing data. Our final solution also has built-in support for missing data.

The second part is to find appropriate models for the classification and regression tasks. I first tried the models introduced in the lectures, including the Logistic Regression, SGD method, SVM with RBF or polynomial kernels, and KNN. For the Logistic Regression, the model is too simple to capture the data's features. The method using SGD has the same issue. The runtime was very high for the SVM with kernels. It needs about ten minutes to train one model. Since I used 5-fold cross-validation to choose the hyperparameters, the runtime was very long. The performance was not that good either. The KNN classifier is fast, but the performance is not good.

I then turned to the ensemble method suitable for both classification and regression. I tried the Gradient Boosting method, a decision tree, and a regression tree type method. Even with the default parameters, this algorithm runs fast and outperforms the methods above. I used x-validation to choose k=1000 for the KNN imputation step.

The histogram-based methods provided by Sklearn also have built-in support for the missing values. I tried without imputation, and the performance during x-validation was even better. I decided to use the Histogram Gradient Boosting Classifier and Regressor for the non-imputed averaged data. The prediction result is submitted.

## Task 3
In this project, we are asked to determine, in a triplet consisting of three images, if the first image is more similar to the second one or the third one. The first image is called the anchor, the second is called the positive, and the third is called the negative. To realize the task, a triplet net was implemented, which consists of three parallel convolutional neural networks to extract the features of three input images and map the features onto a 1024 dimensional output using a fully connected layer. The three outputs were fed into the triplet margin loss, where the Euclidean distances between the anchor and the positive and the negative respectively were calculated. The triplet margin loss is to penalize the high distance between the anchor and the positive. Stochastic gradient descent is used to optimize the loss function. The networks' parameters are all updated during training. After training, the network is fed with the test triplets, calculating the distance between the first two and between the first and the third one. If the first distance is higher, outputs 1, otherwise, it outputs 0.

We used Colab to perform training. We need to change the directory to the folder where the data text files and the folder of food images are located. Run the cells until the trained_net = train() can output the prediction, and the prediction file is saved under the stated directory of the train function.

## Task 4
In this project, we are required to predict the HOMO-LUMO gap of a given molecule. It's a transfer learning task, namely using part of a pretrained network with the features it learns to solve a similar but different task. Because the data set for HOMO-KUMO-gap-labeled samples is very limited, i.e. 100, while the dataset for E-LUMO-labeled data is sufficient, i.e. 50000. We would like to use the E-LUMO dataset to pretrain a neural network. With the features learned, we train a new network with the backbone of the pretrained network to predict the HOMO-LUMO gap of a molecule. Unlike task 3 where I used a pretrained ResNet18, I constructed the backbone neural network constructed myself.

Each of the sample in the pretrain dataset has 1000 features. I first constructed the backbone network, a fully connected NN with 5 layers, mapping from 1000 dimensions to 512, 512 to 256, 256 to 128, 128 to 64, and the last 64 to 1 output. I then fed all the samples, one by one, to the network to train it for five epochs. Feeding the samples one by one increases the randomness but leads to a slower convergence. With the pretrained network, I froze the first layer and use the training set of 100 samples to update the parameters of the last four layers for 4000 epochs until convergence. It doesn't overfit after so many epochs, I consider, because with the first layer fixed, the function class this NN can represent is not complex enough to fit the noise. Therefore, running the training until convergence can give a good result.

I used Colab for the task. To reproduce the result, make sure the datasets are in the corresponding folder and run the code until the last cell.
