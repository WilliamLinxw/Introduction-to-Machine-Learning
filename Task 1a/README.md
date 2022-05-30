In this task, we are required to report 5 ten-fold cross validation errors for 5 different penalty parameters in a ridge regression task.

Reading data and suffling: I first read the data from the provided dataset, then I suffled the data randomly to avoid bias.

Data split, training, and validation: I selected a row of data everay ten rows to put into the validation set to raise randomness. The rest rows are put into the training set. I then performed ridge regression on the training set. With the trained coefficients and intercepts, the x-validation errors were calculated and stored in a list for the average error.

Perform training and validation on the sets with resulting errors averaged: I performed the procedure above by increasing the index to ten(choosing different rows for the validation set), with ten errors calculated. Finally, the average error on the validation set is calculated and reported.
