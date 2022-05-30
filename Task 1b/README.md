In this project, we are asked to used hand crafted features for a regression task.

With the data read from the file, I performed the required feature design by manipulating columns.

I then used the LinearRegression class provided by sklearn to perform the training. Because the last feature is a constantand the method will also returen the intercept after training, I dropped the last feature during training. After training, I added the intercept back into the trained parameter for the constant feature and report the corresponding coefficients.
