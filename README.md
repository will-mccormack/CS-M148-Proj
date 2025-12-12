## Dataset Description [FINAL]

For our project, we selected the Spotify Prediction Dataset, sourced from Spotify’s Web API. This dataset is vast and high-dimensional: it contains 114,000 tracks across 125 genres with 20 features total. Each track has its own row and represents an observational unit. The most relevant columns consist of audio features, including but not limited to danceability, energy, loudness, and tempo, all of which are continuous variables scored from 0-100. Our response variable is "popularity," which Spotify computes algorithmically per track and is primarily based on the number of plays and how recent those plays were (also measured on a 0-100 scale). Other columns provide information about track metadata, such as track_id, artists, and duration_ms. Various adjustments were made to the original dataset for usability purposes in our analysis, all of which are detailed in the following sections.

## Problem Overview [FINAL]

Our primary goal for this project is to develop a model that can predict track popularity, using audio features as our predictor variables. This task is highly complex, as there are dozens of factors that contribute to a track’s popularity, many of which are not included in the dataset (for example, the extent to which a song is promoted prior to release would likely have influence on its popularity; our analysis won’t be able to capture this). Given the limiting nature of solely relying on genre and audio features, the objective becomes identifying data variables that demonstrate a consistent association with popularity. We are also interested to see how different modeling approaches handle this noisy setting.


## Key Methodology [UPDATES NEEDED]

Throughout our analysis, we explored several models of varying complexity and type with the goal of ultimately cross-referencing key model performance metrics across model types to identify the highest performing model. The models include several linear models as well as a neural network. Our Ordinary Least Squares regression model had the most predictive power/strongest interpretability across our linear models.

[Info needed on other models not referenced here]

### Exploratory Data Analysis

Exploratory data analysis was used to examine feature distributions and their relationships with popularity. Histograms and summary statistics revealed large differences in scale and variability across audio features, motivating feature standardization. Exploration of popularity showed weak linear relationships with individual features, suggesting that predictive signal is distributed across many variables and may be nonlinear.

Structural analysis revealed that many tracks appeared multiple times with identical audio features but different genre labels, introducing redundancy. The combination of numerous continuous features and categorical genre indicators also implied a high-dimensional and potentially correlated feature space. These observations motivated later preprocessing decisions, including duplicate consolidation, dimensionality reduction, and careful data splitting. Analysis of the explicit label was conducted for auxiliary classification experiments, but the primary focus remained on popularity prediction.

### Data Pre-processing and Feature Engineering

Preprocessing steps were informed by exploratory findings and model requirements. Low-signal metadata columns were removed, and categorical genre information was encoded using one-hot encoding. Duplicate tracks with identical audio features were grouped together, and their genre indicators were merged to reduce redundancy while preserving multi-genre information. In addition, after going thorugh the data pipeline, we evaluated the popularity metric and decided that values less than 5 contributed to noise, since the popularity metric algorithm is a measure of current popularity, without factoring in historical popularity which causes older songs as well as brand new songs to have an abnormally low popularity score. To deal with this noise, these songs are removed to allow a more predictable model to be trained.

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/f81c929a-c79a-4143-a19a-178e3c2e71b1" />

All continuous features were standardized prior to modeling. PCA was applied after scaling and fit only on the training data to reduce dimensionality and mitigate multicollinearity, particularly for neural network training. The resulting feature representations were used consistently across training and validation sets. No synthetic data augmentation was performed; feature engineering focused on improving representation quality and stability.

### Regression

This project investigated several regression models to check their ability to predict popularity. These include an unregularized Ordinary Least Squares (OLS) and Least Absolute Deviation (LAD) regression, along with Lasso and Ridge regularization on an Ordinary Least Squares model. 

#### Unregularized Models

All the data used was scaled because of the various ranges between different features.

##### Ordinary Least Squares Regression

The Ordinary Least Squares model was generated using sklearn's LinearRegression function, creating a fit using the training data, and making predictions using the testing data. Afterwards, the model can be evaluated with metrics including, R<sup>2</sup>, mean squared error (MSE), mean absolute error (MAE), and root mean squared error (rMSE). Then, through k-fold cross-validation, it can be determined what degree the model should be to best produce a fit. To reduce computational intensity, 3 folds were used, and two degrees were checked.

##### Least Absolute Deviation Regression

The methodology for LAD regression is similar to Ordinary Least Squares Regression, but instead, sklearn's QuantileRegressor function with the quantile parameter set equal to 0.5 is used. Again, fit and predict, and calculate the metrics stated in the Ordinary Least Squares Regression subsection.  For cross-validation, 2 folds were used, and two degrees were checked.

#### Regularized Models

##### Lasso

10-fold cross-validation is used to select the best regularization hyperparameter for Lasso regularization by importing Lasso from sklearn. The goal is to find an alpha that results in the minimum cross-validation score and an alpha that minimizes the score within 1 standard error. Afterwards, we can fit models using the optimal alpha values and calculate the relevant metrics.

##### Ridge

The process for Ridge regularization is the same as Lasso regularization, but instead, Ridge from sklearn is used. As before, find the hyperparameter value that minimizes the cross-validation score and minimize the score within 1 standard error. Then, we can fit models using the optimal alpha values and obtain the metrics.

#### Neural Networks

We also trained a feedforward neural network to capture nonlinear relationships between audio features and popularity. We applied PCA to scaled features before neural network training to mitigate the effects of the data’s inherent high-dimensionality as well as the dimensionality increase caused by genre encoding. From our testing we evaluated that 80% of the explained variance ratio can allow us reduce dimensionality without compromising the performance of our model. We trained the neural network using a validation set with early stopping and performed hyperparameter tuning to determine an optimal learning rate. 

Note that feature importance involving regularization was applied when Lasso and Ridge regularization were applied, but not the unregularized models, where keeping more features were favored over reducing complexity.


## Results [UPDATES NEEDED]

Models were evaluated using a consistent training–validation framework.

##### Ordinary Least Squares Regression

For Ordinary Least Squares, the R<sup>2</sup>, MSE, MAE, and RMSE are 0.55, 140.18, 9.08, and 11.84, respectively. The cross-validation graph is given below.

<p align="center">
  <img src="https://github.com/will-mccormack/CS-M148-Proj/blob/main/assets/ols_final.png" width="400">
</p>

Since the cross-validation score is minimized with degree 1, a model with degree 1 is the better model with an rMSE of about 11.8.

##### Least Absolute Deviation Regression

For LAD regression, the R<sup>2</sup>, MSE, MAE, and RMSE are 0.49, 157.40, 8.68, and 12.55, respectively. The cross-validation graph is given below.

<p align="center">
  <img src="https://github.com/will-mccormack/CS-M148-Proj/blob/main/assets/lad.png" width="400">
</p>

Since the cross-validation score is minimized with degree 2, a model with degree 2 is the better model with an rMSE of about 12.0. However, this is still slightly worse than Ordinary Least Squares.

##### Lasso

The graph of the cross-validation scores versus the alpha values is given below.

<p align="center">
  <img src="https://github.com/will-mccormack/CS-M148-Proj/blob/main/assets/lasso.png" width="400">
</p>

The value of alpha that minimizes the cross-validation score is 0.1, and the value of alpha that minimizes the cross-validation score with 1 standard error is 0.14.

By fitting a model using the alpha that minimizes the cross-validation score, the rMSE, R<sup>2</sup>, MAE, and MSE are 0.55, 140.69, 9.22, and 11.86, respectively. If the alpha that minimizes the cross-validation score within 1 standard error is used instead, the cross-validation score, the rMSE, R<sup>2</sup>, MAE, and MSE are 0.54, 141.97, 9.31, and 11.91, respectively.

##### Ridge

The graph of the cross-validation scores versus the log-alpha values is given below.

<p align="center">
  <img src="https://github.com/will-mccormack/CS-M148-Proj/blob/main/assets/ridge.png" width="400">
</p>

The value of alpha that minimizes the cross-validation score is 2848.04, and the value of alpha that minimizes the cross-validation score with 1 standard error is 27825.59.

By fitting a model using the alpha that minimizes the cross-validation score, the rMSE, R<sup>2</sup>, MAE, and MSE were 0.55, 140.30, 9.18, and 11.84, respectively. If the alpha that minimizes the cross-validation score within 1 standard error is used instead, the cross-validation score, the rMSE, R<sup>2</sup>, MAE, and MSE were 0.48, 161.15, 10.33, and 12.69, respectively.

Lasso and Ridge Regression allows for the investigation of feature importance. [talk about use in nn]

Based on the metrics obtained, Ordinary Least Squares provides the best predictions out of the linear models. However, depending on whether one desires marginally greater accuracy with greater model complexity, choosing the Ridge model would also be reasonable.

##### Neural Network

The neural network achieved a respectable validation RMSE of 11.27 and highest R-squared among the models considered, with a value of 0.5901, indicating, as expected that nonlinear interactions and genre effects contribute additionally towards full predictive power. However, overall R-squared values remained moderate, reflecting the inherent limitations of predicting popularity from audio features alone, without factoring artist or popularity trends over years. For this reason, the neural network is treated as the strongest predictive model within the scope of the dataset.

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/5816bc82-b73a-44b6-9546-8aa63603b8e3" />  

We utilized 5-Fold Cross-Validation to verify the robustness of our Neural Network. The results show a stable convergence profile (narrow standard deviation bands), confirming that the model's architecture is resilient to variations in the training data. The persistent but stable gap between training and validation loss highlights that the model has reached the predictive ceiling for this feature set without suffering from catastrophic overfitting.

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/8e86c5d8-da44-4cd1-9e42-c392472225b9" />

## How to Use the Code [Final]

All project code is available in the GitHub repository and can be run through Jupyter. The main workflow is contained in Final_Project_Main.ipynb, which loads the prepared datasets, applies preprocessing steps, and fits the primary models, regression and the neural network. Running this notebook in full will reproduce the results and evaluation metrics we report. Additional notebooks in our repository document some intermediate experiments but in general are not a significant part of our final workflow.


## Appendix:

### Exploratory Data Analysis

Please refer to the main document for exploratory data analysis procedures.

### Data Pre-processing and Feature Engineering

Please refer to the main document for data pre-processing and feature engineering.

***TODO:***

### Regression and Regularization

Please refer to the main document for a discussion of how regression analysis was applied to the project.

iv. How was logistic regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed?

Logistic regression was applied as a binary classification framework to distinguish between "popular" and "unpopular" tracks, utilizing a specific threshold to binarize the continuous popularity score. To address the dataset's imbalance, where unpopular songs significantly outnumber hits, the model employed class_weight='balanced', which adjusts the loss function to penalize misclassifying the minority class more heavily. The analysis was validated using 5-Fold Cross-Validation, yielding an average AUC of approximately 0.69 and an accuracy of 62.9%. This performance revealed that while audio features contain predictive signals, the relationship between these features and popularity is not strictly linear. Regularization (specifically L2, which is the default in scikit-learn) was necessary to handle the high dimensionality introduced by the one-hot encoded genre features, preventing the model from overfitting to the noise in the sparse data and allowing for the interpretation of feature importance via coefficient magnitude.

v. How were KNN, decision trees, or random forest used for classification on your data? What method worked best for your data and why was it good for the problem you were addressing?

For the non-linear classification tasks, Decision Trees and Random Forests were utilized to capture complex feature interactions that linear models miss, such as how high "energy" might correlate positively with popularity in the Rock genre but negatively in Classical. While K-Nearest Neighbors (KNN) theoretically groups similar songs, it often struggles with this dataset due to the "curse of dimensionality" introduced by the numerous genre columns. Consequently, the Random Forest method typically worked best for this problem. By aggregating the predictions of hundreds of independent decision trees, the Random Forest model effectively reduced the variance and overfitting inherent in single decision trees, providing a robust mechanism to model the complex, non-linear boundaries that define music popularity.

vi. How were PCA and clustering applied on your data? What method worked best for your data and why was it good for the problem you were addressing? 

PCA allowed us to compress the features from 109 to 76 componenets, vastly reducing the dimensionality of our data, which allowed us to address the high dimensionality issue due to one hot encoding track_genre.

### Neural Networks

Please refer to the main document for the application of a neural network.

### Hyperparameter Tuning

Please refer to the main document for hyperparameter tuning.

2. Code in Jupyter Notebooks- 50 points

(a) Please have a single notebook that corresponds to the data analysis pipeline to run your entire project as described in your main document. Include documentation in the notebook to explain your work. Please think of this notebook as the main code that you would share in your portfolio. 

(b) For the remaining analysis discussed in your appendix, please upload corresponding code. If the code from your corresponding check-in that you have submitted has not changed, you can mention that the code was already submitted as part of a check-in. 

(c) You may submit the notebooks as a PDF on Gradescope and also create a GitHub repository for your project. (A GitHub repository is easier to share your work, but it is not absolutely necessary for the project.)
