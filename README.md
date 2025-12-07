(i) Dataset Description

Our analysis uses the Spotify Tracks Dataset, which contains audio features, metadata, and popularity metrics for a large collection of songs on the Spotify platform. The data include numerical descriptors such as danceability, energy, loudness, speechiness, instrumentalness, valence, and tempo, as well as categorical variables including genre and explicitness. The response of interest is the Spotify Popularity score, an integer from 0 to 100. We removed uninformative identifiers, resolved genre inconsistencies, standardized numeric features when appropriate, and created training and validation splits used across all modeling tasks.

(ii) Problem Overview

The goal of the project is to understand and predict track popularity using audio characteristics and metadata as predictors. Because popularity is shaped by both measurable musical features and external factors that are not present in the dataset, the signal available for prediction is limited. Our objective is to identify features that show consistent association with popularity and to develop models that generalize reasonably well to unseen tracks.

(iii) Key Methodology

We structured the modeling pipeline around a sequence of baseline and increasingly flexible models. Simple references such as a mean predictor and ordinary least squares regression provided initial benchmarks for explainable variance. OLS fit the training data but failed to generalize, so we focused on regularized linear regression, using Ridge as our primary interpretable model for assessing directional associations between features and popularity. Alongside this, we trained a feedforward neural network on PCA-reduced audio and genre features to capture nonlinear structure. PCA was used to compress the feature space while retaining most of the variance, and the network was tuned using a validation set with early stopping. These two models serve different roles in the analysis: Ridge provides a stable linear summary of associations, and the neural network is used for improved predictive performance.

(iv) Results

Model evaluation relied on a consistent train–validation framework, with cross-validation used to select the Ridge penalty and to guide neural-network tuning. OLS and the mean predictor showed little ability to explain variation in popularity on held-out data. Ridge reduced validation error modestly and produced coefficients that were stable across folds, making it suitable for interpreting feature contributions. The neural network achieved the lowest validation RMSE and the highest R-squared among all models, indicating that nonlinear interactions and genre effects contribute additional predictive signal. Despite this improvement, overall R-squared remained moderate, which reflects the limited amount of information about popularity contained in the available features. For this reason, Ridge is used for interpretation, and the neural network is treated as the strongest predictive model within the constraints of the dataset.

(v) How to Use the Code

All project code is available in the GitHub repository and can be run directly through Jupyter. The main workflow is contained in Final_Project_Main.ipynb, which loads the prepared training and validation sets, applies the preprocessing steps used throughout the project, and fits the core models including the Ridge regression and neural network. Running this notebook from start to finish reproduces all primary results, figures, and evaluation metrics referenced in the report. Additional notebooks in the repository document exploratory analyses and individual method experiments, but they are not required to regenerate the final outcomes. Standard course-package installations are sufficient for running the code, and all data files needed for analysis are included in the repository.


(b) Appendix: In the appendix you will discuss the methods that you applied to reach your final project goal from the check-ins. You should use this as a check-list to make sure you discuss all of these aspects of your work either in your main report or appendix. If a method is the key method in the main document, you can just mention that it is discussed in the main document. 

i. Explain the exploratory data analysis that you conducted. What was done to visualize your data and split your data for training and testing? 

EDA Summary
- Loaded dataset from HuggingFace.
- Plotted genre-level mean popularity to understand group differences.
- Examined outlier genres such as “comedy.”
- Inspected distribution of features and correlations using visualization tools.
- Created a train–test split (80/20) for unbiased evaluation.

ii. What data pre-processing and feature engineering (or data augmentation) did you complete on your project? 

Pre-Processing & Feature Engineering Summary
- Encoded categorical variables (e.g., genre).
- Removed low-frequency genres and irrelevant text features.
- Standardized numerical acoustic features.
- Dropped or imputed missing values.
- Optionally created a binary popularity label for classification.
- Prepared cleaned matrices X_train, X_test, y_train, y_test for modeling.

iii. How was regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed? 

iv. How was logistic regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed? v. How were KNN, decision trees, or random forest used for classification on your data? What method worked best for your data and why was it good for the problem you were addressing? 

vi. How were PCA and clustering applied on your data? What method worked best for your data and why was it good for the problem you were addressing? 

vii. Explain how your project attempted to use a neural network on the data and the results of that attempt.

viii. Give examples of hyperparameter tuning that you applied in preparing your project and how you chose the best parameters for models.


2. Code in Jupyter Notebooks- 50 points

(a) Please have a single notebook that corresponds to the data analysis pipeline to run your entire project as described in your main document. Include documentation in the notebook to explain your work. Please think of this notebook as the main code that you would share in your portfolio. 

(b) For the remaining analysis discussed in your appendix, please upload corresponding code. If the code from your corresponding check-in that you have submitted has not changed, you can mention that the code was already submitted as part of a check-in. 

(c) You may submit the notebooks as a PDF on Gradescope and also create a GitHub repository for your project. (A GitHub repository is easier to share your work, but it is not absolutely necessary for the project.)
