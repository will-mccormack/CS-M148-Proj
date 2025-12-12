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

(i) Exploratory Data Analysis

We conducted exploratory data analysis to understand the distributions of Spotify audio features and their relationships with the target variables, explicit and popularity. We first examined the distributions of continuous audio features such as danceability, energy, loudness, tempo, and valence using histograms and summary statistics. These plots showed substantial differences in scale and variance across features, as well as skewed distributions for some variables, indicating that feature scaling would be necessary for downstream modeling.

We then explored the target variables directly. For the classification task, we inspected the distribution of the binary explicit label and observed moderate class imbalance, which informed the choice of evaluation metrics beyond accuracy. For the regression task, we examined the distribution of popularity scores and their relationships with selected audio features using scatter plots, which suggested weak but non-linear relationships rather than a single dominant predictor.

We also analyzed structural properties of the dataset. Tracks often appeared multiple times with identical audio features but different genre labels, indicating redundancy in the raw data. In addition, the combination of many audio features with categorical genre information implied a high-dimensional feature space with potential multicollinearity. These observations motivated later preprocessing steps including feature selection, dimensionality reduction, and careful train–test splitting. For model evaluation, the data was split into training and validation or test sets, and cross-validation was used for the classification task to assess generalization performance.

(ii) Data Pre-processing and Feature Engineering

Preprocessing was guided by the issues identified during EDA and the requirements of the models used. We first cleaned the data by removing low-signal or redundant metadata columns such as key and time_signature. The explicit label was converted to a binary integer representation to ensure compatibility with classification models.

Categorical genre information was encoded using one-hot encoding, which increased the dimensionality of the feature space. To address redundancy, duplicate tracks with identical audio features were grouped together and their genre indicators were merged, preserving multi-genre information while reducing repeated observations and limiting potential data leakage across splits.

All continuous features were scaled prior to modeling to account for differences in feature magnitude. This step was necessary for distance-based methods, principal component analysis (PCA), and neural network training. PCA was applied after scaling and fit only on the training data to reduce dimensionality and mitigate multicollinearity among audio features. The explained variance ratios were examined to confirm that the transformed feature space retained most of the original variance.

Finally, the processed features were used to construct consistent training and testing datasets for both classical machine learning models and neural networks. No synthetic data augmentation was performed; feature engineering focused on improving representation quality and model stability rather than expanding the dataset.

***TODO:***

iii. How was regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed? 

iv. How was logistic regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed? v. How were KNN, decision trees, or random forest used for classification on your data? What method worked best for your data and why was it good for the problem you were addressing? 

vi. How were PCA and clustering applied on your data? What method worked best for your data and why was it good for the problem you were addressing? 

vii. Explain how your project attempted to use a neural network on the data and the results of that attempt.

viii. Give examples of hyperparameter tuning that you applied in preparing your project and how you chose the best parameters for models.


2. Code in Jupyter Notebooks- 50 points

(a) Please have a single notebook that corresponds to the data analysis pipeline to run your entire project as described in your main document. Include documentation in the notebook to explain your work. Please think of this notebook as the main code that you would share in your portfolio. 

(b) For the remaining analysis discussed in your appendix, please upload corresponding code. If the code from your corresponding check-in that you have submitted has not changed, you can mention that the code was already submitted as part of a check-in. 

(c) You may submit the notebooks as a PDF on Gradescope and also create a GitHub repository for your project. (A GitHub repository is easier to share your work, but it is not absolutely necessary for the project.)
