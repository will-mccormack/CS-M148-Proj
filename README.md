(i) Dataset Description

Our analysis uses a Spotify tracks dataset containing audio features, genre information, and track popularity scores. The dataset includes continuous audio descriptors such as danceability, energy, loudness, speechiness, instrumentalness, valence, and tempo, along with categorical metadata including genre and explicitness. The primary response variable is Spotify popularity, measured on a 0–100 scale.

Prior to modeling, uninformative or redundant identifiers were removed, numeric features were standardized when required, and categorical genre information was encoded for use in downstream models. Tracks with identical audio features but differing genre labels were consolidated to reduce redundancy. Consistent training and validation splits were used across all modeling tasks.

(ii) Problem Overview

The goal of this project is to predict track popularity using available audio characteristics and metadata. Popularity is influenced by many external factors not captured in the dataset, so the predictive signal available from audio features alone is limited. As a result, the objective is not to perfectly predict popularity, but to identify features that show consistent association with popularity and to evaluate how different modeling approaches handle this noisy setting.

(iii) Key Methodology

We structured the analysis around a progression from simple baselines to more flexible models. A mean predictor and ordinary least squares regression were used as reference points to assess how much variance could be explained without regularization. Because OLS fit the training data but generalized poorly, we focused on Ridge regression as the primary linear model, using it to obtain stable coefficient estimates and assess directional feature associations.

In parallel, we trained a feedforward neural network to capture nonlinear relationships between audio features and popularity. Due to the high dimensionality introduced by genre encoding, PCA was applied to scaled features before neural network training. The neural network was trained using a validation set with early stopping, and limited hyperparameter tuning was performed to select a learning rate. Ridge and the neural network serve complementary roles: Ridge emphasizes interpretability, while the neural network emphasizes predictive performance.

(iv) Results

Models were evaluated using a consistent training–validation framework. The mean predictor and OLS regression showed limited ability to explain variation in popularity on held-out data. Ridge regression modestly reduced validation error and produced coefficients that were more stable across folds, making it suitable for interpretation.

The neural network achieved the lowest validation RMSE and highest R-squared among the models considered, indicating that nonlinear interactions and genre effects contribute additional predictive signal. However, overall R-squared values remained moderate, reflecting the inherent limitations of predicting popularity from audio features alone. For this reason, Ridge regression is used primarily for interpretability, while the neural network is treated as the strongest predictive model within the scope of the dataset.

(v) How to Use the Code

All project code is available in the GitHub repository and can be run through Jupyter. The main workflow is contained in Final_Project_Main.ipynb, which loads the prepared datasets, applies preprocessing steps, and fits the primary models including Ridge regression and the neural network. Running this notebook from start to finish reproduces the results and evaluation metrics reported here. Additional notebooks document exploratory analyses and intermediate experiments but are not required to reproduce the final results.


(b) Appendix:

(i) Exploratory Data Analysis

Exploratory data analysis was used to examine feature distributions and their relationships with popularity. Histograms and summary statistics revealed large differences in scale and variability across audio features, motivating feature standardization. Exploration of popularity showed weak linear relationships with individual features, suggesting that predictive signal is distributed across many variables and may be nonlinear.

Structural analysis revealed that many tracks appeared multiple times with identical audio features but different genre labels, introducing redundancy. The combination of numerous continuous features and categorical genre indicators also implied a high-dimensional and potentially correlated feature space. These observations motivated later preprocessing decisions including duplicate consolidation, dimensionality reduction, and careful data splitting. Analysis of the explicit label was conducted for auxiliary classification experiments, but the primary focus remained on popularity prediction.

(ii) Data Pre-processing and Feature Engineering

Preprocessing steps were informed by exploratory findings and model requirements. Low-signal metadata columns were removed, and categorical genre information was encoded using one-hot encoding. Duplicate tracks with identical audio features were grouped together and their genre indicators merged to reduce redundancy while preserving multi-genre information.

All continuous features were standardized prior to modeling. PCA was applied after scaling and fit only on the training data to reduce dimensionality and mitigate multicollinearity, particularly for neural network training. The resulting feature representations were used consistently across training and validation sets. No synthetic data augmentation was performed; feature engineering focused on improving representation quality and stability.

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
