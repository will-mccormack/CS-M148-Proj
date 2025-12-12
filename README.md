## Dataset Description [FINAL]

For our project, we selected the Spotify Prediction Dataset, sourced from Spotify’s Web API. This dataset is vast and high-dimensional: it contains 114,000 tracks across 125 genres with 20 features total. Each track has its own row and represents an observational unit. The most relevant columns consist of audio features, including but not limited to danceability, energy, loudness, and tempo, all of which are continuous variables scored from 0-100. Our response variable is Popularity, which Spotify computes algorithmically per track and is primarily based on number of plays and how recent those plays were (also measured on a 0-100 scale). Other columns provide information about track metadata, such as track_id, artists, duration_ms. Various adjustments were made to the original dataset for usability purposes in our analysis, all of which are detailed in the following sections.

## Problem Overview [FINAL]

Our primary goal for this project is to develop a model that can predict track popularity, using audio features as our predictor variables. This task is highly complex, as there are dozens of factors that contribute to a track’s popularity, many of which are not included in the dataset (for example, the extent to which a song is promoted prior to release would likely have influence on its popularity; our analysis won’t be able to capture this). Given the limiting nature of solely relying on genre and audio features, the objective becomes identifying data variables that demonstrate a consistent association with popularity. We are also interested to see how different modeling approaches handle this noisy setting.


## Key Methodology [UPDATES NEEDED]

Throughout our analysis, we explored several models of varying complexity and type with the goal of ultimately cross-referencing key model performance metrics across model types to identify the highest performing model. [Include regression info here]. Our Ridge regression model had the most predictive power/strongest interpretability across our linear models.

[Info needed on other models not referenced here]

We also trained a feedforward neural network to capture nonlinear relationships between audio features and popularity. We applied PCA to scaled features before neural network training to mitigate the effects of the data’s inherent high-dimensionality as well as the dimensionality increase caused by genre encoding. We trained the neural network using a validation set with early stopping and performed hyperparameter tuning to determine an optimal learning rate. 


(iv) Results [UPDATES NEEDED]

Models were evaluated using a consistent training–validation framework. The mean predictor and OLS regression showed limited ability to explain variation in popularity on held-out data. Ridge regression modestly reduced validation error and produced coefficients that were more stable across folds, making it suitable for interpretation.

The neural network achieved the lowest validation RMSE and highest R-squared among the models considered, indicating that nonlinear interactions and genre effects contribute additional predictive signal. However, overall R-squared values remained moderate, reflecting the inherent limitations of predicting popularity from audio features alone. For this reason, Ridge regression is used primarily for interpretability, while the neural network is treated as the strongest predictive model within the scope of the dataset.

(v) How to Use the Code [UPDATES NEEDED]

All project code is available in the GitHub repository and can be run through Jupyter. The main workflow is contained in Final_Project_Main.ipynb, which loads the prepared datasets, applies preprocessing steps, and fits the primary models including Ridge regression and the neural network. Running this notebook from start to finish reproduces the results and evaluation metrics reported here. Additional notebooks document exploratory analyses and intermediate experiments but are not required to reproduce the final results.


(b) Appendix:

(i) Exploratory Data Analysis [UPDATES NEEDED]

Exploratory data analysis was used to examine feature distributions and their relationships with popularity. Histograms and summary statistics revealed large differences in scale and variability across audio features, motivating feature standardization. Exploration of popularity showed weak linear relationships with individual features, suggesting that predictive signal is distributed across many variables and may be nonlinear.

Structural analysis revealed that many tracks appeared multiple times with identical audio features but different genre labels, introducing redundancy. The combination of numerous continuous features and categorical genre indicators also implied a high-dimensional and potentially correlated feature space. These observations motivated later preprocessing decisions including duplicate consolidation, dimensionality reduction, and careful data splitting. Analysis of the explicit label was conducted for auxiliary classification experiments, but the primary focus remained on popularity prediction.

(ii) Data Pre-processing and Feature Engineering [UPDATES NEEDED]

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
