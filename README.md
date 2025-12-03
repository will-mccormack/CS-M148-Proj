(a) Main document: 
Your main document can be a readme explaining:

i. the data set your team used 

We used the Spotify Prediction Dataset.

ii. the overview of the problem your team addressed 

We investigated and evaulated various statistical methods to determine what characteristics of songs found on Spotify represent predictors of popularity, measured by Spotify's Popularity score, a 1-100 label assigned to every track in Spotify's database.

iii. the key methodology that worked to address the problem with explanations as to why

- 

iv. results including cross-validation used and evaluation metrics and conclusions such as why you chose the key method and its limitations 

v. how to use the code for your project on the data set. 


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
