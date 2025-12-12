# Spotify Popularity Prediction Project

## 1. Main Report

### Dataset Description

For our project, we selected the Spotify Prediction Dataset, sourced from Spotify’s Web API. This dataset is vast and high-dimensional: it contains 114,000 tracks across 125 genres with 20 features total. Each track has its own row and represents an observational unit. The most relevant columns consist of audio features, including but not limited to danceability, energy, loudness, and tempo, all of which are continuous variables scored from 0-100. Our response variable is "popularity," which Spotify computes algorithmically per track and is primarily based on the number of plays and how recent those plays were (also measured on a 0-100 scale).

### Problem Overview

Our primary goal for this project is to develop a model that can predict track popularity, using audio features as our predictor variables. This task is highly complex, as there are dozens of factors that contribute to a track’s popularity, many of which are not included in the dataset (for example, the extent to which a song is promoted prior to release would likely have influence on its popularity; our analysis won’t be able to capture this). Given the limiting nature of solely relying on genre and audio features, the objective became to determine how much of a song's success is intrinsic to the audio itself.

### Key Methodology: Neural Network Regression with PCA

To address the non-linear relationship between audio features and popularity, we implemented a Deep Neural Network (DNN) regressor. While linear models (like Ridge Regression) failed to capture complex interactions between genres and musical attributes, the Neural Network's depth allowed it to model these nuances.

#### Architecture: We utilized a "Funnel" architecture, starting with a wide input layer (256 neurons) to capture feature interactions, then geometrically compressing the data (128 -> 64 -> 32) to a single output neuron.

#### Feature Reduction: Due to the high dimensionality created by One-Hot Encoding 125 genres, we applied Principal Component Analysis (PCA), retaining components that explained 90% of the variance. This reduced noise and improved training speed.

#### Regularization: To prevent overfitting (a major challenge in earlier iterations), we employed Batch Normalization, Dropout (30%), and the AdamW optimizer with weight decay.

### Results and Evaluation

We evaluated our model using 5-Fold Cross-Validation to ensure stability.

RMSE (Root Mean Squared Error): ~11.30. On average, our prediction is within 11 points of the true popularity score (0-100 scale).

$R^2$ Score: 0.588. Our model explains approximately 59% of the variance in popularity.

####The "Cold Start" Discovery:
Initially, our model struggled with an $R^2$ of only 0.22. Exploratory Data Analysis revealed a "Cold Start" problem: thousands of songs with high-quality audio features had 0 popularity (likely due to being new releases or duplicate entries). Filtering these rows was the single most effective optimization, jumping our accuracy from 22% to 59%.

### Conclusion:
While audio features are significant predictors (explaining ~60% of popularity), the remaining 40% of variance is likely attributable to external factors absent from the dataset, such as artist reputation, marketing budget, and release date.

How to Use the Code

Open Final_Project_Main_v8.ipynb in Google Colab (or any Jupyter environment).

Upload the dataset files (train.csv, test.csv) to the runtime environment.

Run all cells. The notebook pipeline will:

Load and clean the data (removing duplicates and zero-popularity outliers).

Perform One-Hot Encoding and PCA.

Train the Neural Network using 5-Fold Cross-Validation.

Generate Loss curves and Prediction scatter plots.

2. Appendix

i. Exploratory Data Analysis (EDA)

We conducted extensive EDA to understand the distribution of our target variable. We plotted histograms of popularity scores, which revealed a non-normal distribution with a massive spike at 0 (the "Cold Start" problem) and a bell curve centered around 45. This insight directly led to our data cleaning strategy of filtering out zero-popularity outliers.

ii. Data Pre-processing

Cleaning: We converted boolean columns (explicit) to integers and consolidated duplicate tracks by grouping by audio features and taking the maximum popularity score.

Encoding: We applied One-Hot Encoding to the track_genre column, creating over 100 new binary features.

Scaling: We used StandardScaler to normalize all continuous audio features, which is critical for Neural Network convergence.

iii. Regression Analysis

We initially applied Linear and Ridge Regression as baselines. These models yielded low $R^2$ scores (< 0.30), indicating that the relationship between specific audio features (like loudness or tempo) and popularity is not linear. This failure motivated the use of non-linear models like Random Forest and Neural Networks.

iv. Logistic Regression Analysis

We applied Logistic Regression as a baseline for classification (Popular vs Not Popular). We utilized class_weight='balanced' to handle the dataset imbalance (unpopular songs outnumbered hits). The model achieved an AUC of ~0.69. This moderate score confirmed that while linear decision boundaries could capture some signal, they were insufficient for high-accuracy predictions.

v. Classification (KNN, Trees, Random Forest)

In addition to regression, we attempted to classify songs using non-linear models:

KNN struggled due to the high dimensionality of the genre data (the "curse of dimensionality").

Random Forest performed best for classification. Unlike linear models, it naturally handled non-linear feature interactions (e.g., high "energy" correlates with popularity in Rock but not in Classical). It achieved higher accuracy than single decision trees by reducing variance through bagging.

vi. PCA and Clustering

PCA allowed us to compress the features from ~130 to ~80 components (retaining 90% variance), vastly reducing the dimensionality of our data. This addressed the sparsity issue introduced by One-Hot Encoding track_genre and significantly sped up Neural Network training.

vii. Neural Networks

(Detailed in the Main Report Section above).

viii. Hyperparameter Tuning

We applied Grid Search and dynamic scheduling to optimize the Neural Network:

Learning Rate: We tested rates of 0.01, 0.001, and 0.0001. We found 0.001 provided the best balance of convergence speed and stability.

Scheduler: We implemented a ReduceLROnPlateau scheduler. If the validation loss stalled for 5 epochs, the learning rate was automatically halved. This allowed the model to "settle" into a deeper minimum in later epochs.
