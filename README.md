# California Housing Price Prediction

A from-scratch implementation of **Linear Regression** for predicting California housing prices.
This project focuses on understanding the mathematical and machine learning workflow behind linear regression rather than relying entirely on high-level machine learning libraries.

The project includes:

* Data preprocessing
* Exploratory data analysis
* Correlation analysis
* Feature engineering
* Data visualization
* Feature scaling
* Linear Regression using the **Normal Equation**
* Linear Regression using **Gradient Descent**
* Model prediction
* RMSE-based evaluation

---

## Project Overview

The objective of this project is to predict the **median house value** from demographic, geographic, and housing-related features.

The dataset contains information such as:

* Longitude
* Latitude
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Median income
* Median house value
* Ocean proximity

The original dataset is loaded from:

```text
dataset/california_housing_dataset.csv
```

The dataset contains 10 main features after excluding the index column, including the target variable `median_house_value`.

---

## Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Correlation Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Log Transformation
     │
     ▼
Train / Test Split
     │
     ▼
Standardization
     │
     ├───────────────┐
     ▼               ▼
Normal Equation   Gradient Descent
     │               │
     └───────┬───────┘
             ▼
          Prediction
             │
             ▼
        RMSE Evaluation
```

---

## 1. Data Preprocessing

### Removing high-value samples

Housing samples with:

```python
median_house_value >= 500000
```

are removed from the dataset.

```python
df_loc = df[df['median_house_value'].astype(int) < 500000].copy()
```

After this filtering step, the working dataset contains **19,648 samples**.

### Handling Missing Values

The dataset contains missing values in `total_bedrooms`.

The missing values are replaced with the median:

```python
df_loc['total_bedrooms'] = (
    df_loc['total_bedrooms']
    .fillna(df_loc['total_bedrooms'].median())
)
```

After imputation, the dataset contains no missing values.

---

## 2. Exploratory Data Analysis

Descriptive statistics are calculated to understand:

* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

The project also investigates the variability and skewness of numerical features before applying transformations.

---

## 3. Correlation Analysis

A correlation matrix is constructed for numerical variables.

```python
corr_df = df_loc.drop(['ocean_proximity'], axis=1)

sns.heatmap(
    corr_df.corr(),
    annot=True,
    cmap='mako'
)
```

This is used to examine relationships between the input features and `median_house_value`, as well as correlations between features themselves.

---

## 4. Feature Engineering

Several ratio-based features are created to represent relationships between housing characteristics.

### Rooms per Household

```text
total_rooms_per_household
= total_rooms / households
```

### Bedrooms per Household

```text
total_bedrooms_per_household
= total_bedrooms / households
```

### Population per Household

```text
population_per_household
= total_bedrooms / households
```

### Bedrooms-to-Rooms Ratio

```text
total_bedrooms_per_total_rooms
= total_bedrooms / total_rooms
```

These engineered features provide alternative representations of the original housing variables.

---

## 5. Log Transformation

Log transformations are applied to several skewed variables using:

```python
np.log1p(...)
```

The project creates transformed features including:

```text
log_median_income
log_households
log_rooms_per_household
log_population_per_household
log_median_house_value
log_total_bedrooms_per_total_rooms
```

The target variable `median_house_value` is therefore modeled in log space and transformed back to the original scale with:

```python
np.expm1(...)
```

during prediction.

---

## 6. Feature Selection

After feature engineering, several original variables are removed:

```python
df_loc = df_loc.drop(
    [
        'total_rooms',
        'total_bedrooms',
        'population',
        'total_bedrooms_per_household'
    ],
    axis=1
)
```

The resulting dataset contains the selected original variables together with engineered and logarithmically transformed features.

---

## 7. Data Scaling

The numerical input features are standardized manually instead of using a machine learning library.

For the training set:

```text
X_scaled = (X - mean) / standard_deviation
```

The mean and standard deviation are calculated **only from the training data**.

The same training statistics are then applied to the test data:

```python
X_test_scaled = (X_test_raw - X_train_mean) / X_train_std
```

An intercept column is then added:

```python
X_train = np.hstack(
    (np.ones((X_train_scaled.shape[0], 1)), X_train_scaled)
)
```

This allows the regression model to learn an intercept term.

---

# Linear Regression

The project implements Linear Regression using two different optimization approaches.

---

## 8. Method 1 — Normal Equation

The first implementation solves Linear Regression analytically using the **Normal Equation**.

The coefficient vector is calculated as:

```python
theta = np.linalg.inv(
    X_train.T @ X_train
) @ X_train.T @ y_train_raw
```

Mathematically:

```text
θ = (XᵀX)⁻¹Xᵀy
```

This approach obtains the optimal parameters directly without iterative optimization.

The model then predicts:

```python
y_pred = X_test @ theta
```

The predicted logarithmic values are transformed back into the original house-price scale:

```python
y_pred_real_test = np.expm1(y_pred)
```

---

## 9. Method 2 — Gradient Descent

The second implementation trains Linear Regression using **Gradient Descent from scratch**.

The parameter vector is initialized to zero:

```python
theta = np.zeros((n_features, 1))
```

Training parameters:

```python
learning_rate = 0.05
epochs = 50000
```

During each iteration:

```python
y_pred_train = X_train @ theta

error = y_train_raw - y_pred_train

grad = -(1 / n_samples_train) * (
    X_train.T @ error
)

theta = theta - learning_rate * grad
```

The model therefore iteratively updates `theta` to minimize the training loss.

The training loss is calculated as:

```text
Loss = 1 / (2n) Σ(y - ŷ)²
```

The implementation records the training loss during optimization to monitor convergence.

---

## 10. Prediction

After training, the model predicts the test set:

```python
y_pred_log = X_test @ theta
```

Because the target was transformed using `log1p`, the prediction is converted back using:

```python
y_pred_real = np.expm1(y_pred_log)
```

The actual target values are transformed back in the same way:

```python
y_test_real = np.expm1(y_test_raw)
```

---

## 11. Model Evaluation

The primary evaluation metric used in the project is **Root Mean Squared Error (RMSE)**.

```python
test_rmse = np.sqrt(
    np.mean(
        (y_pred_real - y_test_real) ** 2
    )
)
```

RMSE measures the typical magnitude of prediction errors in the original house-price scale.

A lower RMSE indicates smaller prediction errors.

The final result is printed in dollar format:

```python
print(f"RMSE: ${test_rmse:,.2f}")
```

---

## 12. Model Parameters

The Gradient Descent implementation also displays the learned coefficient for each feature:

```text
Feature Name                     | Coefficient (Theta)
-------------------------------------------------------
Intercept (b)                   | ...
longitude                       | ...
latitude                        | ...
...
```

This makes it possible to inspect how the trained Linear Regression model assigns coefficients to the input features.

---

## 13. Log-Likelihood Experiment

A separate notebook in this project demonstrates the relationship between **Maximum Likelihood Estimation (MLE)**, log-likelihood, and optimization.

A normal probability density function is implemented:

```python
def normal_pdf(x, mu, sigma):
    return (
        1.0 / (sigma * np.sqrt(2 * np.pi))
    ) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
```

The log-likelihood is calculated using the sum of log probability densities:

```python
def compute_log_likelihood(mu_candidate, data, sigma):
    log_pdfs = (
        -0.5 * np.log(2 * np.pi * sigma**2)
        - 0.5 * ((data - mu_candidate) / sigma)**2
    )

    return np.sum(log_pdfs)
```

The notebook generates an observed dataset with:

```text
n = 500
μ = 5.0
σ = 1.0
```

and visualizes the resulting distribution.

The project then uses **Gradient Ascent** to find the parameter value that maximizes the log-likelihood. The optimization path is visualized against the log-likelihood surface.

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn

The notebooks implement important parts of the machine learning pipeline directly with NumPy rather than depending on a high-level regression implementation.

---

## Project Structure

```text
.
├── dataset/
│   └── california_housing_dataset.csv
│
├── notebooks/
│   ├── ...
│   └── ...
│
└── README.md
```

---

## Key Concepts Demonstrated

This project covers several fundamental concepts in Machine Learning and Statistics:

* Data preprocessing
* Missing-value imputation
* Exploratory Data Analysis
* Correlation matrices
* Feature engineering
* Ratio features
* Log transformation
* Standardization
* Linear Regression
* Normal Equation
* Gradient Descent
* Mean Squared Error
* Root Mean Squared Error
* Maximum Likelihood Estimation
* Log-Likelihood
* Gradient Ascent
* Model parameter interpretation

---

## Purpose

The main purpose of this project is to understand the mathematical foundations behind machine learning algorithms by implementing the core procedures manually.

Rather than treating Linear Regression as a black-box model, the project demonstrates how the model parameters can be obtained through both:

```text
Closed-form Optimization
        ↓
Normal Equation
```

and

```text
Iterative Optimization
        ↓
Gradient Descent
```

The additional log-likelihood notebook provides a separate visualization of how likelihood-based optimization works and how gradient-based optimization can be used to locate an optimum.
