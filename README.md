# Crop Yield Prediction

## Table of Contents
1. [Crop Yield Prediction using Random Forest](#crop-yield-prediction-using-random-forest)
   - [Objective](#objective)
   - [Random Forest Algorithm](#random-forest-algorithm)
   - [Dataset](#dataset)
   - [Code](#code)
   - [Results](#results)
2. [Crop Yield Prediction using SVR, RBFNN, and BPNN](#crop-yield-prediction-using-svr-rbfnn-and-bpnn)
   - [Objective](#objective-1)
   - [Algorithms](#algorithms)
     - [Support Vector Regression (SVR)](#support-vector-regression-svr)
     - [Radial Basis Function Neural Network (RBFNN)](#radial-basis-function-neural-network-rbfnn)
     - [Back Propagation Neural Network (BPNN)](#back-propagation-neural-network-bpnn)
   - [Code](#code-1)
   - [Results](#results-1)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

---

## Crop Yield Prediction using Random Forest

### Objective
The goal of this project is to predict crop yield using the Random Forest algorithm, a widely-used machine learning model, to analyze data and generate accurate yield predictions.

### Random Forest Algorithm
Random Forest is a powerful supervised learning algorithm used for both classification and regression. It creates multiple decision trees during training and outputs the mode of classifications (for classification tasks) or mean prediction (for regression tasks) of the individual trees.

### Dataset
The dataset contains features influencing crop yield, such as environmental factors, and the target variable is the actual yield (in Q/acre). The data is stored in an Excel file.

### Code
The Python code for this project is built using:

- **pandas**: for data manipulation and preprocessing
- **scikit-learn (sklearn)**: for the Random Forest model and evaluation metrics
- **matplotlib**: for data visualization

### Results
The Random Forest model achieved the following performance metrics on the dataset:

- **R-squared score**: 92.28%
- **Mean Squared Error (MSE)**: 0.075%
- **Root Mean Squared Error (RMSE)**: 22.6

---

## Crop Yield Prediction using SVR, RBFNN, and BPNN

### Objective
The purpose of this part of the project is to predict crop yield using different machine learning algorithms, including SVR, RBFNN, and BPNN. The objective is to evaluate the performance of these algorithms and compare their accuracy.

### Algorithms

#### Support Vector Regression (SVR)
Support Vector Regression is a regression technique based on the Support Vector Machine (SVM). It aims to predict continuous values by minimizing the error margin and maximizing the margin of error tolerance.

#### Radial Basis Function Neural Network (RBFNN)
RBFNN is a neural network composed of an input layer, a hidden layer with radial basis functions as activation functions, and a linear output layer. It is commonly used for regression and classification tasks.

#### Back Propagation Neural Network (BPNN)
BPNN is a type of neural network where the error is backpropagated during training. It adjusts the weights and biases through gradient descent to minimize the loss function.

### Code
The Python code for this part of the project uses:

- **pandas**: for handling the dataset
- **scikit-learn (sklearn)**: for SVR model and evaluation
- **matplotlib**: for result visualization
- **openpyxl**: for writing output predictions into Excel

### Results
Performance metrics for each algorithm:

- **SVR**:
  - R-squared: [value]
  - MSE: 2.0065878741521795e-06
  - RMSE: [value]

- **RBFNN**:
  - R-squared: [value]
  - MSE: 4.3456013950577274e-06
  - RMSE: [value]

- **BPNN**:
  - R-squared: [value]
  - MSE: 0.0550280310073301
  - RMSE: [value]

From the results, SVR and RBFNN performed better than BPNN based on MSE. However, performance depends on the dataset and fine-tuning of hyperparameters.

---

## Usage
To use the code:

1. **Clone the repository**: Download or clone the source code.
2. **Install dependencies**: Install necessary Python libraries using pip (e.g., `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`).
3. **Run the code**: Open the Python script in your IDE or terminal.
4. **Dataset**: Ensure the dataset file is in the correct path specified in the code.
5. **Execution**: Run the script to train the models, evaluate their performance, and generate visualizations.

---

## Contributing
Contributions are welcome! If you have ideas to improve the code or find bugs, please open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.
