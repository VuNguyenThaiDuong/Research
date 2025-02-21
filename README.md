# Data Preprocessing for Machine Learning Models

This project provides a class-based approach to preprocess datasets for machine learning tasks. It handles tasks such as data cleaning, encoding categorical variables, splitting into training and testing sets, and applying various scaling techniques.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

The `DataPreprocessor` class provides methods for:
1. Loading raw data from CSV files.
2. Dropping unnecessary columns.
3. Handling missing values by filling with the mean.
4. Encoding categorical features.
5. Splitting the dataset into training and test sets.
6. Applying various scaling techniques such as:
   - StandardScaler
   - MinMaxScaler
   - Normalizer
   - QuantileTransformer
7. Saving the preprocessed data into CSV files for further use.

## Requirements

To run this project, you need to install the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib
