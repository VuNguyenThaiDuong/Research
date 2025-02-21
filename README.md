# Data Preprocessing for Machine Learning Models

This project provides a class-based approach to preprocess datasets for machine learning tasks. It handles tasks such as data cleaning, encoding categorical variables, splitting into training and testing sets, and applying various scaling techniques.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example Code](#example-code)
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

## Licenses
---

### Mô tả các phần:

- **Example Code** đã được đưa xuống dưới phần **Usage** trong README.md, giải thích chi tiết từng bước về cách khai báo và sử dụng lớp `DataPreprocessor` cho một tệp dữ liệu mới.
- Trong phần **Example Code**, bạn sẽ thấy cách thay đổi các tham số như tên tệp, cột mục tiêu, các cột cần loại bỏ, và kỹ thuật chuẩn hóa.

## Usage

Sau khi bạn đã cài đặt xong các yêu cầu và chuẩn bị môi trường, bạn có thể sử dụng lớp `DataPreprocessor` để tiền xử lý các tệp dữ liệu của mình.

### Bước 1: Đặt các tệp dữ liệu thô

Đảm bảo rằng các tệp dữ liệu thô của bạn được đặt trong một thư mục có tên là `Raw Data` (hoặc cập nhật lại đường dẫn trong mã nguồn nếu cần). Bạn có thể thêm các tệp `.csv` mà bạn muốn tiền xử lý vào thư mục này.

### Bước 2: Cập nhật mã nguồn

Trong tệp mã nguồn, bạn cần chỉnh sửa một số tham số sau cho phù hợp với dữ liệu của bạn:

1. **Raw Data Directory**: Đặt đường dẫn đến thư mục chứa các tệp dữ liệu thô của bạn.
2. **Output Directory**: Đặt đường dẫn đến thư mục nơi bạn muốn lưu các tệp dữ liệu đã được tiền xử lý.
3. **File Paths**: Danh sách tên các tệp `.csv` bạn muốn tiền xử lý.
4. **Target Labels**: Tên của cột mục tiêu (target column) trong dữ liệu của bạn.
5. **Columns to Drop**: Các cột bạn muốn loại bỏ khỏi dữ liệu trong quá trình tiền xử lý.
6. **Scalers**: Phương pháp chuẩn hóa dữ liệu bạn muốn áp dụng (ví dụ: `StandardScaler`, `MinMaxScaler`, `Normalizer`, `QuantileTransformer`).

### Bước 3: Chạy mã nguồn

Sau khi bạn đã chỉnh sửa mã nguồn, bạn có thể chạy mã nguồn để tiến hành tiền xử lý dữ liệu bằng cách thực hiện lệnh dưới đây trong terminal hoặc command prompt:

```
# Import the class for data preprocessing
from data_preprocessor import DataPreprocessor

# Define directories
# Import class đã định nghĩa trước đó
from data_preprocessor import DataPreprocessor
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer

current_directory = os.getcwd()
raw_data_directory = os.path.join(current_directory, 'Raw Data')
output_directory = os.path.join(current_directory, 'Data')

file_paths = ['N_BaIoT_dataloader.csv', 'data_CICIoT2023.csv']
target_labels = ['Default_label', 'Default_label']
columns_to_drop = [
    ["Unnamed: 0", 'Category_label'],
    ['DHCP', 'ece_flag_number', 'Telnet', 'SMTP', 'IRC', 'Category_label']
]
scaler_names = ['StandardScaler']

# Instantiate DataPreprocessor
preprocessor = DataPreprocessor(raw_data_directory, output_directory, scaler_names)

# Preprocess files
preprocessor.preprocess_multiple_files(file_paths, target_labels, columns_to_drop)

```

