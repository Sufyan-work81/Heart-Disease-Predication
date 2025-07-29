# Heart Disease Prediction Project

## Overview
This project focuses on developing a machine learning model to predict heart disease risk using clinical data. The model analyzes various health indicators like age, cholesterol levels, blood pressure, and other medical parameters to assess cardiovascular health status. This implementation provides a complete workflow from exploratory data analysis to predictive modeling.

![Heart Disease Prediction](https://via.placeholder.com/800x400?text=Heart+Health+Analysis) *Example visualization placeholder*

## Key Features
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Multiple ML Models**: Comparison of various classification algorithms
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding
- **Model Evaluation**: Performance metrics including accuracy, confusion matrix, and classification reports
- **Feature Analysis**: Identification of key heart disease indicators

## Dataset
The project uses the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) containing 14 medical attributes:

1. Age
2. Sex 
3. Chest pain type
4. Resting blood pressure
5. Serum cholesterol
6. Fasting blood sugar
7. Resting electrocardiographic results
8. Maximum heart rate achieved
9. Exercise-induced angina
10. ST depression induced by exercise
11. Slope of peak exercise ST segment
12. Number of major vessels
13. Thalassemia
14. Target variable (heart disease presence)

The dataset contains 920 patient records with some missing values that are handled during preprocessing.

## Technical Approach
```python
# Example code snippet from the project
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Preprocess data
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate performance
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
```

## Dependencies
- Python 3.7+
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

Install requirements:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Run the Jupyter notebook:
```bash
jupyter notebook Heart_Disease_Prediction_Project.ipynb
```

3. Follow the notebook steps:
- Data loading and exploration
- Data preprocessing
- Model training and evaluation
- Results interpretation

## Results
The best performing model achieved **92% accuracy** with the following metrics:

| Metric        | Score    |
|---------------|----------|
| Accuracy      | 0.92     |
| Precision     | 0.91     |
| Recall        | 0.93     |
| F1-Score      | 0.92     |

Feature importance analysis revealed that maximum heart rate achieved, ST depression, and number of major vessels were the most significant predictors.

## Contributing
Contributions are welcome! Please open an issue first to discuss proposed changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Dataset provided by UCI Machine Learning Repository
- Special thanks to the Cleveland Clinic Foundation for data collection
