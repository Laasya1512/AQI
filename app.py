from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import os

app = Flask(__name__)

def preprocess_data():
    # Load the dataset
    file_path = 'data1.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError("The dataset file 'data.csv' was not found.")

    data = pd.read_csv(file_path)

    # Replace -1 with NaN
    data.replace(-1, np.nan, inplace=True)

    # Fill categorical NaNs with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Convert datetime column
    data["observationDateTime"] = pd.to_datetime(data["observationDateTime"], errors='coerce')
    data.dropna(subset=["observationDateTime"], inplace=True)  # drop rows with invalid datetime

    # Fill numeric NaNs with median
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)

    # Normalize then standardize numeric columns
    scaler = MinMaxScaler()
    standardizer = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    standardized_data = standardizer.fit_transform(scaled_data)
    data[numeric_cols] = pd.DataFrame(standardized_data, columns=numeric_cols)

    # Remove outliers using Z-score
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    data = data[(z_scores < 3).all(axis=1)]

    return data

# Preprocess data once when the server starts
processed_data = preprocess_data()

@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html exists in the templates folder

@app.route('/get_hourly_aqi', methods=['GET'])
def get_hourly_aqi():
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify({'error': 'No date provided'}), 400

    try:
        selected_datetime = pd.to_datetime(selected_date).date()
        filtered_data = processed_data[
            processed_data['observationDateTime'].dt.date == selected_datetime
        ]

        if filtered_data.empty:
            return jsonify({'error': f'No data available for {selected_date}'}), 404

        filtered_data = filtered_data.copy()
        filtered_data['hour'] = filtered_data['observationDateTime'].dt.hour
        hourly_data = filtered_data.groupby('hour')['airQualityIndex'].mean().reset_index()

        response = {
            'hours': hourly_data['hour'].tolist(),
            'aqi_values': hourly_data['airQualityIndex'].tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
