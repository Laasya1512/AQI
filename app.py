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
        raise FileNotFoundError("The dataset file 'data1.csv' was not found.")

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
    data.dropna(subset=["observationDateTime"], inplace=True)

    # Fill numeric NaNs with median
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)

    # Avoid scaling airQualityIndex
    aqi_col = 'airQualityIndex'
    cols_to_scale = [col for col in numeric_cols if col != aqi_col]

    # Normalize and standardize selected numeric columns
    scaler = MinMaxScaler()
    standardizer = StandardScaler()
    scaled_data = scaler.fit_transform(data[cols_to_scale])
    standardized_data = standardizer.fit_transform(scaled_data)
    data[cols_to_scale] = pd.DataFrame(standardized_data, columns=cols_to_scale)

    # Remove outliers using Z-score (excluding AQI)
    z_scores = np.abs(stats.zscore(data[cols_to_scale]))
    data = data[(z_scores < 3).all(axis=1)]

    return data

# Preprocess once at server startup
processed_data = preprocess_data()

@app.route('/')
def home():
    return render_template('index.html')  # Make sure templates/index.html exists

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
            'aqi_values': hourly_data['airQualityIndex'].round(2).tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
