from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq


app = Flask(__name__)

# Setup Groq LLM
llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")


template = """
You're an environmental analyst. Given air quality data for {date}, summarize the AQI and pollution levels.
Include key pollutants like PM2.5, PM10, NO2, and SO2. Provide health impact insight.

Data Summary:
{summary}
"""

prompt = PromptTemplate(template=template, input_variables=["date", "summary"])
llm_chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/generate-report', methods=['GET'])
def generate_report_endpoint():
    city = request.args.get('city')
    date = request.args.get('date')
    
    if not city or not date:
        return jsonify({'error': 'City and date are required'}), 400
    
    try:
        date_obj = pd.to_datetime(date).date()
        df = processed_data[processed_data['observationDateTime'].dt.date == date_obj]
        
        if df.empty:
            return jsonify({'report': '<div class="alert alert-warning">No data available for this date and city.</div>'}), 200
        
        # Generate a simple report if LLM is not available
        summary = df.describe().to_string()
        try:
            report_text = llm_chain.run({"date": date, "summary": summary})
            # We'll format this text later
        except Exception as e:
            # Fallback if LLM fails
            avg_aqi = df['airQualityIndex'].mean()
            max_aqi = df['airQualityIndex'].max()
            min_aqi = df['airQualityIndex'].min()
            report_text = f"On {date}, the average AQI in {city} was {avg_aqi:.1f} with a maximum of {max_aqi:.1f} and minimum of {min_aqi:.1f}."
        
        # Get AQI category and color
        avg_aqi = df['airQualityIndex'].mean()
        aqi_category, aqi_color, health_implication = get_aqi_info(avg_aqi)
        
        # Get pollutant data
        pm25 = df['pm2p5.avgOverTime'].mean() if 'pm2p5.avgOverTime' in df.columns else 0
        pm10 = df['pm10.avgOverTime'].mean() if 'pm10.avgOverTime' in df.columns else 0
        no2 = df['no2.avgOverTime'].mean() if 'no2.avgOverTime' in df.columns else 0
        so2 = df['so2.avgOverTime'].mean() if 'so2.avgOverTime' in df.columns else 0
        
        # Format the HTML report
        formatted_report = f"""
        <div class="report-summary" style="background-color: {aqi_color}20; border-left: 5px solid {aqi_color}; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
            <h5 style="margin-top: 0;"><i class="fas fa-info-circle me-2"></i> Summary</h5>
            <p style="margin-bottom: 5px;">The air quality in <strong>{city}</strong> on <strong>{date}</strong> was categorized as <strong style="color: {aqi_color}">{aqi_category}</strong>.</p>
            <p style="margin-bottom: 0;">{health_implication}</p>
        </div>
        
        <div class="report-metrics">
            <h5><i class="fas fa-chart-line me-2"></i> Air Quality Metrics</h5>
            <div class="row">
                <div class="col-md-6">
                    <div class="metric-card" style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h6 style="margin-top: 0;">Air Quality Index (AQI)</h6>
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <p style="margin-bottom: 5px;"><strong>Average:</strong> {avg_aqi:.1f}</p>
                                <p style="margin-bottom: 5px;"><strong>Maximum:</strong> {df['airQualityIndex'].max():.1f}</p>
                                <p style="margin-bottom: 0;"><strong>Minimum:</strong> {df['airQualityIndex'].min():.1f}</p>
                            </div>
                            <div style="font-size: 2rem; color: {aqi_color};">
                                <i class="fas fa-wind"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="metric-card" style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <h6 style="margin-top: 0;">Key Pollutants</h6>
                        <p style="margin-bottom: 5px;"><strong>PM2.5:</strong> {pm25:.1f} μg/m³</p>
                        <p style="margin-bottom: 5px;"><strong>PM10:</strong> {pm10:.1f} μg/m³</p>
                        <p style="margin-bottom: 5px;"><strong>NO₂:</strong> {no2:.1f} μg/m³</p>
                        <p style="margin-bottom: 0;"><strong>SO₂:</strong> {so2:.1f} μg/m³</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="report-analysis">
            <h5><i class="fas fa-microscope me-2"></i> Detailed Analysis</h5>
            <p>{report_text}</p>
        </div>
        
        <div class="report-recommendations" style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h5 style="margin-top: 0;"><i class="fas fa-lightbulb me-2"></i> Recommendations</h5>
            <ul style="margin-bottom: 0; padding-left: 20px;">
                {get_recommendations(aqi_category)}
            </ul>
        </div>
        """
        
        return jsonify({'report': formatted_report})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_aqi_info(aqi_value):
    """Return AQI category, color, and health implication based on AQI value"""
    if aqi_value <= 50:
        return "Good", "#a8e6cf", "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "#dcedc1", "Air quality is acceptable; however, there may be a moderate health concern for a very small number of people."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#ffd3b6", "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
    elif aqi_value <= 200:
        return "Unhealthy", "#ffaaa5", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "#ff8b94", "Health warnings of emergency conditions. The entire population is more likely to be affected."
    else:
        return "Hazardous", "#d291bc", "Health alert: everyone may experience more serious health effects."

def get_recommendations(aqi_category):
    """Return recommendations based on AQI category"""
    if aqi_category == "Good":
        return """
        <li>Enjoy outdoor activities</li>
        <li>Keep windows open for fresh air</li>
        <li>Continue monitoring air quality</li>
        """
    elif aqi_category == "Moderate":
        return """
        <li>Sensitive individuals should consider reducing prolonged outdoor exertion</li>
        <li>Keep monitoring air quality changes throughout the day</li>
        <li>Close windows during peak traffic hours</li>
        """
    elif aqi_category == "Unhealthy for Sensitive Groups":
        return """
        <li>People with respiratory or heart disease, the elderly and children should limit prolonged outdoor exertion</li>
        <li>Consider using air purifiers indoors</li>
        <li>Keep windows closed, especially during peak pollution hours</li>
        <li>Wear masks when outdoors for extended periods</li>
        """
    elif aqi_category == "Unhealthy":
        return """
        <li>Everyone should limit outdoor exertion</li>
        <li>Use air purifiers indoors</li>
        <li>Keep all windows closed</li>
        <li>Wear N95 masks when outdoors</li>
        <li>Consider rescheduling outdoor activities</li>
        """
    elif aqi_category == "Very Unhealthy":
        return """
        <li>Avoid all outdoor activities</li>
        <li>Keep all windows and doors closed</li>
        <li>Run air purifiers continuously</li>
        <li>Wear N95 masks if you must go outside</li>
        <li>Check on elderly neighbors and those with respiratory conditions</li>
        """
    else:  # Hazardous
        return """
        <li>Stay indoors with windows and doors closed</li>
        <li>Run air purifiers on highest setting</li>
        <li>Avoid any outdoor activity</li>
        <li>Wear N95 masks if evacuation is necessary</li>
        <li>Follow local health department advisories</li>
        <li>Consider temporarily relocating if possible</li>
        """

def preprocess_data():
    file_path = 'data1.csv'
    if not os.path.exists(file_path):
        # Create dummy data for testing if file doesn't exist
        print("Warning: data1.csv not found. Creating dummy data for testing.")
        dummy_data = pd.DataFrame({
            'observationDateTime': pd.date_range(start='2021-01-01', end='2022-12-31', freq='H'),
            'airQualityIndex': np.random.randint(0, 300, size=17544),
            'pm2p5.avgOverTime': np.random.randint(0, 100, size=17544),
            'pm10.avgOverTime': np.random.randint(0, 150, size=17544),
            'no2.avgOverTime': np.random.randint(0, 80, size=17544),
            'so2.avgOverTime': np.random.randint(0, 40, size=17544),
            'co.avgOverTime': np.random.randint(0, 10, size=17544),
            'airTemperature.avgOverTime': np.random.uniform(10, 40, size=17544),
            'relativeHumidity.avgOverTime': np.random.uniform(30, 90, size=17544)
        })
        return dummy_data
        
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

    # Parameters to exclude from scaling and outlier removal
    untouched_cols = [
        'airQualityIndex',
        'pm2p5.avgOverTime',
        'pm10.avgOverTime',
        'co.avgOverTime',
        'no2.avgOverTime',
        'so2.avgOverTime'
    ]

    # Columns to normalize and standardize
    cols_to_scale = [col for col in numeric_cols if col not in untouched_cols]

    # Normalize and standardize selected numeric columns
    scaler = MinMaxScaler()
    standardizer = StandardScaler()
    scaled_data = scaler.fit_transform(data[cols_to_scale])
    standardized_data = standardizer.fit_transform(scaled_data)
    data[cols_to_scale] = pd.DataFrame(standardized_data, columns=cols_to_scale)

    # Remove outliers only from scaled data columns
    z_scores = np.abs(stats.zscore(data[cols_to_scale]))
    data = data[(z_scores < 3).all(axis=1)]

    return data


# Preprocess once at server startup
processed_data = preprocess_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aqi-data')
def aqi_data():
    try:
        df = processed_data.copy()
        df['date'] = pd.to_datetime(df['observationDateTime']).dt.date
        daily_avg = df.groupby('date')['airQualityIndex'].mean().reset_index()

        if daily_avg.empty:
            return jsonify({'error': 'No daily data found'}), 404

        data = {
            'labels': daily_avg['date'].astype(str).tolist(),
            'values': daily_avg['airQualityIndex'].round(2).tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'An error occurred in /aqi-data: {str(e)}'}), 500

@app.route('/get-hourly-aqi', methods=['GET'])
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

@app.route('/parameter-data/<date_str>')
def parameter_data(date_str):
    try:
        df = processed_data.copy()
        df['date'] = pd.to_datetime(df['observationDateTime']).dt.date

        selected_date = pd.to_datetime(date_str).date()
        df = df[df['date'] == selected_date]

        df['hour'] = pd.to_datetime(df['observationDateTime']).dt.hour

        parameters = [
            'pm2p5.avgOverTime',
            'pm10.avgOverTime',
            'no2.avgOverTime',
            'so2.avgOverTime',
        ]

        hourly_avg = df.groupby('hour')[parameters].mean().reset_index()
        hourly_avg['hour'] = hourly_avg['hour'].astype(str)

        return jsonify({
            'labels': hourly_avg['hour'].tolist(),
            'data': {param: hourly_avg[param].round(2).tolist() for param in parameters}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/temperature-humidity/<date_str>')
def temperature_humidity(date_str):
    try:
        df = processed_data.copy()
        df['date'] = pd.to_datetime(df['observationDateTime']).dt.date
        selected_date = pd.to_datetime(date_str).date()

        # Filter for the given date
        df = df[df['date'] == selected_date]

        # Extract hour
        df['hour'] = df['observationDateTime'].dt.hour

        # Columns to use
        parameters = [
            'airTemperature.avgOverTime',
            'relativeHumidity.avgOverTime'
        ]

        hourly_avg = df.groupby('hour')[parameters].mean().reset_index()
        hourly_avg['hour'] = hourly_avg['hour'].astype(str)

        return jsonify({
            'labels': hourly_avg['hour'].tolist(),
            'temperature': hourly_avg['airTemperature.avgOverTime'].round(2).tolist(),
            'humidity': hourly_avg['relativeHumidity.avgOverTime'].round(2).tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-available-dates')
def get_available_dates():
    try:
        df = processed_data.copy()
        df['date'] = pd.to_datetime(df['observationDateTime']).dt.date
        df['year'] = df['observationDateTime'].dt.year
        df['month'] = df['observationDateTime'].dt.month
        df['day'] = df['observationDateTime'].dt.day
        
        # Get unique years, months, days
        years = sorted(df['year'].unique().tolist())
        
        # Get months with data for each year
        months_by_year = {}
        for year in years:
            year_data = df[df['year'] == year]
            months_by_year[year] = sorted(year_data['month'].unique().tolist())
        
        # Get days with data for each year-month combination
        days_by_year_month = {}
        for year in years:
            days_by_year_month[year] = {}
            for month in months_by_year[year]:
                month_data = df[(df['year'] == year) & (df['month'] == month)]
                days_by_year_month[year][month] = sorted(month_data['day'].unique().tolist())
        
        return jsonify({
            'years': years,
            'months_by_year': months_by_year,
            'days_by_year_month': days_by_year_month
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

