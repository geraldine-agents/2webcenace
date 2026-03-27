# app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import pickle
import os

app = Flask(__name__)

def prepare_time_features(df):
    """Prepare time-based features from timestamp"""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['Timestamp'].dt.dayofweek >= 5
    df['is_business_hour'] = df['hour'].between(9, 17)
    # Add lag features (previous 3 hours)
    for i in range(1, 4):
        df[f'lag_{i}h'] = df.groupby('Area')['Value'].shift(i)
    # Add rolling mean features
    df['rolling_mean_6h'] = df.groupby('Area')['Value'].rolling(window=6).mean().reset_index(0, drop=True)
    df['rolling_mean_24h'] = df.groupby('Area')['Value'].rolling(window=24).mean().reset_index(0, drop=True)
    return df

def train_model(data, features):
    """Train XGBoost model with specified features"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Drop rows with NaN values (will occur for the first few hours due to lag features)
    data_clean = data.dropna(subset=features + ['Value'])
    
    model.fit(data_clean[features], data_clean['Value'])
    return model

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/data_info', methods=['GET'])
def get_data_info():
    """Get information about the available data range"""
    df = pd.read_csv('data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Get unique areas
    areas = ['BCS', 'CEN', 'NES', 'NOR', 'NTE', 'OCC', 'ORI', 'PEN', 'BCA']
    
    info = {
        'min_date': df['Timestamp'].min().strftime('%Y-%m-%dT%H:%M'),
        'max_date': df['Timestamp'].max().strftime('%Y-%m-%dT%H:%M'),
        'areas': areas
    }
    return jsonify(info)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    start_date = pd.to_datetime(data['start_date'])
    end_date = pd.to_datetime(data['end_date'])
    area = data['area']
    
    # Read and filter CSV
    df = pd.read_csv('data.csv')
    
    # Add a buffer for lag features
    buffer_start = start_date - timedelta(hours=24)
    df = prepare_time_features(df)
    
    mask = (df['Timestamp'] >= buffer_start) & \
           (df['Timestamp'] <= end_date) & \
           (df['Area'] == area)
    
    train_data = df[mask]
    
    if train_data.empty:
        return jsonify({'status': 'error', 'message': 'No data found for the specified period'}), 400
    
    features = [
        'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'is_business_hour',
        'lag_1h', 'lag_2h', 'lag_3h', 'rolling_mean_6h', 'rolling_mean_24h'
    ]
    
    model = train_model(train_data, features)
    
    # Save model using pickle (use /tmp since Vercel deployment filesystem is read-only)
    with open(f'/tmp/model_{area}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return jsonify({'status': 'success'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    start_date = pd.to_datetime(data['start_date'])
    end_date = pd.to_datetime(data['end_date'])
    area = data['area']
    
    # Check /tmp first (user-trained this session), then fall back to bundled pre-trained models
    model_path = f'/tmp/model_{area}.pkl'
    if not os.path.exists(model_path):
        model_path = f'models/model_{area}.pkl'
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'Model not trained yet'}), 400
    
    # Load model using pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Read and prepare prediction data
    df = pd.read_csv('data.csv')
    
    # Add a buffer for lag features
    buffer_start = start_date - timedelta(hours=24)
    df = prepare_time_features(df)
    
    mask = (df['Timestamp'] >= buffer_start) & \
           (df['Timestamp'] <= end_date) & \
           (df['Area'] == area)
    
    pred_data = df[mask]
    
    if pred_data.empty:
        return jsonify({'status': 'error', 'message': 'No data found for the specified period'}), 400
    
    features = [
        'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'is_business_hour',
        'lag_1h', 'lag_2h', 'lag_3h', 'rolling_mean_6h', 'rolling_mean_24h'
    ]
    
    # Filter out buffer period and get predictions
    pred_data = pred_data[pred_data['Timestamp'] >= start_date].copy()
    predictions = model.predict(pred_data[features])
    
    # Calculate error metrics
    mse = ((predictions - pred_data['Value']) ** 2).mean()
    mae = abs(predictions - pred_data['Value']).mean()
    
    # Calculate R-squared
    ss_res = ((pred_data['Value'] - predictions) ** 2).sum()
    ss_tot = ((pred_data['Value'] - pred_data['Value'].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    response = {
        'timestamps': pred_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'predictions': predictions.tolist(),
        'actual': pred_data['Value'].tolist(),
        'metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(mse ** 0.5),
            'r2': float(r2)
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

