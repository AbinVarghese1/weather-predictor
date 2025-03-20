import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_wbt(temp, humidity):
    """Calculate wet bulb temperature using empirical formula."""
    try:
        es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
        e = (humidity / 100) * es
        wbt = temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
              np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + \
              0.00391838 * (humidity) ** (3/2) * np.arctan(0.023101 * humidity) - 4.686035
        return round(wbt, 1)
    except Exception:
        return np.nan

def calculate_dew_point(temp, humidity):
    """Calculate dew point temperature."""
    try:
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity/100.0)
        dew_point = (b * alpha) / (a - alpha)
        return round(dew_point, 1)
    except Exception:
        return np.nan

def clean_conditions(conditions):
    """Clean and standardize weather conditions."""
    if pd.isna(conditions) or conditions == 'nan':
        return 'Unknown'
    return str(conditions).strip()

def handle_seasonal_patterns(df):
    """Create seasonal features from datetime."""
    # Add cyclical encoding for month and day of year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def load_and_preprocess_data(filepaths):
    """Load and preprocess weather data from multiple CSV files."""
    # Load and combine data
    dataframes = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not dataframes:
        raise ValueError("No valid data files could be loaded")
    
    data = pd.concat(dataframes, ignore_index=True)
    
    # Convert and standardize datetime
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])
    
    # Extract datetime features
    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.day
    data['month'] = data['datetime'].dt.month
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['day_of_year'] = data['datetime'].dt.dayofyear
    
    # Add seasonal patterns
    data = handle_seasonal_patterns(data)
    
    # Calculate additional weather features
    data['wbt'] = data.apply(lambda row: calculate_wbt(row['temp'], row['humidity']), axis=1)
    data['dew_point'] = data.apply(lambda row: calculate_dew_point(row['temp'], row['humidity']), axis=1)
    
    # Temperature differences
    data['temp_range'] = data['tempmax'] - data['tempmin']
    
    # Clean and encode weather conditions
    data['conditions'] = data['conditions'].apply(clean_conditions)
    label_encoder = LabelEncoder()
    data['conditions_encoded'] = label_encoder.fit_transform(data['conditions'])
    
    # Define feature and target columns
    features = [
        'hour', 'day', 'month', 'day_of_week', 'day_of_year', 
        'temp', 'humidity', 'sealevelpressure', 'uvindex', 
        'windspeed', 'cloudcover', 'winddir',
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
        'wbt', 'dew_point', 'temp_range'
    ]
    
    # Ensure all features exist in the dataframe
    for feature in features:
        if feature not in data.columns:
            if feature == 'cloudcover' and 'cloudcover' not in data.columns:
                # If cloudcover is missing, estimate it from conditions
                data['cloudcover'] = data['conditions'].apply(
                    lambda x: 100 if 'Overcast' in x else 
                              50 if 'Partially cloudy' in x else 
                              80 if 'Rain' in x else 10
                )
            else:
                # Create empty columns for missing features
                data[feature] = np.nan
    
    # Filter to only include valid features
    available_features = [f for f in features if f in data.columns]
    
    targets = {
        'temp': ['temp', 'tempmax', 'tempmin'],
        'weather': ['windspeed', 'humidity', 'uvindex', 'sealevelpressure', 'precip', 'winddir'],
        'conditions': 'conditions_encoded'
    }
    
    # Handle missing values
    for feature in available_features:
        if feature in data.columns:
            data[feature] = data[feature].ffill().bfill()
    
    for target_list in [targets['temp'], targets['weather']]:
        for target in target_list:
            if target in data.columns:
                data[target] = data[target].ffill().bfill()
    
    # Create target dataframes
    temp_df = data[targets['temp']]
    weather_df = data[targets['weather']]
    conditions = data[targets['conditions']]
    
    # Final check for any remaining NaN values
    missing_columns = data[available_features].columns[
        data[available_features].isna().any()
    ].tolist()
    
    if missing_columns:
        print(f"Warning: NaN values found in columns: {missing_columns}")
        # Fill any remaining NaNs with column means
        for col in missing_columns:
            data[col] = data[col].fillna(data[col].mean())
    
    return data[available_features], temp_df, weather_df, conditions, label_encoder

def evaluate_regression_model(y_true, y_pred, model_name):
    """Evaluate regression model performance."""
    # Convert to numpy arrays if they're pandas Series/DataFrames
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    # Ensure proper dimensions
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Regression Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def evaluate_classification_model(y_true, y_pred, label_encoder, model_name):
    """Evaluate classification model performance."""
    # Calculate metrics with zero_division parameter
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name} Classification Metrics:")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Get class names from label encoder
    class_names = label_encoder.classes_.tolist()
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=class_names,
                              zero_division=0))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'class_names': class_names
    }

def train_and_evaluate_models(data_filepaths, output_dir='models'):
    """Train and evaluate all weather prediction models."""
    print("Loading and preprocessing data...")
    X, y_temp, y_weather, y_conditions, label_encoder = load_and_preprocess_data(data_filepaths)
    
    # Split data with stratification for conditions
    print("Splitting data into train and test sets...")
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    _, _, y_weather_train, y_weather_test = train_test_split(X, y_weather, test_size=0.2, random_state=42)
    _, _, y_conditions_train, y_conditions_test = train_test_split(X, y_conditions, test_size=0.2, random_state=42, stratify=y_conditions)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate temperature model
    print("\nTraining temperature model...")
    temp_model = RandomForestRegressor(
        n_estimators=150, 
        random_state=42, 
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True
    )
    temp_model.fit(X_train_scaled, y_temp_train)
    y_temp_pred = temp_model.predict(X_test_scaled)
    
    # Handle case where y_temp_test is a DataFrame
    if isinstance(y_temp_test, pd.DataFrame):
        temp_metrics = {}
        for i, col in enumerate(['Temperature', 'Maximum Temperature', 'Minimum Temperature']):
            temp_metrics[col] = evaluate_regression_model(
                y_temp_test.iloc[:, i],
                y_temp_pred[:, i] if y_temp_pred.ndim > 1 else y_temp_pred,
                col
            )
    else:
        # Handle case where y_temp_test is a Series or ndarray
        temp_metrics = {'Temperature': evaluate_regression_model(y_temp_test, y_temp_pred, 'Temperature')}
    
    # Train and evaluate weather model
    print("\nTraining weather model...")
    weather_model = RandomForestRegressor(
        n_estimators=150, 
        random_state=42, 
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True
    )
    weather_model.fit(X_train_scaled, y_weather_train)
    y_weather_pred = weather_model.predict(X_test_scaled)
    
    # Handle case where y_weather_test is a DataFrame
    if isinstance(y_weather_test, pd.DataFrame):
        weather_metrics = {}
        col_names = ['Wind Speed', 'Humidity', 'UV Index', 'Sea Level Pressure', 'Precipitation', 'Wind Direction']
        for i, col in enumerate(col_names[:len(y_weather_test.columns)]):
            weather_metrics[col] = evaluate_regression_model(
                y_weather_test.iloc[:, i],
                y_weather_pred[:, i] if y_weather_pred.ndim > 1 else y_weather_pred,
                col
            )
    else:
        # Handle case where y_weather_test is a Series or ndarray
        weather_metrics = {'Weather': evaluate_regression_model(y_weather_test, y_weather_pred, 'Weather')}
    
    # Train and evaluate conditions model
    print("\nTraining conditions model...")
    # Use class weighting for imbalanced classes
    class_weights = {i: 1.0 for i in range(len(np.unique(y_conditions_train)))}
    # Adjust weights for minority classes
    unique_classes, class_counts = np.unique(y_conditions_train, return_counts=True)
    max_count = max(class_counts)
    for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
        if count < max_count / 10:  # If class is less than 10% of the majority class
            class_weights[cls] = max_count / count
    
    conditions_model = RandomForestClassifier(
        n_estimators=150, 
        random_state=42, 
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True,
        class_weight=class_weights
    )
    conditions_model.fit(X_train_scaled, y_conditions_train)
    y_conditions_pred = conditions_model.predict(X_test_scaled)
    
    conditions_metrics = evaluate_classification_model(
        y_conditions_test,
        y_conditions_pred,
        label_encoder,
        "Weather Conditions"
    )
    
    # Save models and preprocessing objects
    print(f"\nSaving models and metrics to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    model_files = {
        'temp_model.joblib': temp_model,
        'weather_model.joblib': weather_model,
        'conditions_model.joblib': conditions_model,
        'scaler.joblib': scaler,
        'label_encoder.joblib': label_encoder
    }
    
    for filename, model in model_files.items():
        filepath = os.path.join(output_dir, filename)
        try:
            joblib.dump(model, filepath)
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    # Save metrics
    metrics = {
        'temperature': temp_metrics,
        'weather': weather_metrics,
        'conditions': conditions_metrics
    }
    
    try:
        joblib.dump(metrics, os.path.join(output_dir, 'evaluation_metrics.joblib'))
        print("Saved evaluation metrics")
    except Exception as e:
        print(f"Error saving metrics: {e}")
    
    return metrics

if __name__ == "__main__":
    # Generate file paths for datasets
    data_filepaths = [f"C:\\Users\\User\\OROSTAT APP\\Backend\\data\\dataset {year}_converted.csv" for year in range(2000, 2025)]
    
    # Train models and get evaluation metrics
    try:
        metrics = train_and_evaluate_models(data_filepaths)
        print("\nModel training and evaluation completed successfully!")
    except Exception as e:
        print(f"\nError during model training and evaluation: {e}")
        import traceback
        traceback.print_exc()