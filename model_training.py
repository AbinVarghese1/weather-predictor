import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from supabase import create_client

class ModelTrainer:
    def __init__(self, supabase_url, supabase_key, models_dir='models'):
        """
        Initialize ModelTrainer with Supabase integration

        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase API key
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        self.supabase = create_client(supabase_url, supabase_key)
        os.makedirs(self.models_dir, exist_ok=True)

    def preprocess_data(self, data):
        """Preprocess data for model training"""
        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
        data['hour'] = data['datetime'].dt.hour
        data['day'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month
        data['day_of_week'] = data['datetime'].dt.weekday
        data['day_of_year'] = data['datetime'].dt.dayofyear

        label_encoder = LabelEncoder()
        data['conditions_encoded'] = label_encoder.fit_transform(data['conditions'])

        features = ['hour', 'day', 'month', 'day_of_week', 'day_of_year', 'temp', 'humidity',
                    'sealevelpressure', 'uvindex', 'windgust', 'winddir']
        targets = {
            'temp': data['temp'],
            'weather': data[['windgust', 'humidity', 'uvindex', 'sealevelpressure']].values,
            'conditions': data['conditions_encoded']
        }
        return data[features], targets, label_encoder

    def train_models(self, data_filepath):
        """Train models for weather prediction"""
        data = pd.read_csv(data_filepath)
        X, targets, label_encoder = self.preprocess_data(data)

        X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, targets['temp'], test_size=0.2, random_state=42)
        _, _, y_weather_train, y_weather_test = train_test_split(X, targets['weather'], test_size=0.2, random_state=42)
        _, _, y_conditions_train, y_conditions_test = train_test_split(X, targets['conditions'], test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        temp_model.fit(X_train_scaled, y_temp_train)

        weather_model = RandomForestRegressor(n_estimators=100, random_state=42)
        weather_model.fit(X_train_scaled, y_weather_train)

        conditions_model = RandomForestClassifier(n_estimators=100, random_state=42)
        conditions_model.fit(X_train_scaled, y_conditions_train)

        self.save_models(temp_model, weather_model, conditions_model, scaler, label_encoder)
        metrics = {
            'temp_model_score': temp_model.score(X_test_scaled, y_temp_test),
            'weather_model_score': weather_model.score(X_test_scaled, y_weather_test),
            'conditions_model_score': conditions_model.score(X_test_scaled, y_conditions_test)
        }
        self.upload_metrics_to_supabase(metrics)

        print("Models trained and saved successfully!")
        return metrics

    def save_models(self, temp_model, weather_model, conditions_model, scaler, label_encoder):
        """Save trained models and delete old ones from Supabase"""
        self.supabase.storage.from_('models').remove(['temp_model.joblib', 'weather_model.joblib',
                                                      'conditions_model.joblib', 'scaler.joblib', 'label_encoder.joblib'])

        joblib.dump(temp_model, os.path.join(self.models_dir, 'temp_model.joblib'))
        joblib.dump(weather_model, os.path.join(self.models_dir, 'weather_model.joblib'))
        joblib.dump(conditions_model, os.path.join(self.models_dir, 'conditions_model.joblib'))
        joblib.dump(scaler, os.path.join(self.models_dir, 'scaler.joblib'))
        joblib.dump(label_encoder, os.path.join(self.models_dir, 'label_encoder.joblib'))

        self.upload_model_to_supabase('temp_model.joblib')
        self.upload_model_to_supabase('weather_model.joblib')
        self.upload_model_to_supabase('conditions_model.joblib')
        self.upload_model_to_supabase('scaler.joblib')
        self.upload_model_to_supabase('label_encoder.joblib')

    def upload_model_to_supabase(self, model_name):
        """Upload a model file to Supabase"""
        with open(os.path.join(self.models_dir, model_name), 'rb') as file:
            self.supabase.storage.from_('models').upload(model_name, file)

    def upload_metrics_to_supabase(self, metrics):
        """Upload training metrics to Supabase"""
        self.supabase.table('training_metrics').insert(metrics).execute()

if __name__ == "__main__":
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

    trainer = ModelTrainer(SUPABASE_URL, SUPABASE_KEY)
    metrics = trainer.train_models('path/to/latest_weather_data.csv')
    print("Training metrics:", metrics)