import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import os
from supabase import create_client
import time
import pytz

class WeatherPredictor:
    def __init__(self, supabase_url, supabase_key, api_key, location, models_dir='models'):
        """
        Initialize WeatherPredictor with real-time data integration
        
        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase API key
            api_key (str): Visual Crossing API key
            location (str): Location for weather prediction
            models_dir (str): Directory containing trained models
        """
        self.models_dir = models_dir
        self.supabase = create_client(supabase_url, supabase_key)
        self.api_key = api_key
        self.location = location
        self.load_models()
        
        # Set timezone to IST
        self.timezone = pytz.timezone('Asia/Kolkata')

    def load_models(self):
        """Load pre-trained models from local directory"""
        try:
            self.temp_model = joblib.load(os.path.join(self.models_dir, 'temp_model.joblib'))
            self.weather_model = joblib.load(os.path.join(self.models_dir, 'weather_model.joblib'))
            self.conditions_model = joblib.load(os.path.join(self.models_dir, 'conditions_model.joblib'))
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.joblib'))
            self.label_encoder = joblib.load(os.path.join(self.models_dir, 'label_encoder.joblib'))
        except FileNotFoundError as e:
            print(f"Model loading error: {e}")
            raise

    def fetch_current_sensor_data(self):
        """Fetch latest sensor data from Supabase real-time database"""
        try:
            response = self.supabase.table('sensor_data')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if response.data:
                data = response.data[0]
                return {
                    'temperature': float(data['temperature']),
                    'humidity': float(data['humidity']),
                    'pressure': float(data['pressure']),
                    'uv_index': float(data['uv_index'])
                }
            raise Exception("No sensor data available")
        except Exception as e:
            print(f"Error fetching sensor data: {e}")
            raise

    def fetch_api_data(self):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{self.location}"
        params = {
            'unitGroup': 'metric',
            'key': self.api_key,
            'contentType': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('currentConditions', {})
            
            return {
                'windspeed': float(current.get('windspeed', 0) or 0),
                'winddir': float(current.get('winddir', 0) or 0),
                'cloudcover': float(current.get('cloudcover', 0) or 0),
                'visibility': float(current.get('visibility', 10) or 10),
                'rain_level': float(current.get('precip', 0) or 0),
                'sunrise': current.get('sunrise', '06:00:00'),
                'sunset': current.get('sunset', '18:00:00')
            }
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            raise
        except ValueError as e:
            print(f"Error converting API data: {e}")
            raise
        except Exception as e:
            print(f"Error fetching API data: {e}")
            raise

    def store_prediction_inputs(self, sensor_data, api_data):
        """Store the input data used for predictions"""
        try:
            current_time = datetime.now(self.timezone)
            
            input_data = {
                'timestamp': current_time.isoformat(),
                'temperature': sensor_data['temperature'],
                'humidity': sensor_data['humidity'],
                'pressure': sensor_data['pressure'],
                'uv_index': sensor_data['uv_index'],
                'windspeed': api_data['windspeed'],
                'winddir': api_data['winddir'],
                'cloudcover': api_data['cloudcover'],
                'visibility': api_data['visibility'],
                'rain_level': api_data['rain_level'],
                'sunrise': api_data['sunrise'],
                'sunset': api_data['sunset']
            }
            
            self.supabase.table('prediction_inputs').insert(input_data).execute()
            print(f"Successfully stored prediction inputs at {current_time}")
            return True
        except Exception as e:
            print(f"Error storing prediction inputs: {e}")
            return False

    def calculate_wbt(self, temp, humidity):
        """Calculate Wet Bulb Temperature"""
        try:
            es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
            wbt = temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                  np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + \
                  0.00391838 * (humidity) ** (3 / 2) * np.arctan(0.023101 * humidity) - 4.686035
            return round(float(wbt), 1)
        except Exception as e:
            print(f"Error calculating WBT: {e}")
            return None

    def predict_hourly(self, sensor_data, api_data):
        """Make hourly predictions for the next 8 days using sliding window approach"""
        predictions = []
        current_time = datetime.now(self.timezone)
        daily_temps = {}
        
        try:
            # Initialize with current data
            current_temp = sensor_data['temperature']
            current_humidity = sensor_data['humidity']
            current_pressure = sensor_data['pressure']
            current_uv_index = sensor_data['uv_index']
            current_windspeed = api_data['windspeed']
            current_cloudcover = api_data['cloudcover']
            current_winddir = api_data['winddir']
            current_rain_level = api_data['rain_level']
            
            # Predict for next 8 days * 24 hours
            for i in range(8 * 24):
                future_time = current_time + timedelta(hours=i)
                date_key = future_time.date().isoformat()
                
                features = {
                    'hour': future_time.hour,
                    'day': future_time.day,
                    'month': future_time.month,
                    'day_of_week': future_time.weekday(),
                    'day_of_year': future_time.timetuple().tm_yday,
                    'temp': current_temp,
                    'humidity': current_humidity,
                    'sealevelpressure': current_pressure,
                    'uvindex': current_uv_index,
                    'windspeed': current_windspeed,
                    'cloudcover': current_cloudcover,
                    'winddir': current_winddir
                }

                feature_order = [
                    'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                    'temp', 'humidity', 'sealevelpressure', 'uvindex',
                    'windspeed', 'cloudcover', 'winddir'
                ]
                
                features_df = pd.DataFrame([{key: features[key] for key in feature_order}])
                features_scaled = self.scaler.transform(features_df)

                # Get temperature predictions
                temp_predictions = self.temp_model.predict(features_scaled)[0]
                temp = temp_predictions[0]  # Regular temperature
                temp_max = temp_predictions[1]  # Maximum temperature
                temp_min = temp_predictions[2]  # Minimum temperature

                # Get weather predictions
                weather = self.weather_model.predict(features_scaled)[0]
                wind = weather[0]
                humidity = weather[1]
                uv_index = weather[2]
                pressure = weather[3]
                rain_chance = weather[4]
                
                # Get condition prediction
                condition_encoded = self.conditions_model.predict(features_scaled)[0]
                condition = self.label_encoder.inverse_transform([condition_encoded])[0]

                # Calculate wet bulb temperature
                wbt = self.calculate_wbt(temp, humidity)
                
                # Track daily min/max temperatures
                if date_key not in daily_temps:
                    daily_temps[date_key] = {'min_temp': temp_min, 'max_temp': temp_max}
                else:
                    daily_temps[date_key]['min_temp'] = min(daily_temps[date_key]['min_temp'], temp_min)
                    daily_temps[date_key]['max_temp'] = max(daily_temps[date_key]['max_temp'], temp_max)
                
                # Create prediction object
                prediction = {
                    'datetime': future_time.isoformat(),
                    'temp': round(float(temp), 1),
                    'temp_max': round(float(temp_max), 1),
                    'temp_min': round(float(temp_min), 1),
                    'condition': condition,
                    'wind': round(float(wind), 1),
                    'humidity': round(float(humidity), 1),
                    'uvIndex': round(float(uv_index), 1),
                    'pressure': round(float(pressure), 1),
                    'rainChance': round(float(rain_chance * 100), 1),
                    'wbt': wbt,
                    'rain_level': round(float(current_rain_level), 1),
                    'wind_direction': round(float(current_winddir), 1),
                    'sunrise': api_data['sunrise'],
                    'sunset': api_data['sunset'],
                    'min_temp': round(float(daily_temps[date_key]['min_temp']), 1),
                    'max_temp': round(float(daily_temps[date_key]['max_temp']), 1),
                    'prediction_made_at': current_time.isoformat()
                }
                
                predictions.append(prediction)
                
                # Update the current values for the next iteration (sliding window)
                current_temp = temp
                current_humidity = humidity
                current_pressure = pressure
                current_uv_index = uv_index
                current_windspeed = wind
                
                # Conditionally update cloudcover and rain_level based on predictions
                if rain_chance > 0.5:  # If rain is likely
                    current_cloudcover = min(100, current_cloudcover + 10)  # Increase cloud cover
                    current_rain_level = max(0, min(25, current_rain_level + (rain_chance * 2)))  # Adjust rain level
                else:
                    current_cloudcover = max(0, current_cloudcover - 5)  # Decrease cloud cover
                    current_rain_level = max(0, current_rain_level - 0.5)  # Decrease rain level
                
                # Randomly adjust wind direction slightly for more realistic predictions
                current_winddir = (current_winddir + np.random.uniform(-10, 10)) % 360

            return predictions
        except Exception as e:
            print(f"Error in predict_hourly: {e}")
            raise

    def update_predictions_in_supabase(self, predictions):
        """Update predictions in Supabase"""
        try:
            current_time = datetime.now(self.timezone)
            
            # Delete future predictions only
            self.supabase.table('weather_predictions')\
                .delete()\
                .gte('datetime', current_time.isoformat())\
                .execute()
            
            # Insert new predictions in batches
            batch_size = 100
            for i in range(0, len(predictions), batch_size):
                batch = predictions[i:i + batch_size]
                self.supabase.table('weather_predictions').insert(batch).execute()
            
            print(f"Successfully updated predictions at {current_time}")
            return True
        except Exception as e:
            print(f"Error updating predictions: {e}")
            return False

    def run_prediction_cycle(self):
        """Run a complete prediction cycle"""
        try:
            print(f"\nStarting prediction cycle at {datetime.now(self.timezone)}")
            
            print("Fetching sensor data...")
            sensor_data = self.fetch_current_sensor_data()
            
            print("Fetching API data...")
            api_data = self.fetch_api_data()
            
            print("Storing prediction inputs...")
            self.store_prediction_inputs(sensor_data, api_data)
            
            print("Making predictions with sliding window approach...")
            predictions = self.predict_hourly(sensor_data, api_data)
            
            print("Updating predictions in database...")
            success = self.update_predictions_in_supabase(predictions)
            
            if success:
                print("Prediction cycle completed successfully")
            else:
                print("Failed to update predictions in database")
            
            return success
        except Exception as e:
            print(f"Error in prediction cycle: {e}")
            return False

def main():
    # Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    API_KEY = os.getenv('API_KEY')
    LOCATION = os.getenv('LOCATION')
    
    try:
        predictor = WeatherPredictor(SUPABASE_URL, SUPABASE_KEY, API_KEY, LOCATION)
        
        while True:
            success = predictor.run_prediction_cycle()
            
            if not success:
                print("Prediction cycle failed, waiting before retry...")
            
            print(f"\nWaiting for 4 hours before next prediction cycle...")
            time.sleep(4 * 60 * 60)
            
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"\nCritical error in main loop: {e}")
        raise

if __name__ == "__main__":
    main()
