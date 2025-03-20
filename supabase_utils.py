import pandas as pd

# Load the CSV file
df = pd.read_csv('historical_weather_all.csv')

# Ensure the 'name' column is consistently 'choondacherry'
df['name'] = 'choondacherry'

# Save the corrected data to a new CSV file
df.to_csv('corrected_historical_weather_all.csv', index=False)

print("The 'name' column has been standardized to 'choondacherry' and saved to 'corrected_historical_weather_all.csv'.")