import pandas as pd
import os

# Define the file paths for 2000–2022 (adjusted to match your range)
data_filepaths = [f"C:\\Users\\User\\OROSTAT APP\\Backend\\data\\dataset {year}.csv" for year in range(2023, 2025)]

# Function to convert Fahrenheit to Celsius


# Function to detect and parse datetime format
def parse_datetime(date_str, year):
    formats = ['%d-%m-%y', '%d-%m-%Y']
    for fmt in formats:
        try:
            dt = pd.to_datetime(date_str, format=fmt, dayfirst=True)
            if fmt == '%d-%m-%y' and dt.year < 2000:
                dt = dt.replace(year=year)
            return dt
        except ValueError:
            continue
    return pd.NaT

# Process each file
for filepath in data_filepaths:
    if os.path.exists(filepath):
        print(f"Processing {filepath}...")
        
        # Load the CSV
        df = pd.read_csv(filepath)

        # Extract year from filename
        year = int(filepath.split('dataset ')[1].split('.csv')[0])

        # Parse datetime column
        df['datetime'] = df['datetime'].apply(lambda x: parse_datetime(x, year))
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d 00:00:00').fillna('1900-01-01 00:00:00')

        # Log invalid dates
        invalid_dates = df[df['datetime'] == '1900-01-01 00:00:00']
        if not invalid_dates.empty:
            print(f"Warning: {len(invalid_dates)} rows in {filepath} have invalid dates (set to 1900-01-01):")
            print(invalid_dates[['name', 'datetime']].head())

        # Convert temperature columns (NOT NULL
        # Fix sunrise and sunset (NOT NULL)
        for col in ['sunrise', 'sunset']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].str.replace('T', ' '), format='%Y-%m-%d %H:%M:%S', errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('1900-01-01 00:00:00')

        # Handle numeric data (NOT NULL and nullable)
        numeric_columns_not_null = ['humidity', 'precip', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover']
        numeric_columns_nullable = ['windgust', 'uvindex']
        for col in numeric_columns_not_null:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # NOT NULL
        for col in numeric_columns_nullable:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Nullable, NaN allowed

        # Ensure all required columns exist (NOT NULL text columns)
        required_text_columns = ['name', 'conditions', 'icon']
        for col in required_text_columns:
            if col not in df.columns:
                df[col] = 'Unknown'  # Default for NOT NULL text
            else:
                df[col] = df[col].fillna('Unknown')

        # Handle nullable text column
        if 'preciptype' not in df.columns:
            df['preciptype'] = pd.NA
        else:
            df['preciptype'] = df['preciptype'].where(pd.notna(df['preciptype']), pd.NA)

        # Define expected columns (18 data columns, excluding id)
        expected_columns = ['name', 'datetime', 'tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'preciptype', 
                           'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'uvindex', 
                           'sunrise', 'sunset', 'conditions', 'icon']
        df = df[expected_columns]  # Reorder and select only these columns

        # Save with explicit formatting
        output_filepath = filepath.replace('.csv', '_converted.csv')
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig', sep=',')
        print(f"Saved converted data to {output_filepath}")
    else:
        print(f"File not found: {filepath}")

print("Processing complete!")