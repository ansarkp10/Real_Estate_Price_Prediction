import pandas as pd
import numpy as np
import os

def generate_sample_dataset(n_samples=10000, output_path='data/raw_data.csv'):
    """Generate sample dataset matching your column structure"""
    
    np.random.seed(42)
    
    # Define options
    area_types = ['Super built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area']
    availability_options = ['Ready To Move', 'Immediate Possession', 
                           'In 2 years', 'In 3 years', 'In 1 year']
    
    locations = ['Whitefield', 'Electronic City', 'Sarjapur Road', 
                'Indiranagar', 'Koramangala', 'Jayanagar', 'Marathahalli',
                'HSR Layout', 'BTM Layout', 'Hebbal', 'Bellandur']
    
    societies = ['Prestige', 'Sobha', 'Brigade', 'Godrej', 'Purva', 
                'Salarpuria', 'Mantri', 'Total Environment', 'Assetz',
                'Sumadhura', 'Shriram', 'DLF', 'Unitech', 'Raheja']
    
    sizes = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK']
    
    # Generate data
    data = {
        'area_type': np.random.choice(area_types, n_samples),
        'availability': np.random.choice(availability_options, n_samples),
        'location': np.random.choice(locations, n_samples),
        'size': np.random.choice(sizes, n_samples),
        'society': np.random.choice(societies + [None], n_samples, p=[0.7] + [0.3/len(societies)]*len(societies)),
        'total_sqft': np.random.choice(
            [f"{int(x)}" for x in np.random.uniform(500, 3000, 100)] + 
            [f"{int(x)}-{int(x+np.random.uniform(100,500))}" for x in np.random.uniform(500, 2500, 100)],
            n_samples
        ),
        'bath': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.5, 0.3, 0.1]),
        'balcony': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.4, 0.4, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic prices (in lakhs)
    base_price_per_sqft = np.random.normal(6000, 1500, n_samples)
    
    # Multipliers
    area_type_multiplier = {
        'Super built-up Area': 1.2,
        'Built-up Area': 1.0,
        'Plot Area': 0.8,
        'Carpet Area': 0.9
    }
    
    location_multiplier = {
        'Whitefield': 1.3,
        'Electronic City': 0.9,
        'Sarjapur Road': 1.2,
        'Indiranagar': 1.5,
        'Koramangala': 1.4,
        'Jayanagar': 1.3,
        'Marathahalli': 1.1,
        'HSR Layout': 1.2,
        'BTM Layout': 1.0,
        'Hebbal': 1.1,
        'Bellandur': 1.3
    }
    
    bhk_multiplier = {
        '1 BHK': 0.7,
        '2 BHK': 1.0,
        '3 BHK': 1.4,
        '4 BHK': 1.8,
        '5 BHK': 2.2
    }
    
    # Calculate price in lakhs
    def calculate_sqft(sqft_str):
        if '-' in sqft_str:
            parts = sqft_str.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(sqft_str)
    
    sqft_values = df['total_sqft'].apply(calculate_sqft)
    
    df['price'] = (
        sqft_values * base_price_per_sqft *
        df['area_type'].map(area_type_multiplier) *
        df['location'].map(location_multiplier) *
        df['size'].map(bhk_multiplier) +
        df['bath'] * 50000 +
        df['balcony'] * 30000 +
        np.random.normal(0, 300000, n_samples)
    ) / 100000  # Convert to lakhs
    
    # Format price with 2 decimal places
    df['price'] = df['price'].round(2)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset generated with {n_samples} records at {output_path}")
    
    # Show sample
    print("\nSample of generated data:")
    print(df.head())
    
    # Show statistics
    print("\nDataset Statistics:")
    print(f"Average price: ₹{df['price'].mean():.2f} lakhs")
    print(f"Price range: ₹{df['price'].min():.2f} - ₹{df['price'].max():.2f} lakhs")
    print(f"Most common location: {df['location'].mode()[0]}")
    print(f"Most common size: {df['size'].mode()[0]}")
    
    return df

if __name__ == "__main__":
    generate_sample_dataset(10000, 'data/raw_data.csv')