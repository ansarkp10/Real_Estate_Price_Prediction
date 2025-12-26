import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from django.conf import settings

class PropertyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        self.is_trained = False
        
    def auto_train(self):
        """Automatically train model on startup"""
        print("🔧 Auto-training ML model...")
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            X, y, features = self.preprocess_data(df)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X, y)
            
            self.feature_names = features
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            print(f"✅ Model trained on {len(df)} samples")
            print(f"📊 Features: {len(features)}")
            print(f"🎯 R² Score: {self.model.score(X, y):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def load_data(self):
        """Load dataset"""
        dataset_path = os.path.join(settings.BASE_DIR, 'data', 'your_dataset.csv')
        
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"📁 Loaded dataset: {len(df)} records")
        else:
            print("📝 Creating sample dataset...")
            df = self.create_sample_data()
            df.to_csv(dataset_path, index=False)
            print(f"✅ Created sample dataset: {len(df)} records")
        
        return df
    
    def create_sample_data(self):
        """Create realistic sample data"""
        np.random.seed(42)
        n_samples = 5000
        
        # Realistic data
        data = {
            'area_type': np.random.choice(['Super built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area'], 
                                          n_samples, p=[0.5, 0.3, 0.1, 0.1]),
            'availability': np.random.choice(['Ready To Move', 'Immediate Possession', 'In 1 year', 'In 2 years', 'In 3 years'],
                                            n_samples, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
            'location': np.random.choice(['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata'],
                                        n_samples),
            'size': np.random.choice(['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK'],
                                    n_samples, p=[0.1, 0.4, 0.3, 0.15, 0.05]),
            'society': np.random.choice(['Prestige', 'Sobha', 'Brigade', 'Godrej', 'Purva', None],
                                       n_samples, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
            'total_sqft': np.random.randint(500, 3000, n_samples),
            'bath': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.5, 0.3, 0.1]),
            'balcony': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.4, 0.4, 0.1]),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate realistic prices in lakhs
        base_price = df['total_sqft'] * 5  # ₹5000 per sqft
        
        # Adjustments
        size_multiplier = {'1 BHK': 0.8, '2 BHK': 1.0, '3 BHK': 1.5, '4 BHK': 2.0, '5 BHK': 2.5}
        location_multiplier = {
            'Bangalore': 1.2, 'Mumbai': 1.5, 'Delhi': 1.3, 
            'Chennai': 1.0, 'Hyderabad': 1.1, 'Pune': 1.0, 'Kolkata': 0.9
        }
        area_multiplier = {
            'Super built-up Area': 1.2,
            'Built-up Area': 1.0,
            'Plot Area': 0.8,
            'Carpet Area': 0.9
        }
        
        df['price'] = (base_price * 
                       df['size'].map(size_multiplier) * 
                       df['location'].map(location_multiplier) * 
                       df['area_type'].map(area_multiplier))
        
        # Add society premium
        society_premium = {'Prestige': 1.1, 'Sobha': 1.08, 'Brigade': 1.05, 'Godrej': 1.07, 'Purva': 1.06, None: 1.0}
        df['price'] = df['price'] * df['society'].map(society_premium)
        
        # Add amenities
        df['price'] += df['bath'] * 0.5  # Each bath adds ₹0.5L
        df['price'] += df['balcony'] * 0.3  # Each balcony adds ₹0.3L
        
        # Add random variation
        df['price'] += np.random.normal(0, 2, n_samples)
        
        # Ensure positive prices
        df['price'] = df['price'].clip(lower=10)
        df['price'] = df['price'].round(2)
        
        return df
    
    def preprocess_data(self, df):
        """Prepare data for training"""
        df_clean = df.copy()
        
        # Extract BHK
        df_clean['bhk'] = df_clean['size'].str.extract('(\d+)').astype(float)
        df_clean['society'].fillna('Unknown', inplace=True)
        
        # Encode categorical
        categorical = ['area_type', 'availability', 'location', 'society']
        for col in categorical:
            self.encoders[col] = LabelEncoder()
            df_clean[col] = self.encoders[col].fit_transform(df_clean[col])
        
        # Features
        features = ['area_type', 'availability', 'location', 'society', 'bhk', 'total_sqft', 'bath', 'balcony']
        features = [f for f in features if f in df_clean.columns]
        
        X = df_clean[features]
        y = df_clean['price']
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, features
    
    def predict(self, features_dict):
        """Predict price"""
        if not self.is_trained:
            return None
        
        try:
            # Prepare input
            input_df = pd.DataFrame([features_dict])
            
            # Extract BHK
            input_df['bhk'] = input_df['size'].str.extract('(\d+)').astype(float)
            input_df['society'].fillna('Unknown', inplace=True)
            
            # Encode
            for col, encoder in self.encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except:
                        input_df[col] = 0
            
            # Ensure all features
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Scale and predict
            X = input_df[self.feature_names]
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            
            return round(float(prediction), 2)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def save_model(self):
        """Save model"""
        model_path = os.path.join(settings.BASE_DIR, 'estate_app', 'ml_model', 'trained_model.joblib')
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(save_data, model_path)
    
    def load_model(self):
        """Load trained model"""
        model_path = os.path.join(settings.BASE_DIR, 'estate_app', 'ml_model', 'trained_model.joblib')
        
        if os.path.exists(model_path):
            try:
                save_data = joblib.load(model_path)
                self.model = save_data['model']
                self.scaler = save_data['scaler']
                self.encoders = save_data['encoders']
                self.feature_names = save_data['feature_names']
                self.is_trained = save_data['is_trained']
                print("✅ Loaded trained model")
                return True
            except:
                pass
        
        return False

# Global instance - auto train on import
predictor = PropertyPredictor()
if not predictor.load_model():
    predictor.auto_train()