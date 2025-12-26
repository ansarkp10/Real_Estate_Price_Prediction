import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from django.conf import settings

class RealEstatePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_type = 'random_forest'
        self.model_performance = {}
        self.feature_importance = None
        
    def train_model(self, X_train, X_test, y_train, y_test, model_type='random_forest'):
        """Train the model"""
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif model_type == 'svr':
            self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate Mean Absolute Percentage Error
        y_test_nonzero = y_test[y_test != 0]
        y_pred_nonzero = y_pred[y_test != 0]
        
        if len(y_test_nonzero) > 0:
            mape = np.mean(np.abs((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)) * 100
        else:
            mape = 0
        
        # Calculate accuracy within percentage ranges
        within_10_percent = np.sum(np.abs((y_test - y_pred) / y_test) <= 0.10) / len(y_test) * 100
        within_20_percent = np.sum(np.abs((y_test - y_pred) / y_test) <= 0.20) / len(y_test) * 100
        
        self.model_performance = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'within_10_percent': within_10_percent,
            'within_20_percent': within_20_percent,
            'avg_prediction': np.mean(y_pred),
            'avg_actual': np.mean(y_test)
        }
        
        return self.model_performance
    
    def predict(self, input_features):
        """Make prediction - returns price in lakhs"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        prediction = self.model.predict(input_features)
        return float(prediction[0])  # Price in lakhs
    
    def get_feature_importance_df(self, feature_names):
        """Get feature importance as DataFrame"""
        if self.feature_importance is not None and feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def save_model(self, model_path, preprocessor_path):
        """Save model and preprocessor"""
        if self.model:
            joblib.dump(self.model, model_path)
        if self.preprocessor:
            self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Save model info
        model_info = {
            'model_type': self.model_type,
            'performance': self.model_performance,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        info_path = os.path.join(os.path.dirname(model_path), 'model_info.json')
        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
    
    def load_model(self, model_path, preprocessor_path):
        """Load model and preprocessor"""
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            self.model = joblib.load(model_path)
            
            # Load preprocessor
            from .preprocessing import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
            
            # Load model info
            info_path = os.path.join(os.path.dirname(model_path), 'model_info.json')
            if os.path.exists(info_path):
                import json
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                    self.model_type = model_info.get('model_type', 'random_forest')
                    self.model_performance = model_info.get('performance', {})
                    feature_imp = model_info.get('feature_importance')
                    if feature_imp:
                        self.feature_importance = np.array(feature_imp)
            
            return True
        return False