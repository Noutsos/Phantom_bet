# predictor.py
import logging
import os
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from src.utils import get_feature_names_from_pipeline

class PredictPipeline:
    """Standalone class for making predictions with trained pipelines"""
    
    def __init__(self, pipeline=None, log_dir="logs/predict"):
        """
        Initialize Predictor with a trained MLPipeline
        
        Parameters:
        -----------
        pipeline : MLPipeline, optional
            A trained MLPipeline instance with model, paths, etc.
        log_dir : str, optional
            Directory where log files will be saved
        """
        self.pipeline = pipeline
        self.model = pipeline.model if pipeline else None
        self.task_type = pipeline.task_type if pipeline else None
        self.paths = getattr(pipeline, 'paths', {}) if pipeline else {}
        self.le = getattr(pipeline, 'le', None) if pipeline else None
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        self.predictions = None
        self.missing_features = []
    


    def _setup_logging(self):
        """Set up logging with both console and file handlers"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        
        # File handler - create a new log file for each session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"predictor_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def load_trained_model(self, model_path):
        """
        Load a trained model from disk with memory mapping for large files
        """
        try:
            self.logger.info(f"Loading trained model from {model_path}")
            
            # Check file size
            file_size = os.path.getsize(model_path)
            self.logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
            
            # Use memory mapping for large files
            if file_size > 100 * 1024 * 1024:  # If larger than 100MB
                self.logger.info("Using memory-mapped loading for large model file")
                self.model = joblib.load(model_path, mmap_mode='r')
            else:
                self.model = joblib.load(model_path)
            
            # Try to determine task type from the model
            if hasattr(self.model, 'predict_proba'):
                self.task_type = 'classification'
                self.logger.info("Detected classification model (has predict_proba method)")
            else:
                self.task_type = 'regression'
                self.logger.info("Detected regression model")
            
            # Try to load label encoder if it exists in the same directory
            model_dir = os.path.dirname(model_path)
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                self.le = joblib.load(encoder_path)
                self.logger.info(f"Label encoder loaded from {encoder_path}")
            
            self.logger.info("Trained model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {str(e)}", exc_info=True)
            return False
    
    def predict_upcoming_fixtures(self, data_folder='data', raw_data_path=None, default_value=0):
        """
        Enhanced prediction pipeline for upcoming fixtures with robust feature handling
        """
        if self.model is None:
            self.logger.error("Model not trained yet. Load a trained model first.")
            raise ValueError("Model not trained yet. Load a trained model first.")
        
        self.logger.info("Starting prediction of upcoming fixtures...")
        
        # 1. Load the data
        if raw_data_path:
            final_processed_path = raw_data_path
        else:
            final_processed_path = os.path.join(data_folder, 'final_processed.csv')
        
        try:
            df = pd.read_csv(final_processed_path)
            self.logger.info(f"Loaded data from {final_processed_path}")
        except FileNotFoundError:
            error_msg = f"Data not found at {final_processed_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Convert date and filter for NS games
        df['date'] = pd.to_datetime(df['date'])
        ns_games = df[df['status'] == 'NS'].copy()
        
        if len(ns_games) == 0:
            self.logger.warning("No upcoming fixtures (NS status) found in the data")
            return None
        
        self.logger.info(f"Found {len(ns_games)} upcoming fixtures to predict")

        # 2. Get the features the model expects
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
            self.logger.info(f"Model expects {len(feature_names)} features (from feature_names_in_)")
        else:
            # Fallback approaches
            try:
                feature_names = get_feature_names_from_pipeline(self.model)
                self.logger.info(f"Model expects {len(feature_names)} features (from pipeline extraction)")
            except Exception as e:
                error_msg = f"Could not determine feature names from model: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Convert feature_names to list if it's a numpy array
        if hasattr(feature_names, 'tolist'):
            feature_names = feature_names.tolist()
        
        # Debug: log first few feature names
        if feature_names and len(feature_names) > 0:
            self.logger.debug(f"First 10 feature names: {feature_names[:10]}")
        
        # Identify missing features
        missing_features = [f for f in feature_names if f not in ns_games.columns]
        existing_features = [f for f in feature_names if f in ns_games.columns]
        
        self.logger.info(f"Existing features: {len(existing_features)}, Missing features: {len(missing_features)}")
        
        if missing_features:
            self.logger.warning(f"Missing {len(missing_features)} features, adding with default value {default_value}")
            if len(missing_features) <= 20:
                self.logger.warning(f"Missing features: {missing_features}")
            else:
                self.logger.warning(f"First 20 missing features: {missing_features[:20]}")
            
            # Add missing features with default value
            for feature in missing_features:
                ns_games[feature] = default_value
        
        # Select only the features the model expects
        X_pred = ns_games[feature_names]
        
        # 3. Make predictions with error handling
        try:
            self.logger.info("Making predictions on upcoming fixtures...")
            
            if self.task_type == 'classification':
                predictions = self.model.predict_proba(X_pred)
                pred_classes = self.model.predict(X_pred)
            else:
                predictions = self.model.predict(X_pred)
                pred_classes = predictions
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

        # 4. Format results
        results = ns_games[['league_name', 'date', 'home_team', 'away_team', 'odds_home_win', 'odds_draw', 'odds_away_win']].copy()
        
        if self.task_type == 'classification':
            # Add probabilities for classification
            if predictions.shape[1] == 3:  # Assuming 3 classes: away, draw, home
                results['away_win_prob'] = predictions[:, 0]
                results['draw_prob'] = predictions[:, 1]
                results['home_win_prob'] = predictions[:, 2]
                results['confidence'] = np.max(predictions, axis=1)
            else:
                for i in range(predictions.shape[1]):
                    results[f'class_{i}_prob'] = predictions[:, i]
            
            # Decode predicted outcomes if encoder is available
            if self.le is not None:
                try:
                    results['predicted_outcome'] = self.le.inverse_transform(pred_classes)
                except Exception as e:
                    self.logger.warning(f"Could not decode predictions with label encoder: {str(e)}")
                    results['predicted_outcome'] = pred_classes
            else:
                results['predicted_outcome'] = pred_classes
                
            #results['predicted_outcome_code'] = pred_classes
            
        else:
            results['predicted_value'] = predictions
            results['prediction_confidence'] = 1.0

        # 5. Add metadata about missing features
        #results['missing_features_count'] = len(missing_features)
        if missing_features:
            missing_features_str = f"{len(missing_features)} features missing" if len(missing_features) > 10 else ", ".join(missing_features)
            #results['missing_features'] = missing_features_str

        # 6. Save predictions
        save_path = os.path.join(data_folder, 'predictions')
        os.makedirs(save_path, exist_ok=True)
        
        output_file = os.path.join(save_path, 'predictions.csv')
        results.to_csv(output_file, index=False)
        self.logger.info(f"Saved predictions to {output_file}")

        # Store predictions and missing features
        self.predictions = results
        self.missing_features = missing_features
        
        # Log prediction summary
        if self.task_type == 'classification' and self.predictions is not None:
            outcome_counts = self.predictions['predicted_outcome'].value_counts()
            self.logger.info("Prediction summary:")
            for outcome, count in outcome_counts.items():
                self.logger.info(f"  {outcome}: {count} predictions")
        
        self.logger.info("Prediction completed successfully!")
        return self.predictions
    
    def analyze_feature_availability(self, data_folder='data'):
        """
        Analyze which features are available vs missing in the prediction data
        """
        if self.model is None:
            self.logger.error("Model not available for analysis")
            return None
        
        # Load the data
        final_processed_path = os.path.join(data_folder, 'final_processed.csv')
        try:
            df = pd.read_csv(final_processed_path)
            self.logger.info(f"Successfully loaded data from {final_processed_path}")
        except FileNotFoundError:
            self.logger.error(f"Data not found at {final_processed_path}")
            return None
        
        # Get model expected features
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            try:
                feature_names = get_feature_names_from_pipeline(self.model)
            except:
                self.logger.error("Could not determine feature names from model")
                return None
        
        # Convert feature_names to list if it's a numpy array
        if hasattr(feature_names, 'tolist'):
            feature_names = feature_names.tolist()
        
        # Analyze availability
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        analysis = {
            'total_features_expected': len(feature_names),
            'available_features_count': len(available_features),
            'missing_features_count': len(missing_features),
            'coverage_percentage': (len(available_features) / len(feature_names)) * 100 if len(feature_names) > 0 else 0,
            'available_features': available_features,
            'missing_features': missing_features
        }
        
        self.logger.info(f"Feature availability analysis:")
        self.logger.info(f"  Expected: {analysis['total_features_expected']} features")
        self.logger.info(f"  Available: {analysis['available_features_count']} features")
        self.logger.info(f"  Missing: {analysis['missing_features_count']} features")
        self.logger.info(f"  Coverage: {analysis['coverage_percentage']:.1f}%")
        
        return analysis

    def get_prediction_summary(self):
        """
        Get a summary of the predictions
        """
        if self.predictions is None:
            self.logger.warning("No predictions available. Call predict_upcoming_fixtures() first.")
            return None
        
        if self.task_type == 'classification':
            summary_cols = ['home_team', 'away_team', 'home_win_prob', 
                          'draw_prob', 'away_win_prob', 'predicted_outcome', 'confidence']
        else:
            summary_cols = ['home_team', 'away_team', 'predicted_value', 'prediction_confidence']
        
        return self.predictions[summary_cols].copy()
    
    def save_predictions_to_excel(self, output_path=None):
        """
        Save predictions to Excel format
        """
        if self.predictions is None:
            self.logger.warning("No predictions available. Call predict_upcoming_fixtures() first.")
            return False
        
        if output_path is None:
            output_path = 'predictions.xlsx'
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                self.predictions.to_excel(writer, sheet_name='All Predictions', index=False)
                
                if 'league_name' in self.predictions.columns:
                    for league in self.predictions['league_name'].unique():
                        league_df = self.predictions[self.predictions['league_name'] == league]
                        sheet_name = str(league)[:31]
                        league_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Predictions saved to Excel: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save Excel file: {str(e)}")
            return False

