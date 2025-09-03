import os
import pandas as pd
import pickle
from pathlib import Path
import joblib  # Better than pickle for sklearn objects
import warnings
warnings.filterwarnings('ignore')

def predict_upcoming_fixtures(data_folder='data', 
                            model_path='artifacts/xgboost/importance/models/xgboost_model.pkl',
                            features_path='artifacts/xgboost/importance/metrics/feature_importances.csv',
                            encoder_path='artifacts/xgboost/importance/models/label_encoder.pkl'):
    """
    Enhanced prediction pipeline with:
    - Proper label encoding handling
    - Better error checking
    - More robust path handling
    """
    # 1. Load the final processed data
    final_processed_path = Path(data_folder) / 'final_processed.csv'
    try:
        df = pd.read_csv(final_processed_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Processed data not found at {final_processed_path}")

    # Convert date and filter for NS games
    df['date'] = pd.to_datetime(df['date'])
    ns_games = df[df['status'] == 'NS'].copy()
    
    if len(ns_games) == 0:
        print("No upcoming fixtures (NS status) found in the data")
        return None
    else:
        print(f"Found {len(ns_games)} upcoming fixtures to predict")

    # 2. Load model, features, and encoder
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        #selected_features = pd.read_csv(features_path)['feature'].tolist()
        le = joblib.load(encoder_path)  # Load the label encoder
        print(f"Label encoder loaded from {encoder_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required artifact missing: {str(e)}")

    # Ensure ALL model-expected features exist
    for feature in model.feature_names_in_:
        if feature not in ns_games:
            ns_games[feature] = 0  # Or np.nan, or column mean

    # Now select ALL required features
    X_pred = ns_games[model.feature_names_in_]
    
    # 5. Make predictions with error handling
    try:
        print("Making predictions on upcoming fixtures...")
        predictions = model.predict_proba(X_pred)
        pred_classes = model.predict(X_pred)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

    # 6. Format results with proper label decoding
    results = ns_games[['country', 'league_id', 'league_name', 'season', 
                       'home_team', 'away_team', 'date']].copy()
    
    # Add probabilities
    results['home_win_prob'] = predictions[:, 2]  # Assuming home=1, draw=0, away=2
    results['draw_prob'] = predictions[:, 1]
    results['away_win_prob'] = predictions[:, 0]
    
    # Decode predicted outcomes
    results['predicted_outcome'] = le.inverse_transform(pred_classes)
    results['predicted_outcome_code'] = pred_classes  # Keep encoded version

    # 7. Save predictions with proper structure
    output_dfs = []
    for (country, league_id, league_name, season), group in results.groupby(
        ['country', 'league_id', 'league_name', 'season']):
        
        # Create directory path using Path for cross-platform compatibility
        save_path = Path(data_folder) / 'predictions' / str(country) / f"{league_id}_{league_name}" / str(season)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        output_file = save_path / 'predictions.csv'
        group.to_csv(output_file, index=False)
        output_dfs.append(group)
        print(f"Saved predictions to {output_file}")

    # Return concatenated results
    return pd.concat(output_dfs) if output_dfs else None

if __name__ == "__main__":
    try:
        predictions = predict_upcoming_fixtures()
        if predictions is not None:
            print(predictions[['home_team', 'away_team', 'home_win_prob', 
                             'draw_prob', 'away_win_prob', 'predicted_outcome']])
    except Exception as e:
        print(f"Prediction failed: {str(e)}")