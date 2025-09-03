from flask import Flask, render_template, request, jsonify, send_file
import threading
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import the lazy-initialized pipeline classes
from src import CollectPipeline, ExtractPipeline, ProcessPipeline, TrainPipeline, PredictPipeline

app = Flask(__name__)

# Configuration
class Config:
    COUNTRIES = ['England', 'Spain', 'Germany', 'Italy', 'France']
    LEAGUES = {
        'England': ['Premier League', 'Championship'],
        'Spain': ['La Liga', 'Segunda Division'],
        'Germany': ['Bundesliga', '2. Bundesliga'],
        'Italy': ['Serie A', 'Serie B'],
        'France': ['Ligue 1', 'Ligue 2']
    }
    DATA_TYPES = ['matches', 'stats', 'lineups', 'events', 'standings']
    CURRENT_SEASON = "2023/2024"
    API_KEY = os.environ.get('API_KEY', 'your-api-key-here')
    
    # Pipeline config
    PIPELINE_CONFIG = {
        'raw_dir': 'data/extracted',
        'final_output': 'data/final_processed_pipeline.csv',
        'verbose': True,
        'rolling_windows': [5],
        'min_matches': 5,
        'drop_non_roll_features': True,
        'drop_original_metrics': True,
    }

app.config.from_object(Config)

# Initialize pipeline components with lazy initialization
collector = CollectPipeline(config={'api_key': app.config['API_KEY']})
extractor = ExtractPipeline()
processor = ProcessPipeline(config=app.config['PIPELINE_CONFIG'])
trainer = TrainPipeline(config={
    'task_type': 'auto',
    'random_state': 42
})
predictor = PredictPipeline()

# Global state for pipeline status
pipeline_status = {
    'current_task': None,
    'progress': 0,
    'status': 'idle',
    'message': '',
    'last_run': None,
    'predictions_file': None,
    'model_metrics': None
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_league_id(country, league):
    """Map country and league to league ID"""
    league_mapping = {
        'England': {'Premier League': 39, 'Championship': 40},
        'Spain': {'La Liga': 140, 'Segunda Division': 141},
        'Germany': {'Bundesliga': 78, '2. Bundesliga': 79},
        'Italy': {'Serie A': 135, 'Serie B': 136},
        'France': {'Ligue 1': 61, 'Ligue 2': 62}
    }
    return league_mapping.get(country, {}).get(league)

def get_season_dates(season):
    """Get start and end dates for a season"""
    season_years = season.split('/')
    if len(season_years) == 2:
        start_date = f"{season_years[0]}-08-01"
        end_date = f"{season_years[1]}-05-31"
        return start_date, end_date
    return None, None

def run_pipeline_phase1(country, league, season, data_types):
    """Run collection phase 1 pipeline for past games"""
    try:
        pipeline_status.update({
            'current_task': 'collection_phase1',
            'progress': 0,
            'status': 'running',
            'message': 'Starting Phase 1: Collecting past games data...'
        })
        
        league_id = get_league_id(country, league)
        
        # Determine dates based on season type
        if season == app.config['CURRENT_SEASON']:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date, _ = get_season_dates(season)
        else:
            start_date, end_date = get_season_dates(season)
        
        # Collect data
        pipeline_status['message'] = 'Collecting completed games data...'
        collector.collect_data(
            league_id=league_id, 
            season=season, 
            data_types=data_types, 
            keep_progress=False,
            start_date=start_date, 
            end_date=end_date, 
            collection_phase=1
        )
        pipeline_status['progress'] = 25
        
        # Extract data
        pipeline_status['message'] = 'Extracting data from JSON files...'
        extractor.extract_all_league()
        pipeline_status['progress'] = 50
        
        # Process data
        pipeline_status['message'] = 'Processing and feature engineering...'
        processor.process_all_leagues()
        pipeline_status['progress'] = 75
        
        # Train model
        pipeline_status['message'] = 'Training machine learning model...'
        
        # Check if processed file exists
        processed_file = app.config['PIPELINE_CONFIG']['final_output']
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed file not found: {processed_file}")
        
        df = pd.read_csv(processed_file)
        
        # Train the model
        model, X_holdout, y_holdout = trainer.train_all_leagues(
            df, 
            target_col='outcome', 
            model_type='random_forest',
            feature_selection_method='importance', 
            top_n_features=30,
            use_bayesian=False, 
            bayesian_iter=50, 
            use_grid_search=False,
            use_random_search=False, 
            random_search_iter=50, 
            load_params=False,
            holdout_ratio=0.2
        )
        
        # Generate visualizations and get metrics
        metrics = trainer.generate_visualizations(X_holdout, y_holdout)
        pipeline_status['model_metrics'] = metrics
        
        pipeline_status['progress'] = 100
        
        pipeline_status.update({
            'status': 'completed',
            'message': 'Phase 1 completed: Past games collected and model trained',
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in phase 1: {str(e)}", exc_info=True)
        pipeline_status.update({
            'status': 'error',
            'message': f'Error in Phase 1: {str(e)}'
        })

def run_pipeline_phase2(country, league, season, data_types):
    """Run collection phase 2 pipeline for predictions"""
    try:
        pipeline_status.update({
            'current_task': 'collection_phase2',
            'progress': 0,
            'status': 'running',
            'message': 'Starting Phase 2: Collecting upcoming games for predictions...'
        })
        
        league_id = get_league_id(country, league)
        
        # Collect not started games
        pipeline_status['message'] = 'Collecting upcoming games data...'
        collector.collect_league_data_filter(
            league_id=league_id, 
            season=season, 
            data_types=data_types, 
            keep_progress=False,
            collection_phase=2
        )
        pipeline_status['progress'] = 33
        
        # Extract data
        pipeline_status['message'] = 'Extracting upcoming games data...'
        extractor.process_all_leagues_seasons()
        pipeline_status['progress'] = 66
        
        # Make predictions
        pipeline_status['message'] = 'Making predictions for upcoming games...'
        
        # Check if processed file exists
        processed_file = app.config['PIPELINE_CONFIG']['final_output']
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed file not found: {processed_file}")
        
        df = pd.read_csv(processed_file)
        
        # Make predictions
        predictions_df = predictor.predict(df)
        
        # Save predictions
        predictions_dir = 'predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        predictions_file = os.path.join(
            predictions_dir, 
            f'predictions_{country}_{league}_{season}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        predictions_df.to_csv(predictions_file, index=False)
        
        pipeline_status['progress'] = 100
        pipeline_status['predictions_file'] = predictions_file
        
        pipeline_status.update({
            'status': 'completed',
            'message': 'Phase 2 completed: Predictions generated for upcoming games',
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in phase 2: {str(e)}", exc_info=True)
        pipeline_status.update({
            'status': 'error',
            'message': f'Error in Phase 2: {str(e)}'
        })

def run_pipeline_phase3(country, league, season, data_types):
    """Run collection phase 3 pipeline for updates"""
    try:
        pipeline_status.update({
            'current_task': 'collection_phase3',
            'progress': 0,
            'status': 'running',
            'message': 'Starting Phase 3: Updating games and retraining model...'
        })
        
        league_id = get_league_id(country, league)
        
        # Collect updated games
        pipeline_status['message'] = 'Collecting updated games data...'
        collector.collect_league_data_filter(
            league_id=league_id, 
            season=season, 
            data_types=data_types, 
            keep_progress=False,
            collection_phase=3
        )
        pipeline_status['progress'] = 25
        
        # Extract data
        pipeline_status['message'] = 'Extracting updated data...'
        extractor.process_all_leagues_seasons()
        pipeline_status['progress'] = 50
        
        # Process data
        pipeline_status['message'] = 'Processing updated data...'
        processor.run_pipeline()
        pipeline_status['progress'] = 75
        
        # Retrain model with updated data
        pipeline_status['message'] = 'Retraining model with updated data...'
        
        # Check if processed file exists
        processed_file = app.config['PIPELINE_CONFIG']['final_output']
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed file not found: {processed_file}")
        
        df = pd.read_csv(processed_file)
        
        # Retrain the model
        model, X_holdout, y_holdout = trainer.train(
            df, 
            target_col='outcome', 
            model_type='random_forest',
            feature_selection_method='importance', 
            top_n_features=30,
            use_bayesian=False, 
            bayesian_iter=50, 
            use_grid_search=False,
            use_random_search=False, 
            random_search_iter=50, 
            load_params=False,
            holdout_ratio=0.2
        )
        
        # Generate updated visualizations and metrics
        metrics = trainer.generate_visualizations(X_holdout, y_holdout)
        pipeline_status['model_metrics'] = metrics
        
        pipeline_status['progress'] = 100
        
        pipeline_status.update({
            'status': 'completed',
            'message': 'Phase 3 completed: Data updated and model retrained',
            'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in phase 3: {str(e)}", exc_info=True)
        pipeline_status.update({
            'status': 'error',
            'message': f'Error in Phase 3: {str(e)}'
        })

if __name__ == '__main__':
   
    app.run(debug=True, host='0.0.0.0', port=5000)