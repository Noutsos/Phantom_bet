import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging
from src.collect_pipeline import CollectPipeline
from src.extract_pipeline import ExtractPipeline
from src.process_pipeline import ProcessPipeline
from src.train_pipeline import TrainPipeline
from src.predict_pipeline import PredictPipeline


# Configuration
PIPELINE_CONFIG = {
    'raw_dir': 'data/extracted',
    'merged_dir': 'data/processed',
    'final_output': 'data/final_processed.csv',
    'verbose': True,
    'data_types': {
        'fixtures': 'fixture_events.csv',
        'team_stats': 'team_statistics.csv'
    },
    'required_cols': {
        'fixtures': ['fixture_id', 'home_team_id', 'away_team_id', 'date'],
        'team_stats': ['fixture_id', 'team_id']
    },
    'rolling_windows': [5],
    'min_matches': 5,
    'merge_first': True,
    'h2h_store': 'data/processed/h2h_store.pkl',
    'standings_store': 'data/processed/standings_store.pkl',
    'drop_non_roll_features': True,
    'drop_original_metrics': True,
    'drop_non_roll_standings': True
}
# Training configuration
TRAIN_CONFIG = {
    'target_col': 'outcome',
    'model_type': 'xgboost',
    'feature_selection_method': 'rfe',
    'top_n_features': 60,
    'use_bayesian': False,
    'bayesian_iter': 50,
    'use_grid_search': False,
    'use_random_search': False,
    'random_search_iter': 50,
    'load_params': True,
    'holdout_ratio': 0.2,
    'handle_class_imbalance': True,
    'class_weights_dict': None,  # e.g., {0:1, 1:2, 2:1}
    'use_smote': True,  # ← Add this
    'smote_strategy': {1: 1500}  # ← And this
}

def run_collection(league_id, season, data_types, keep_progress, batch_size, 
               progress_file, start_date, end_date, collection_phase,
               filter_ids, filter_tiers, filter_cups, filter_categories):
    """Run collection phase 1 pipeline for past games"""
    
    API_KEY = "25c02ce9f07df0edc1e69866fbe7d156"
    collector = CollectPipeline(API_KEY)
    
    # Collect data
    collector.collect_league_data_filter(
        league_id=league_id, 
        season=season, 
        data_types=data_types, 
        keep_progress=keep_progress,
        batch_size=batch_size,
        progress_file=progress_file,
        start_date=start_date,
        end_date=end_date,
        collection_phase=collection_phase,
        filter_ids=filter_ids,
        filter_tiers=filter_tiers,
        filter_cups=filter_cups,
        filter_categories=filter_categories
    )
        
def run_extraction():

    # Initialize extractor with config - this only sets up the object
    extractor = ExtractPipeline()
    extractor.process_all_leagues_seasons()

def run_processing():
    # Create processing-specific config using your PIPELINE_CONFIG
    PROCESS_CONFIG = {
        'raw_dir': 'data/extracted',
        'merged_dir': 'data/processed',
        'final_output': PIPELINE_CONFIG['final_output'],  # Use your final_output
        'verbose': PIPELINE_CONFIG['verbose'],
        'data_types': {
            'fixtures': 'fixture_events.csv',
            'team_stats': 'team_statistics.csv'
        },
        'required_cols': {
            'fixtures': ['fixture_id', 'home_team_id', 'away_team_id', 'date'],
            'team_stats': ['fixture_id', 'team_id']
        },
        'rolling_windows': PIPELINE_CONFIG['rolling_windows'],
        'min_matches': PIPELINE_CONFIG['min_matches'],
        'merge_first': True,
        'h2h_store': 'data/processed/h2h_store.pkl',
        'standings_store': 'data/processed/standings_store.pkl',
        'drop_non_roll_features': PIPELINE_CONFIG.get('drop_non_roll_features', True),
        'drop_original_metrics': PIPELINE_CONFIG.get('drop_original_metrics', True),
        'drop_non_roll_standings': PIPELINE_CONFIG.get('drop_non_roll_standings', True)
    }
    
    # Initialize processor with config - this only sets up the object
    processor = ProcessPipeline(config=PROCESS_CONFIG)
    
    # This is what actually runs the pipeline
    processor.run_pipeline(force_processing=True)

def run_training():
    # Load processed data
    df = pd.read_csv(PIPELINE_CONFIG['final_output'])
    
    # Create training-specific config using your TRAIN_CONFIG
    TRAIN_CONFIG = {
        'target_col': 'outcome',
        'model_type': 'xgboost',
        'feature_selection_method': 'rfe',
        'top_n_features': 70,
        'use_bayesian': False,
        'bayesian_iter': 50,
        'use_grid_search': False,
        'use_random_search': False,
        'random_search_iter': 50,
        'load_params': True,
        'holdout_ratio': 0.2,
        'task_type': 'auto',  # Add any other parameters you need
        'random_state': 42,
        'handle_class_imbalance': True,
        'class_weights_dict': None, # away_win:1, draw:2.5, home_win:1
        'use_smote': True,  # ← Add this
        'smote_strategy': 'auto'  # ← And this
    }
    
    # Initialize trainer with config - this only sets up the object
    trainer = TrainPipeline(
        task_type=TRAIN_CONFIG.get('task_type', 'auto'),
        random_state=TRAIN_CONFIG.get('random_state', 42),
        log_level=logging.INFO
    )
    
    # This is what actually runs the training
    trainer.train(
        df=df,
        target_col=TRAIN_CONFIG['target_col'],
        model_type=TRAIN_CONFIG['model_type'],
        feature_selection_method=TRAIN_CONFIG['feature_selection_method'],
        top_n_features=TRAIN_CONFIG['top_n_features'],
        use_bayesian=TRAIN_CONFIG['use_bayesian'],
        bayesian_iter=TRAIN_CONFIG['bayesian_iter'],
        use_grid_search=TRAIN_CONFIG['use_grid_search'],
        use_random_search=TRAIN_CONFIG['use_random_search'],
        random_search_iter=TRAIN_CONFIG['random_search_iter'],
        load_params=TRAIN_CONFIG['load_params'],
        holdout_ratio=TRAIN_CONFIG['holdout_ratio'],
        handle_class_imbalance=TRAIN_CONFIG['handle_class_imbalance'],
        class_weights_dict=TRAIN_CONFIG['class_weights_dict'],
        use_smote=TRAIN_CONFIG['use_smote'],
        smote_strategy=TRAIN_CONFIG['smote_strategy']
    )

def run_prediction():

    # Initialize predictor with config - this only sets up the object
    predictor = PredictPipeline()
    model_path = f"artifacts/{TRAIN_CONFIG['model_type']}/{TRAIN_CONFIG['feature_selection_method']}/models/{TRAIN_CONFIG['model_type']}_model.pkl"
    
    predictor.load_trained_model(model_path=model_path)
    predictor.predict_upcoming_fixtures()

def main():
    parser = argparse.ArgumentParser(description='Soccer Data Pipeline')
    
    # Pipeline phase selection
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
                       help='Pipeline phase (1: Collection, 2: Extraction, 3: Processing, 4: Training, 5: Prediction, 6: Full pipeline)')
    
    # Collection phase arguments
    parser.add_argument('--league-id', type=int, default=None, help='League ID to process')
    parser.add_argument('--season', type=int, default=None, help='Season year')
    
    # Define default data types based on collection phase
    parser.add_argument('--collection-phase', type=int, default=1,
                       help='Collection phase (1: historical, 2: current, etc.)')
    
    # Data types will be set conditionally after parsing
    parser.add_argument('--data-types', type=str, nargs='+', default=None,
                       help='Data types to collect')
    
    parser.add_argument('--keep-progress', action='store_true', default=False,
                       help='Keep progress file for resuming collection')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for API requests')
    parser.add_argument('--progress-file', type=str, default="data_collection_progress.json",
                       help='File to store progress data')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--filter-ids', type=int, nargs='+', default=None,
                       help='Filter by specific fixture IDs')
    parser.add_argument('--filter-tiers', type=str, nargs='+', default=None,
                       help='Filter by competition tiers')
    parser.add_argument('--filter-cups', type=str, nargs='+', default=None,
                       help='Filter by cup competitions')
    parser.add_argument('--filter-categories', type=str, nargs='+', default=None,
                       help='Filter by competition categories')
    
    # Training phase arguments (optional overrides)
    parser.add_argument('--target-col', type=str, default=None, help='Target column for training')
    parser.add_argument('--model-type', type=str, default=None, help='Model type for training')
    parser.add_argument('--top-n-features', type=int, default=None, help='Number of top features to select')
    parser.add_argument('--holdout-ratio', type=float, default=None, help='Holdout ratio for validation')
    
    args = parser.parse_args()
    
    # Set default data types based on collection phase if not provided by user
    if args.data_types is None:
        if args.collection_phase == 1:
            # Default data types for historical collection (phase 1)
            args.data_types = ['fixture_events', 'team_statistics']
        elif args.collection_phase == 2:
            # Default data types for current collection (phase 2)
            args.data_types = ['fixture_events', 'oddds']
        else:
            # Fallback default
            args.data_types = ['fixture_events', 'team_statistics', 'team_standings', 'player_statistics', 'injuries', 'lineups']
    
    # Convert date strings to datetime objects if provided
    start_date = args.start_date  # Keep as string '2025-08-29'
    end_date = args.end_date      # Keep as string '2025-09-01'

    # Override training config with command line arguments if provided
    if args.target_col:
        TRAIN_CONFIG['target_col'] = args.target_col
    if args.model_type:
        TRAIN_CONFIG['model_type'] = args.model_type
    if args.top_n_features:
        TRAIN_CONFIG['top_n_features'] = args.top_n_features
    if args.holdout_ratio:
        TRAIN_CONFIG['holdout_ratio'] = args.holdout_ratio
    
    print(f"\nStarting pipeline phase {args.phase}:")
    if args.phase == 1:
        print(f"  League ID: {args.league_id}")
        print(f"  Season: {args.season}")
        print(f"  Collection Phase: {args.collection_phase}")
        print(f"  Data types: {args.data_types}")
    print("-" * 50)
    
    try:
        if args.phase == 1:
            
            run_collection(
                league_id=args.league_id,
                season=args.season,
                data_types=args.data_types,
                keep_progress=args.keep_progress,
                batch_size=args.batch_size,
                progress_file=args.progress_file,
                start_date=start_date,
                end_date=end_date,
                collection_phase=args.collection_phase,
                filter_ids=args.filter_ids,
                filter_tiers=args.filter_tiers,
                filter_cups=args.filter_cups,
                filter_categories=args.filter_categories
            )
        elif args.phase == 2:
            run_extraction()
        elif args.phase == 3:
            run_processing()
        elif args.phase == 4:
            run_training()
        elif args.phase == 5:
            run_prediction()
        elif args.phase == 6:
            # Full pipeline - run all phases
            if not args.league_id or not args.season:
                raise ValueError("League ID and Season are required for full pipeline")
            print("Running full pipeline...")
            run_collection(
                league_id=args.league_id,
                season=args.season,
                data_types=args.data_types,
                keep_progress=args.keep_progress,
                batch_size=args.batch_size,
                progress_file=args.progress_file,
                start_date=start_date,
                end_date=end_date,
                collection_phase=args.collection_phase,
                filter_ids=args.filter_ids,
                filter_tiers=args.filter_tiers,
                filter_cups=args.filter_cups,
                filter_categories=args.filter_categories
            )
            run_extraction()
            run_processing()
            run_training()
            run_prediction()
        
        print(f"\nPipeline phase {args.phase} completed successfully!")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()