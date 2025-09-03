import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from src.collect_pipeline import CollectPipeline
from src.extract_pipeline import ExtractPipeline
from src.process_pipeline import ProcessPipeline
from src.train_pipeline import TrainPipeline
from src.predict_pipeline import PredictPipeline
from src.utils import LEAGUES

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

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
    'class_weights_dict': None,
    'use_smote': True,
    'smote_strategy': {1: 1500}
}



# Helper functions to filter leagues
def get_leagues_by_categories(categories):
    """Get leagues by category names"""
    matching_leagues = []
    for region, countries in LEAGUES.items():
        if region == 'Europe':  # Special case for international competitions
            for league_id, league_info in countries.items():
                if league_info['category'] in categories:
                    matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': 'International'})
        else:
            for country, leagues in countries.items():
                for league_id, league_info in leagues.items():
                    if league_info['category'] in categories:
                        matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': country})
    return matching_leagues

def get_leagues_by_countries(countries):
    """Get leagues by country names"""
    matching_leagues = []
    for region, country_data in LEAGUES.items():
        if region == 'Europe':  # Special case for international competitions
            if 'International' in countries:
                for league_id, league_info in country_data.items():
                    matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': 'International'})
        else:
            for country, leagues in country_data.items():
                if country in countries:
                    for league_id, league_info in leagues.items():
                        matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': country})
    return matching_leagues

def get_leagues_by_regions(regions):
    """Get leagues by region names"""
    matching_leagues = []
    for region, country_data in LEAGUES.items():
        if region in regions:
            if region == 'Europe':  # Special case for international competitions
                for league_id, league_info in country_data.items():
                    matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': 'International'})
            else:
                for country, leagues in country_data.items():
                    for league_id, league_info in leagues.items():
                        matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': country})
    return matching_leagues

def get_leagues_by_names(league_names):
    """Get leagues by league names"""
    matching_leagues = []
    for region, countries in LEAGUES.items():
        if region == 'Europe':  # Special case for international competitions
            for league_id, league_info in countries.items():
                if league_info['name'] in league_names:
                    matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': 'International'})
        else:
            for country, leagues in countries.items():
                for league_id, league_info in leagues.items():
                    if league_info['name'] in league_names:
                        matching_leagues.append({'id': int(league_id), 'name': league_info['name'], 'country': country})
    return matching_leagues

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collection', methods=['GET', 'POST'])
def collection():
    if request.method == 'POST':
        # Get form data
        league_id = request.form.get('league_id')
        season = request.form.get('season')
        
        # Check if league_id is required
        filter_ids = request.form.get('filter_ids')
        filter_categories = request.form.get('filter_categories')
        filter_countries = request.form.get('filter_countries')
        filter_regions = request.form.get('filter_regions')
        filter_leagues = request.form.get('filter_leagues')
        
        # If using filter IDs, categories, countries, regions, or league names, league_id is not required
        if not league_id and not filter_ids and not filter_categories and not filter_countries and not filter_regions and not filter_leagues:
            return render_template('collection.html', error="League ID is required unless using filters")
        
        # Get league IDs based on filters
        league_ids = []
        
        # If specific league ID is provided
        if league_id:
            league_ids.append(int(league_id))
        
        # If filter by IDs is provided
        if filter_ids:
            try:
                ids = [int(id_str.strip()) for id_str in filter_ids.split(',')]
                league_ids.extend(ids)
            except ValueError:
                return render_template('collection.html', error="Invalid format for Filter IDs. Please use comma-separated numbers.")
        
        # If filter by categories is provided
        if filter_categories:
            categories = [cat.strip() for cat in filter_categories.split(',')]
            for league in get_leagues_by_categories(categories):
                league_ids.append(league['id'])
        
        # If filter by countries is provided
        if filter_countries:
            countries = [country.strip() for country in filter_countries.split(',')]
            for league in get_leagues_by_countries(countries):
                league_ids.append(league['id'])
        
        # If filter by regions is provided
        if filter_regions:
            regions = [region.strip() for region in filter_regions.split(',')]
            for league in get_leagues_by_regions(regions):
                league_ids.append(league['id'])
        
        # If filter by league names is provided
        if filter_leagues:
            league_names = [name.strip() for name in filter_leagues.split(',')]
            for league in get_leagues_by_names(league_names):
                league_ids.append(league['id'])
        
        # Remove duplicates
        league_ids = list(set(league_ids))
        
        if not league_ids:
            return render_template('collection.html', error="No leagues found matching the specified filters")
        
        session['collection_config'] = {
            'league_ids': league_ids,
            'season': int(season) if season else None,
            'data_types': request.form.getlist('data_types'),
            'keep_progress': 'keep_progress' in request.form,
            'batch_size': int(request.form.get('batch_size', 50)),
            'progress_file': request.form.get('progress_file', 'data_collection_progress.json'),
            'start_date': request.form.get('start_date'),
            'end_date': request.form.get('end_date'),
            'collection_phase': int(request.form.get('collection_phase', 1)),
            'filter_ids': filter_ids,
            'filter_tiers': request.form.get('filter_tiers'),
            'filter_cups': request.form.get('filter_cups'),
            'filter_categories': filter_categories,
            'filter_countries': filter_countries,
            'filter_regions': filter_regions,
            'filter_leagues': filter_leagues
        }
        
        # Run collection for all league IDs
        try:
            results = []
            for league_id in league_ids:
                result = run_collection(
                    league_id=league_id,
                    season=session['collection_config']['season'],
                    data_types=session['collection_config']['data_types'],
                    keep_progress=session['collection_config']['keep_progress'],
                    batch_size=session['collection_config']['batch_size'],
                    progress_file=session['collection_config']['progress_file'],
                    start_date=session['collection_config']['start_date'],
                    end_date=session['collection_config']['end_date'],
                    collection_phase=session['collection_config']['collection_phase']
                )
                results.append(result)
            
            # Store results in session for display
            session['collection_results'] = results
            return redirect(url_for('results', phase='collection', success=True, league_count=len(league_ids)))
        except Exception as e:
            return redirect(url_for('results', phase='collection', success=False, error=str(e)))
    
    return render_template('collection.html')
    
@app.route('/extraction', methods=['GET', 'POST'])
def extraction():
    if request.method == 'POST':
        try:
            run_extraction()
            return redirect(url_for('results', phase='extraction', success=True))
        except Exception as e:
            return redirect(url_for('results', phase='extraction', success=False, error=str(e)))
    
    return render_template('extraction.html')

@app.route('/processing', methods=['GET', 'POST'])
def processing():
    if request.method == 'POST':
        try:
            # Get processing parameters from form
            processing_config = {
                'rolling_windows': [int(x.strip()) for x in request.form.get('rolling_windows', '5').split(',')],
                'min_matches': int(request.form.get('min_matches', 5)),
                'drop_non_roll_features': 'drop_non_roll_features' in request.form,
                'drop_original_metrics': 'drop_original_metrics' in request.form,
                'drop_non_roll_standings': 'drop_non_roll_standings' in request.form,
                'force_processing': 'force_processing' in request.form
            }
            
            session['processing_config'] = processing_config
            
            run_processing()
            return redirect(url_for('results', phase='processing', success=True))
        except Exception as e:
            return redirect(url_for('results', phase='processing', success=False, error=str(e)))
    
    return render_template('processing.html')

@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        # Get form data and update training config
        training_config = {
            'target_col': request.form.get('target_col', 'outcome'),
            'model_type': request.form.get('model_type', 'xgboost'),
            'feature_selection_method': request.form.get('feature_selection_method', 'rfe'),
            'top_n_features': int(request.form.get('top_n_features', 60)),
            'use_bayesian': 'use_bayesian' in request.form,
            'bayesian_iter': int(request.form.get('bayesian_iter', 50)),
            'use_grid_search': 'use_grid_search' in request.form,
            'use_random_search': 'use_random_search' in request.form,
            'random_search_iter': int(request.form.get('random_search_iter', 50)),
            'load_params': 'load_params' in request.form,
            'holdout_ratio': float(request.form.get('holdout_ratio', 0.2)),
            'handle_class_imbalance': 'handle_class_imbalance' in request.form,
            'class_weights_dict': request.form.get('class_weights_dict'),
            'use_smote': 'use_smote' in request.form,
            'smote_strategy': request.form.get('smote_strategy')
        }
        
        session['training_config'] = training_config
        
        try:
            run_training()
            return redirect(url_for('results', phase='training', success=True))
        except Exception as e:
            return redirect(url_for('results', phase='training', success=False, error=str(e)))
    
    return render_template('training.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            run_prediction()
            return redirect(url_for('results', phase='prediction', success=True))
        except Exception as e:
            return redirect(url_for('results', phase='prediction', success=False, error=str(e)))
    
    return render_template('prediction.html')

@app.route('/results')
def results():
    phase = request.args.get('phase', '')
    success = request.args.get('success', 'false').lower() == 'true'
    error = request.args.get('error', '')
    league_count = request.args.get('league_count', 0)
    
    # Get collection results if available
    collection_results = session.get('collection_results', [])
    
    return render_template('results.html', phase=phase, success=success, error=error, 
                          league_count=league_count, collection_results=collection_results)

# Collection function that works with your pipeline
def run_collection(league_id, season, data_types, keep_progress, batch_size, 
               progress_file, start_date, end_date, collection_phase, **kwargs):
    """Run collection phase pipeline for a league"""
    
    API_KEY = "25c02ce9f07df0edc1e69866fbe7d156"
    collector = CollectPipeline(API_KEY)
    
    # Get league info for country and league name
    country_name, league_info = collector.get_league_info(league_id)
    league_name = league_info['name']
    
    # Process the single league
    result = collector._process_single_league(
        league_id=league_id,
        season=season,
        data_types=data_types,
        keep_progress=keep_progress,
        batch_size=batch_size,
        progress_file=progress_file,
        start_date=start_date,
        end_date=end_date,
        collection_phase=collection_phase,
        country_name=country_name,
        league_name=league_name
    )
    
    return result

def run_extraction():
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
    
    # Get training config from session
    training_config = session.get('training_config', TRAIN_CONFIG)
    
    trainer = TrainPipeline(
        task_type=training_config.get('task_type', 'auto'),
        random_state=training_config.get('random_state', 42),
        log_level=logging.INFO
    )
    
    trainer.train(
        df=df,
        target_col=training_config['target_col'],
        model_type=training_config['model_type'],
        feature_selection_method=training_config['feature_selection_method'],
        top_n_features=training_config['top_n_features'],
        use_bayesian=training_config['use_bayesian'],
        bayesian_iter=training_config['bayesian_iter'],
        use_grid_search=training_config['use_grid_search'],
        use_random_search=training_config['use_random_search'],
        random_search_iter=training_config['random_search_iter'],
        load_params=training_config['load_params'],
        holdout_ratio=training_config['holdout_ratio'],
        handle_class_imbalance=training_config['handle_class_imbalance'],
        class_weights_dict=training_config['class_weights_dict'],
        use_smote=training_config['use_smote'],
        smote_strategy=training_config['smote_strategy']
    )

def run_prediction():
    predictor = PredictPipeline()
    training_config = session.get('training_config', TRAIN_CONFIG)
    model_path = f"artifacts/{training_config['model_type']}/{training_config['feature_selection_method']}/models/{training_config['model_type']}_model.pkl"
    
    predictor.load_trained_model(model_path=model_path)
    predictor.predict_upcoming_fixtures()

if __name__ == '__main__':
    app.run(debug=True)