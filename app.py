import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from src.collect_pipeline import CollectPipeline
from src.extract_pipeline import ExtractPipeline
from src.process_pipeline import ProcessPipeline
from src.train_pipeline import TrainPipeline
from src.predict_pipeline import PredictPipeline
from src.utils import LEAGUES 
from jinja2 import Environment

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
        'team_stats': 'team_statistics.csv',
        'odds': 'odds.csv'
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collection', methods=['GET', 'POST'])
def collection():
    collector = CollectPipeline("dummy_key")
    
    if request.method == 'POST':
        # Get form data
        season = request.form.get('season')
        
        # Get unified selection data
        selected_regions = request.form.getlist('selected_regions')
        selected_countries = request.form.getlist('selected_countries')
        selected_leagues = request.form.getlist('selected_leagues')  # This was missing!
        
        # Debug output
        print(f"DEBUG - Selected regions: {selected_regions}")
        print(f"DEBUG - Selected countries: {selected_countries}")
        print(f"DEBUG - Selected leagues: {selected_leagues}")
        
        # Convert to appropriate types
        filter_ids = [int(league_id) for league_id in selected_leagues if league_id and league_id.strip()] 
        filter_ids = filter_ids if filter_ids else None
        
        # If no selection made
        if not selected_regions and not selected_countries and not selected_leagues:
            leagues_by_region = collector.get_all_leagues()
            return render_template('collection.html', 
                                 error="Please select at least one region, country, or league",
                                 leagues_by_region=leagues_by_region)
        
        session['collection_config'] = {
            'league_id': None,
            'season': int(season) if season else None,
            'data_types': request.form.getlist('data_types'),
            'keep_progress': 'keep_progress' in request.form,
            'batch_size': int(request.form.get('batch_size', 50)),
            'progress_file': request.form.get('progress_file', 'data_collection_progress.json'),
            'start_date': request.form.get('start_date'),
            'end_date': request.form.get('end_date'),
            'collection_phase': int(request.form.get('collection_phase', 1)),
            'selected_regions': selected_regions,
            'selected_countries': selected_countries,
            'selected_leagues': selected_leagues,
            'filter_ids': filter_ids
        }
        
        print(f"DEBUG - Final session config: {session['collection_config']}")
        
        # Run collection
        try:
            run_collection(**session['collection_config'])
            return redirect(url_for('results', phase='collection', success=True))
        except Exception as e:
            return redirect(url_for('results', phase='collection', success=False, error=str(e)))
    
    # GET request - show available options
    leagues_by_region = collector.get_all_leagues()
    
    return render_template('collection.html', 
                         leagues_by_region=leagues_by_region)

@app.template_filter('sum_lengths')
def sum_lengths_filter(dict_values):
    """Custom filter to sum lengths of lists in dictionary values"""
    try:
        return sum(len(items) for items in dict_values)
    except:
        return 0

# Register the filter
app.jinja_env.filters['sum_lengths'] = sum_lengths_filter

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
            
            # Pass the force_processing parameter to run_processing
            run_processing(force_processing=processing_config['force_processing'])
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
            # Run prediction
            run_prediction()
            return redirect(url_for('view_predictions'))
        except Exception as e:
            return redirect(url_for('results', phase='prediction', success=False, error=str(e)))
    
    return render_template('prediction.html')



@app.route('/view_predictions', methods=['GET', 'POST'])
def view_predictions():
    # Get filter parameters from request
    league_filter = request.args.get('league', '')
    date_filter = request.args.get('date', '')
    outcome_filter = request.args.get('outcome', '')
    min_odds_filter = request.args.get('min_odds', '')
    max_odds_filter = request.args.get('max_odds', '')
    
    # Check if we have existing predictions to display
    predictions_file = 'data/predictions/predictions.csv'
    if os.path.exists(predictions_file):
        try:
            # Get file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(predictions_file))
            
            predictions_df = pd.read_csv(predictions_file)
            
            # Apply filters
            if league_filter:
                predictions_df = predictions_df[predictions_df['league_name'].str.contains(league_filter, case=False, na=False)]
            
            if date_filter:
                # Convert both dates to datetime for comparison
                predictions_df['date_only'] = pd.to_datetime(predictions_df['date']).dt.date
                filter_date = pd.to_datetime(date_filter).date()
                predictions_df = predictions_df[predictions_df['date_only'] == filter_date]
            
            if outcome_filter:
                predictions_df = predictions_df[predictions_df['predicted_outcome'].str.contains(outcome_filter, case=False, na=False)]
            
            # Apply odds filters if provided
            odds_columns = ['odds_home_win', 'odds_draw', 'odds_away_win']
            for col in odds_columns:
                if col in predictions_df.columns:
                    if min_odds_filter:
                        try:
                            min_odds = float(min_odds_filter)
                            predictions_df = predictions_df[predictions_df[col] >= min_odds]
                        except ValueError:
                            pass
                    if max_odds_filter:
                        try:
                            max_odds = float(max_odds_filter)
                            predictions_df = predictions_df[predictions_df[col] <= max_odds]
                        except ValueError:
                            pass
            
            # Format the date column for better readability
            if 'date' in predictions_df.columns:
                predictions_df['formatted_date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Format probability columns to percentages
            prob_columns = ['away_win_prob', 'draw_prob', 'home_win_prob', 'confidence']
            for col in prob_columns:
                if col in predictions_df.columns:
                    predictions_df[col] = predictions_df[col].apply(lambda x: f'{float(x):.1%}')
            
            # Format odds columns to 2 decimal places
            odds_columns = ['odds_home_win', 'odds_draw', 'odds_away_win']
            for col in odds_columns:
                if col in predictions_df.columns:
                    predictions_df[col] = predictions_df[col].apply(lambda x: f'{float(x):.2f}' if pd.notna(x) else 'N/A')
            
            # Add CSS classes for different outcomes
            predictions_df['outcome_class'] = predictions_df['predicted_outcome'].map({
                'home_win': 'outcome-home-win',
                'draw': 'outcome-draw',
                'away_win': 'outcome-away-win'
            })
            
            # Determine which columns to display (only include existing columns)
            display_columns = ['formatted_date', 'league_name', 'home_team', 'away_team', 
                             'away_win_prob', 'draw_prob', 'home_win_prob', 
                             'confidence', 'predicted_outcome']
            
            # Add odds columns if they exist
            for col in ['odds_home_win', 'odds_draw', 'odds_away_win']:
                if col in predictions_df.columns:
                    display_columns.append(col)
            
            # Convert to HTML with custom formatting
            predictions_html = predictions_df.to_html(
                classes='table table-striped prediction-table', 
                index=False,
                escape=False,
                columns=display_columns
            )
            
            match_count = len(predictions_df)
            
            # Get unique values for filter dropdowns
            unique_leagues = []
            unique_dates = []
            unique_outcomes = []
            
            if os.path.exists(predictions_file):
                all_predictions = pd.read_csv(predictions_file)
                if 'league_name' in all_predictions.columns:
                    unique_leagues = sorted(all_predictions['league_name'].unique().tolist())
                if 'date' in all_predictions.columns:
                    # Extract unique dates
                    all_predictions['date_only'] = pd.to_datetime(all_predictions['date']).dt.date
                    unique_dates = sorted(all_predictions['date_only'].unique().tolist())
                    unique_dates = [date.strftime('%Y-%m-%d') for date in unique_dates]
                if 'predicted_outcome' in all_predictions.columns:
                    unique_outcomes = sorted(all_predictions['predicted_outcome'].unique().tolist())
            
            return render_template('view_predictions.html', 
                                 predictions_table=predictions_html,
                                 match_count=match_count,
                                 generated_time=file_time.strftime('%Y-%m-%d %H:%M'),
                                 league_filter=league_filter,
                                 date_filter=date_filter,
                                 outcome_filter=outcome_filter,
                                 min_odds_filter=min_odds_filter,
                                 max_odds_filter=max_odds_filter,
                                 unique_leagues=unique_leagues,
                                 unique_dates=unique_dates,
                                 unique_outcomes=unique_outcomes)
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return render_template('view_predictions.html', error=str(e))
    
    return render_template('view_predictions.html', no_predictions=True)




@app.route('/download_predictions')
def download_predictions():
    predictions_file = 'data/predictions/predictions.csv'
    if os.path.exists(predictions_file):
        return send_file(
            predictions_file,
            as_attachment=True,
            download_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
    return redirect(url_for('view_predictions'))

@app.route('/results')
def results():
    phase = request.args.get('phase', '')
    success = request.args.get('success', 'false').lower() == 'true'
    error = request.args.get('error', '')
    
    return render_template('results.html', phase=phase, success=success, error=error)

# Your existing functions (slightly modified to work with the web app)
def run_collection(season, data_types, keep_progress, batch_size, progress_file, collection_phase,
                  selected_regions=None, selected_countries=None, selected_leagues=None, **kwargs):
    """Run collection with unified selection parameters"""
    
    API_KEY = "25c02ce9f07df0edc1e69866fbe7d156"
    collector = CollectPipeline(API_KEY)
    
    # Convert selected leagues to filter_ids
    filter_ids = [int(league_id) for league_id in selected_leagues] if selected_leagues else None
    
    # If regions or countries are selected, we need to get all leagues from those groups
    if selected_regions or selected_countries:
        all_leagues = collector.get_all_leagues()
        league_ids = []
        
        # Add leagues from selected regions
        if selected_regions:
            for region in selected_regions:
                if region in all_leagues:
                    for country_leagues in all_leagues[region].values():
                        league_ids.extend([league['id'] for league in country_leagues])
        
        # Add leagues from selected countries
        if selected_countries:
            for country in selected_countries:
                for region_data in all_leagues.values():
                    if country in region_data:
                        league_ids.extend([league['id'] for league in region_data[country]])
        
        # Combine with individually selected leagues
        if filter_ids:
            league_ids.extend(filter_ids)
        
        filter_ids = list(set(league_ids))  # Remove duplicates
    
    # Collect data
    collector.collect_league_data_filter(
        season=season, 
        data_types=data_types, 
        keep_progress=keep_progress,
        batch_size=batch_size,
        progress_file=progress_file,
        collection_phase=collection_phase,
        filter_ids=filter_ids
    )

def run_extraction():
    extractor = ExtractPipeline()
    extractor.process_all_leagues_seasons()

def run_processing(force_processing=True):
    # Create processing-specific config using your PIPELINE_CONFIG
    PROCESS_CONFIG = {
        'raw_dir': 'data/extracted',
        'merged_dir': 'data/processed',
        'final_output': PIPELINE_CONFIG['final_output'],  # Use your final_output
        'verbose': PIPELINE_CONFIG['verbose'],
        'data_types': {
            'fixtures': 'fixture_events.csv',
            'team_stats': 'team_statistics.csv',
            'odds': 'odds.csv'
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
    processor.run_pipeline(force_processing=force_processing)

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

    trainer.generate_visualizations()

def run_prediction():
    predictor = PredictPipeline()
    training_config = session.get('training_config', TRAIN_CONFIG)
    model_path = f"artifacts/{training_config['model_type']}/{training_config['feature_selection_method']}/models/{training_config['model_type']}_model.pkl"
    
    predictor.load_trained_model(model_path=model_path)
    predictor.predict_upcoming_fixtures()

@app.template_filter('groupby')
def groupby_filter(seq, attribute):
    """Jinja2 filter to group by attribute"""
    groups = {}
    for item in seq:
        key = getattr(item, attribute, None)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups.items()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)