import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocess.preprocess_fixtures import preprocess_fixtures,feature_engineer_fixtures
from preprocess.preprocess_injuries import preprocess_injuries,feature_engineer_injuries
from preprocess.preprocess_lineups import preprocess_lineups,feature_engineer_lineups
from preprocess.preprocess_player_stats import preprocess_player_stats,feature_engineer_player_stats
from preprocess.preprocess_standings import preprocess_team_standings,feature_engineer_team_standings
from preprocess.preprocess_team_stats import preprocess_team_stats,feature_engineer_team_stats

import os
import pandas as pd
from typing import Callable, Dict, Optional, List

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def preprocess_fixtures(data):
    """
    Enhanced fixture preprocessing with time-based and categorical features.
    Includes more sophisticated datetime features and proper handling of missing values.
    """
    # Convert and extract datetime features
    data['date'] = pd.to_datetime(data['date'])
    
    # Basic temporal features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['hour'] = data['date'].dt.hour
    data['is_weekend'] = data['day_of_week'].isin([5,6]).astype(int)
    
    # Advanced temporal features
    data['day_of_year'] = data['date'].dt.dayofyear
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
    
    # Season period with more granular categories
    season_period_map = {
        **{m: 'season_start' for m in [8, 9]},
        **{m: 'season_mid' for m in [10, 11, 12]},
        **{m: 'winter_break' for m in [1]},
        **{m: 'season_mid' for m in [2]},
        **{m: 'season_end' for m in [3, 4, 5]},
        **{m: 'off_season' for m in [6, 7]}
    }
    data['season_period'] = data['month'].map(season_period_map)
    
    # Encode cyclical features
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week']/7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week']/7)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    
    # Handle missing values more robustly
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna('unknown')
    
    return data

def feature_engineer_fixtures(data):
    """
    Enhanced match-level features with advanced metrics and team performance indicators.
    Includes more sophisticated game dynamics features.
    """
    # Basic outcome features
    

    
    # Match outcome encoding
    data['outcome'] = np.select(
        [
            data['home_goals'] > data['away_goals'],
            data['home_goals'] < data['away_goals']
        ],
        [1, 2],  # 1=Home win, 2=Away win
        default=0  # Draw
    )
    
 

    
    # Match importance indicators (simplified)
    data['end_of_season'] = data['month'].isin([4, 5]).astype(int)
    data['start_of_season'] = data['month'].isin([8, 9]).astype(int)
    
 
    
    # Handle missing values
    data = data.fillna(0)
    
    # Ensure numeric columns have appropriate types
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].astype(np.float32)
    
    return data



def preprocess_injuries(data):
    """Preprocess injury data with enhanced cleaning and validation"""
    # Convert and validate date column
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    if data['date'].isna().any():
        print(f"Warning: {data['date'].isna().sum()} rows with invalid dates")
    
    # Clean and standardize injury reasons
    data['injury_reason'] = (
        data['injury_reason']
        .str.strip()
        .str.title()
        .replace({
            'Hamstring Injury': 'Hamstring',
            'Muscle Injury': 'Muscle',
            'Knee Injury': 'Knee',
            'Ankle Injury': 'Ankle'
        })
    )
    
    # Handle missing values more robustly
    data['player_id'] = data['player_id'].fillna(0).astype(int)
    data['team_id'] = data['team_id'].fillna(0).astype(int)
    
    # Enhanced season extraction
    if 'season' in data.columns:
        data['season'] = (
            data['season']
            .astype(str)
            .str.extract(r'(\d{4})')[0]
            .fillna(pd.to_datetime(data['date']).dt.year.astype(str))
        )
    
    return data.dropna(subset=['date']).reset_index(drop=True)

def feature_engineer_injuries(data):
    """Create advanced injury impact metrics with additional features"""
    # Enhanced severity classification with subcategories
    severity_mapping = {
        # High severity (3)
        'Hamstring': 3, 'Acl': 3, 'Mcl': 3, 'Pcl': 3, 
        'Fracture': 3, 'Concussion': 3, 'Groin': 3,
        # Medium severity (2)
        'Muscle': 2, 'Calf': 2, 'Thigh': 2, 'Ankle': 2,
        'Knee': 2, 'Back': 2, 'Shoulder': 2, 'Foot': 2,
        # Low severity (1)
        'Illness': 1, 'Virus': 1, 'Fever': 1, 'Knock': 1,
        'Fatigue': 1, 'Unknown': 1, 'Other': 1
    }
    
    # Create severity features
    data['injury_severity'] = (
        data['injury_reason']
        .map(severity_mapping)
        .fillna(1)  # Default to low severity
        .astype(int)
    )
    
    # Injury duration estimation based on severity
    data['est_recovery_days'] = data['injury_severity'].map({
        1: 3,    # Minor injuries ~3 days
        2: 14,    # Moderate injuries ~2 weeks
        3: 42     # Severe injuries ~6 weeks
    })
    
    # Team-level injury features
    team_injuries = data.groupby(['team_id', 'date']).agg({
        'player_id': 'count',
        'injury_severity': ['sum', 'mean'],
        'est_recovery_days': 'sum'
    }).reset_index()
    
    # Flatten multi-index columns
    team_injuries.columns = [
        'team_id', 'date', 'daily_injury_count', 
        'daily_severity_sum', 'daily_severity_mean',
        'total_recovery_days'
    ]
    
    # Calculate rolling injury burden metrics - FIXED VERSION
    team_injuries = team_injuries.sort_values(['team_id', 'date'])
    
    # Set date as index temporarily for time-based rolling
    team_injuries = team_injuries.set_index('date')
    
    # Define the rolling periods in days
    periods = {
        '7day': 7,
        '30day': 30,
        '90day': 90
    }
    
    for name, window in periods.items():
        # Group by team_id first, then apply rolling within each group
        team_injuries[f'{name}_injury_count'] = (
            team_injuries.groupby('team_id', group_keys=False)['daily_injury_count']
            .apply(lambda x: x.rolling(window=window, min_periods=1).sum())
        )
        
        team_injuries[f'{name}_severity'] = (
            team_injuries.groupby('team_id', group_keys=False)['daily_severity_sum']
            .apply(lambda x: x.rolling(window=window, min_periods=1).sum())
        )
    
    # Reset index to bring date back as a column
    team_injuries = team_injuries.reset_index()
    
    # Calculate injury density (injuries per available player)
    # This would require squad size data for more accuracy
    team_injuries['injury_density'] = (
        team_injuries['30day_injury_count'] / 25  # Assuming 25-player squad
    )
    
    # Merge back with original data
    data = data.merge(team_injuries, on=['team_id', 'date'], how='left')
    
    # Player injury history features
    data = data.sort_values(['player_id', 'date'])
    data['days_since_last_injury'] = (
        data.groupby('player_id')['date']
        .diff()
        .dt.days
        .fillna(365)  # If no previous injury, assume 1 year
    )
    
    # Calculate injury proneness (frequency per player)
    player_injury_counts = (
        data.groupby('player_id')
        .size()
        .reset_index(name='total_injuries')
    )
    
    data = data.merge(player_injury_counts, on='player_id', how='left')
    
    # Injury type features with more categories
    injury_categories = {
        'Muscle': ['Hamstring', 'Calf', 'Groin', 'Thigh', 'Muscle'],
        'Joint': ['Knee', 'Ankle', 'Shoulder'],
        'Illness': ['Illness', 'Virus', 'Fever'],
        'Trauma': ['Fracture', 'Concussion'],
        'Other': ['Unknown', 'Fatigue', 'Knock', 'Other']
    }
    
    # Create category mapping
    injury_type_map = {}
    for category, terms in injury_categories.items():
        for term in terms:
            injury_type_map[term] = category
    
    data['injury_category'] = (
        data['injury_reason']
        .map(injury_type_map)
        .fillna('Other')
    )
    
    # Create dummy variables for injury categories
    injury_dummies = pd.get_dummies(
        data['injury_category'], 
        prefix='injury'
    )
    data = pd.concat([data, injury_dummies], axis=1)
    

    # Enhanced impact score calculation
    data['injury_impact_score'] = (
        data['injury_severity'] * 
        np.log1p(data['total_injuries']) *  # Players with more injuries are more likely to be key players
        (1 + data['30day_severity'] / 10)  # Team injury burden multiplier
    )
    
    return data





def preprocess_lineups(data):
    """Preprocess lineup data with comprehensive error handling"""
    if data.empty:
        return pd.DataFrame()
    
    data = data.copy()
    data.drop(columns=['team_logo', 'player_photo','team_colors_player_primary' ,
                       'team_colors_player_number', 'team_colors_player_border',
                       'team_colors_goalkeeper_primary', 'team_colors_goalkeeper_number' ,
                       'team_colors_goalkeeper_border' , 'coach_photo'], inplace=True, errors='ignore')
    
    # 1. Position Standardization
    if 'player_position' in data.columns:
        position_map = {
            'G': 'GK', 'GK': 'GK', 'Goalkeeper': 'GK',
            'D': 'DF', 'DF': 'DF', 'Defender': 'DF',
            'M': 'MF', 'MF': 'MF', 'Midfielder': 'MF',
            'F': 'FW', 'FW': 'FW', 'Forward': 'FW',
            'ATT': 'FW', 'DEF': 'DF', 'MID': 'MF'
        }
        data['position'] = (
            data['player_position']
            .str.upper()
            .str.strip()
            .map(position_map))
        data['position'] = data['position'].fillna('UNKNOWN')
    else:
        data['position'] = 'UNKNOWN'
    
    # 2. Substitute/Starter Identification
    if 'is_substitute' in data.columns:
        # Handle various formats
        if data['is_substitute'].dtype == 'object':
            data['is_substitute'] = (
                data['is_substitute']
                .str.lower()
                .map({'true': True, 'false': False, 'yes': True, 'no': False})
                .fillna(False)
            )
        data['is_substitute'] = data['is_substitute'].astype(bool)
        data['is_starter'] = ~data['is_substitute']
    else:
        # Inference logic when column missing
        if 'player_grid' in data.columns:
            data['is_starter'] = data['player_grid'].notna()
        elif 'player_number' in data.columns:
            data['is_starter'] = data['player_number'].between(1, 11, inclusive='both')
        else:
            data['is_starter'] = True  # Conservative default
        data['is_substitute'] = ~data['is_starter']
    
    # 3. Formation Processing
    if 'formation' in data.columns:
        data['formation'] = (
            data['formation']
            .astype(str)
            .str.replace(r'[^0-9-]', '', regex=True)
            .str.replace('-', '')
            .str.extract(r'(\d{3,4})')[0]
            .fillna('UNKNOWN')
        )
    else:
        data['formation'] = 'UNKNOWN'
    
    # 4. Grid Coordinates Processing
    if 'player_grid' in data.columns:
        grid_split = (
            data['player_grid']
            .astype(str)
            .str.split(':', expand=True)
            .rename(columns={0: 'grid_x', 1: 'grid_y'})
        )
        grid_split['grid_x'] = pd.to_numeric(grid_split['grid_x'], errors='coerce')
        grid_split['grid_y'] = pd.to_numeric(grid_split['grid_y'], errors='coerce')
        data = pd.concat([data, grid_split], axis=1)
    else:
        data['grid_x'] = np.nan
        data['grid_y'] = np.nan
    
    # 5. Player Number Processing
    if 'player_number' in data.columns:
        data['player_number'] = pd.to_numeric(data['player_number'], errors='coerce')
        data['likely_starter'] = data['player_number'].between(1, 11, inclusive='both')
    else:
        data['player_number'] = 0
        data['likely_starter'] = False
    
    return data

def feature_engineer_lineups(data):
    """Create team-specific lineup features with comprehensive error handling"""
    if data.empty:
        return pd.DataFrame()
    
    # Ensure minimum required columns
    required_cols = ['fixture_id', 'team_id']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    data = data.copy()
    
    # Base aggregation
    agg_dict = {
        'player_id': ['count', 'nunique'],
        'is_substitute': 'sum',
        'is_starter': 'sum',
        'player_number': ['mean', 'std', 'min', 'max']
    }
    
    # Only include existing columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
    
    # Grouping columns
    group_cols = ['fixture_id', 'team_id']
    optional_cols = ['team_name', 'coach_name', 'formation']
    group_cols.extend([col for col in optional_cols if col in data.columns])
    
    # Perform aggregation
    team_features = data.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten multi-index columns
    team_features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                           for col in team_features.columns]
    
    # Column renaming
    rename_map = {
        'player_id_count': 'squad_size',
        'player_id_nunique': 'unique_players',
        'is_substitute_sum': 'substitutes_count',
        'is_starter_sum': 'starters_count',
        'player_number_mean': 'avg_jersey_number',
        'player_number_std': 'std_jersey_number',
        'player_number_min': 'min_jersey_number',
        'player_number_max': 'max_jersey_number'
    }
    team_features = team_features.rename(columns=rename_map)
    
    # Position distribution features
    if 'position' in data.columns:
        try:
            pos_counts = pd.crosstab(
                [data['fixture_id'], data['team_id']],
                data['position'],
                normalize='index'
            ).add_prefix('pos_pct_').reset_index()
            team_features = team_features.merge(pos_counts, on=['fixture_id', 'team_id'], how='left')
        except Exception as e:
            print(f"Position distribution failed: {str(e)}")
    
    # Starting XI features
    if all(col in data.columns for col in ['position', 'is_starter']):
        try:
            starters = data[data['is_starter'] == 1]
            starter_pos = pd.crosstab(
                [starters['fixture_id'], starters['team_id']],
                starters['position'],
                normalize='index'
            ).add_prefix('starter_pct_').reset_index()
            team_features = team_features.merge(starter_pos, on=['fixture_id', 'team_id'], how='left')
        except Exception as e:
            print(f"Starting XI features failed: {str(e)}")
    
    # Grid-based tactical features
    if all(col in data.columns for col in ['grid_x', 'grid_y', 'is_starter']):
        try:
            starters = data[data['is_starter'] == 1]
            if not starters.empty:
                grid_agg = starters.groupby(['fixture_id', 'team_id']).agg({
                    'grid_x': ['mean', 'std'],
                    'grid_y': ['mean', 'std']
                }).reset_index()
                grid_agg.columns = ['fixture_id', 'team_id', 
                                   'avg_position_x', 'position_std_x',
                                   'avg_position_y', 'position_std_y']
                team_features = team_features.merge(grid_agg, on=['fixture_id', 'team_id'], how='left')
        except Exception as e:
            print(f"Grid features failed: {str(e)}")

       # After all feature engineering is done, clean column names
    team_features.columns = team_features.columns.str.rstrip('_')
    
    return team_features




def preprocess_player_stats(data):
    """
    Enhanced preprocessing for player statistics with robust data cleaning
    and comprehensive feature preparation.
    """
    # Standardize column names
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    # Convert numeric columns with better error handling
    numeric_cols = [
        'games_minutes', 'games_rating', 'shots_total', 'shots_on', 
        'goals_total', 'goals_conceded', 'goals_assists', 'goals_saves',
        'passes_total', 'passes_key', 'passes_accuracy', 'tackles_total',
        'tackles_blocks', 'tackles_interceptions', 'duels_total', 'duels_won',
        'dribbles_attempts', 'dribbles_success', 'dribbles_past',
        'fouls_drawn', 'fouls_committed', 'cards_yellow', 'cards_red',
        'penalty_won', 'penalty_commited', 'penalty_scored', 'penalty_missed', 'penalty_saved'
    ]
    
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Enhanced position cleaning with more categories
    position_map = {
        'G': 'Goalkeeper',
        'D': 'Defender',
        'M': 'Midfielder',
        'F': 'Forward'
    }
    
    data['position_group'] = (
        data['games_position'].str.upper().str[0]
        .replace(position_map.keys(), position_map)
        .where(lambda x: x.isin(position_map.values()), 'Unknown')
    )
    
    # Add detailed position categories
    data['position_detail'] = (
        data['games_position'].str.upper()
        .str.extract(r'([A-Z]+)')[0]
        .fillna('UNK')
    )
    
    # Handle missing values intelligently
    fill_values = {
        'games_minutes': 0,
        'games_rating': data['games_rating'].median(),
        'passes_accuracy': data['passes_accuracy'].median(),
        'shots_total': 0,
        'shots_on': 0,
        'goals_total': 0,
        'goals_assists': 0
    }
    
    for col, val in fill_values.items():
        if col in data.columns:
            data[col].fillna(val, inplace=True)
    
    # Convert boolean columns safely
    bool_cols = ['games_captain', 'games_substitute']
    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0).astype(int)
    
    # Ensure non-negative values
    non_negative_cols = [col for col in numeric_cols if col in data.columns]
    data[non_negative_cols] = data[non_negative_cols].clip(lower=0)
    
    return data

def feature_engineer_player_stats(data):
    """
    Advanced feature engineering for player performance metrics with
    position-specific analysis and comprehensive performance indicators.
    """
    eps = 1e-6  # Small constant to avoid division by zero
    
   
    
    
    # 2. Efficiency Metrics
    data['shot_accuracy'] = data['shots_on'] / (data['shots_total'] + eps)
    data['pass_accuracy'] = data['passes_accuracy'] / 100
    data['dribble_success_rate'] = data['dribbles_success'] / (data['dribbles_attempts'] + eps)
    data['duel_success_rate'] = data['duels_won'] / (data['duels_total'] + eps)
    
    # 3. Defensive Metrics
    data['defensive_actions'] = (
        data['tackles_total'] + 
        data['tackles_interceptions'] + 
        data['tackles_blocks']
    )
    data['defensive_actions_per90'] = data['defensive_actions'] / (data['games_minutes'] + eps) * 90
    
  
    

    
  
    # 6. Recent Form Calculations
    data = data.sort_values(['player_id', 'fixture_id'])
    
    form_metrics = [
        'games_rating', 'goals_total', 'goals_assists',
        'defensive_actions', 'passes_key', 'shots_on'
    ]
    
    for metric in form_metrics:
        if metric in data.columns:
            data[f'{metric}_last5_avg'] = (
                data.groupby('player_id')[metric]
                .rolling(5, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
    
    # 7. Consistency Metrics
    data['rating_consistency'] = 1 - (
        data.groupby('player_id')['games_rating']
        .transform(lambda x: x.rolling(10, min_periods=3).std())
        .fillna(0)
    )
    
    # Fill any remaining NA values
    data.fillna(0, inplace=True)
    
    return data    



def preprocess_team_standings(data):
    """
    Enhanced preprocessing for team standings data with robust error handling.
    """
    # Standardize column names
    data.columns = data.columns.str.strip().str.lower()
    
    # Clean form string with comprehensive handling
    if 'form' in data.columns:
        data['form'] = (
            data['form'].astype(str).str.upper()
            .str.replace(r'[^WDL]', '', regex=True)
            .replace(['NAN', 'NONE', 'NULL', ''], 'UNKNOWN')
            .replace('UNKNOWN', '')
        )
    
    # Numeric columns conversion with coersion
    numeric_cols = ['rank', 'points', 'goals_diff', 'played', 'wins', 'draws', 'losses',
                   'goals_for', 'goals_against', 'home_played', 'home_wins', 'home_draws', 
                   'home_losses', 'home_goals_for', 'home_goals_against', 'away_played', 
                   'away_wins', 'away_draws', 'away_losses', 'away_goals_for', 'away_goals_against']
    
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Calculate derived fields if missing
    data['played'] = data.get('played', data['wins'] + data['draws'] + data['losses'])
    data['home_played'] = data.get('home_played', data['home_wins'] + data['home_draws'] + data['home_losses'])
    data['away_played'] = data.get('away_played', data['away_wins'] + data['away_draws'] + data['away_losses'])
    
    # Smart filling of missing values
    fill_rules = {
        'points': 0,
        'goals_diff': 0,
        'form': '',
        'goals_for': data['avg_goals_for'] * data['played'] if 'avg_goals_for' in data.columns else 0,
        'goals_against': data['avg_goals_against'] * data['played'] if 'avg_goals_against' in data.columns else 0
    }
    
    for col, rule in fill_rules.items():
        if col in data.columns:
            data[col].fillna(rule, inplace=True)
    
    # Ensure non-negative values for appropriate columns
    non_negative_cols = ['played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against']
    data[non_negative_cols] = data[non_negative_cols].clip(lower=0)
    
    return data

def feature_engineer_team_standings(data):
    """
    Create advanced team performance metrics from the given standings data.
    """
    # Small value to avoid division by zero
    eps = 1e-6


    # Basic performance metrics
    data['win_pct'] = data['wins'] / (data['played'] + eps)
    data['draw_pct'] = data['draws'] / (data['played'] + eps)
    data['loss_pct'] = data['losses'] / (data['played'] + eps)
    
    # Goal metrics
    data['avg_goals_for'] = data['goals_for'] / (data['played'] + eps)
    data['avg_goals_against'] = data['goals_against'] / (data['played'] + eps)
    data['goal_ratio'] = (data['avg_goals_for'] + eps) / (data['avg_goals_against'] + eps)
    
    # Home performance metrics
    data['home_win_pct'] = data['home_wins'] / (data['home_played'] + eps)
    data['home_draw_pct'] = data['home_draws'] / (data['home_played'] + eps)
    data['home_loss_pct'] = data['home_losses'] / (data['home_played'] + eps)
    data['home_avg_goals_for'] = data['home_goals_for'] / (data['home_played'] + eps)
    data['home_avg_goals_against'] = data['home_goals_against'] / (data['home_played'] + eps)
    
    # Away performance metrics
    data['away_win_pct'] = data['away_wins'] / (data['away_played'] + eps)
    data['away_draw_pct'] = data['away_draws'] / (data['away_played'] + eps)
    data['away_loss_pct'] = data['away_losses'] / (data['away_played'] + eps)
    data['away_avg_goals_for'] = data['away_goals_for'] / (data['away_played'] + eps)
    data['away_avg_goals_against'] = data['away_goals_against'] / (data['away_played'] + eps)
    
    # Venue comparison metrics
    data['home_advantage'] = data['home_win_pct'] - data['away_win_pct']
    data['venue_consistency'] = 1 - (abs(data['home_win_pct'] - data['away_win_pct']))
    
    # Form analysis (using last 5 matches)
    if 'form' in data.columns:
        # Form strength (points per match in last 5)
        data['form_strength'] = data['form'].apply(
            lambda x: (x.count('W')*3 + x.count('D')) / max(1, len(x))
        )
        
        # Recent form (last 3 matches)
        data['recent_form'] = data['form'].str[:3].apply(
            lambda x: (x.count('W')*3 + x.count('D')) / max(1, len(x))
        )
        
        # Corrected form momentum calculation
        data['form_momentum'] = data.apply(
            lambda row: (
                (row['recent_form'] - 
                row['form_strength'] * len(row['form']) - 
                row['recent_form'] * min(3, len(row['form']))) / 
                max(1, len(row['form']) - 3)
            ) if len(row['form']) > 3 else 0,
            axis=1
        )
        
        # Streak analysis
        data['current_streak'] = data['form'].apply(
            lambda x: len(x) - len(x.lstrip(x[0])) if len(x) > 0 else 0
        )
        data['streak_type'] = data['form'].apply(
            lambda x: x[0] if len(x) > 0 else 'N'
        )
    
    # Defense metrics
    data['clean_sheet_pct'] = 1 - ((data['goals_against'] > 0).astype(int) / (data['played'] + eps))
    data['home_clean_sheet_pct'] = 1 - ((data['home_goals_against'] > 0).astype(int) / (data['home_played'] + eps))
    data['away_clean_sheet_pct'] = 1 - ((data['away_goals_against'] > 0).astype(int) / (data['away_played'] + eps))
    
    # Create performance score
    performance_metrics = [
        'points', 'win_pct', 'goals_diff', 'form_strength', 
        'home_advantage', 'venue_consistency'
    ]
    
    # Normalize metrics
    scaler = MinMaxScaler()
    for metric in performance_metrics:
        if metric in data.columns:
            data[f'scaled_{metric}'] = scaler.fit_transform(data[[metric]])
    
    # Weighted performance score
    weights = {
        'scaled_points': 0.3,
        'scaled_win_pct': 0.25,
        'scaled_goals_diff': 0.2,
        'scaled_form_strength': 0.15,
        'scaled_home_advantage': 0.05,
        'scaled_venue_consistency': 0.05
    }
    
    data['performance_score'] = sum(
        data[col] * weight for col, weight in weights.items() if col in data.columns
    )
    
    # Strength categories based on rank
    conditions = [
        (data['rank'] <= 3),
        (data['rank'] <= 6),
        (data['rank'] <= 10),
        (data['rank'] <= 15),
        (data['rank'] > 15)
    ]
    choices = ['Top Tier', 'European Contenders', 'Mid Table', 'Relegation Threatened', 'Bottom Tier']
    data['strength_category'] = np.select(conditions, choices, default='Mid Table')

    data.drop(columns=['wins', 'played', 'draws', 'losses', 'goals_for', 'goals_against,'
    'avg_goals_for', 'avg_goals_against', 'home_wins', 'home_draws', 'home_loses',
    'home_goals_for', 'home_goals_against', 'away_wins', 'away_draws', 'away_loses',
    'away_goals_for', 'away_goals_against'], inplace=True, errors='ignore')
    
    return data



def preprocess_team_stats(data):
    """
    Enhanced preprocessing for team statistics data with robust error handling
    and additional data quality checks.
    """
    # Standardize column names
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

    # Ensure expected_goals exists with proper validation
    if 'expected_goals' not in data.columns:
        data['expected_goals'] = 0.0
    
    # Convert percentage columns with comprehensive handling
    percentage_cols = ['ball_possession', 'passes_%']
    for col in percentage_cols:
        if col in data.columns:
            data[col] = (
                data[col].astype(str)
                .str.replace('%', '')
                .str.replace(',', '.')
                .replace(['nan', 'na', ''], np.nan)
                .astype(float) / 100
            )
    
    # Convert numeric columns with better error handling
    numeric_cols = [
        'shots_on_goal', 'shots_off_goal', 'total_shots', 'blocked_shots',
        'shots_insidebox', 'shots_outsidebox', 'fouls', 'corner_kicks',
        'offsides', 'yellow_cards', 'red_cards', 'goalkeeper_saves',
        'total_passes', 'passes_accurate', 'expected_goals'
    ]
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Smart filling of missing values
    fill_values = {
        'ball_possession': data['ball_possession'].median() if 'ball_possession' in data.columns else 0.5,
        'passes_%': (data['passes_accurate'] / data['total_passes']).clip(0, 1) 
                    if all(x in data.columns for x in ['passes_accurate', 'total_passes']) 
                    else 0.75,
        'expected_goals': 0,
        'shots_on_goal': data['total_shots'] * 0.3 if 'total_shots' in data.columns else 0,
        'goalkeeper_saves': data['shots_on_goal'] * 0.7 if 'shots_on_goal' in data.columns else 0
    }
    
    for col, val in fill_values.items():
        if col in data.columns:
            data[col].fillna(val, inplace=True)
    
    # Calculate basic derived metrics
    data['shot_accuracy'] = data['shots_on_goal'] / (data['shots_on_goal'] + data.get('shots_off_goal', 0)).replace(0, 1)
    data['shot_distribution'] = data.get('shots_insidebox', 0) / (data.get('shots_insidebox', 0) + data.get('shots_outsidebox', 0)).replace(0, 1)
    
    
    # Ensure non-negative values for appropriate columns
    non_negative_cols = [col for col in numeric_cols if col in data.columns]
    data[non_negative_cols] = data[non_negative_cols].clip(lower=0)
    
    return data

def feature_engineer_team_stats(data):
    """
    Enhanced feature engineering for team performance metrics with
    more sophisticated calculations and normalization.
    """
    eps = 1e-6  # Small constant to avoid division by zero
    

    
    # 1. Shooting Metrics
    data['shots_on_target_pct'] = data['shots_on_goal'] / (data['total_shots'] + eps)
    data['box_penetration'] = data.get('shots_insidebox', 0) / (data['total_shots'] + eps)
    data['xg_per_shot'] = data['expected_goals'] / (data['total_shots'] + eps)
    data['shot_efficiency'] = data['shots_on_goal'] / (data['expected_goals'] + eps)
    
    # 2. Passing Metrics
    data['pass_accuracy'] = data['passes_accurate'] / (data['total_passes'] + eps)
    data['passing_efficiency'] = data['pass_accuracy'] * data['ball_possession']
    data['pass_intensity'] = data['total_passes'] / (data['ball_possession'] + eps)
    
    # 3. Defensive Metrics
    defensive_actions = data.get('blocked_shots', 0) + data.get('fouls', 0) + data.get('yellow_cards', 0)
    data['defensive_engagement'] = defensive_actions / (data.get('total_shots', 1) + eps)
    data['defensive_pressure'] = (data.get('fouls', 0) + data.get('yellow_cards', 0)) / (data['ball_possession'] + eps)
    
    # 4. Set Piece Metrics
    data['set_piece_danger'] = (
        0.6 * data.get('corner_kicks', 0) + 
        0.3 * data.get('fouls', 0) + 
        0.1 * data.get('shots_outsidebox', 0)
    )
    
    # 5. Discipline Metrics
    data['discipline_score'] = 1 - (
        0.1 * data['yellow_cards'] + 
        0.3 * data['red_cards']
    ).clip(0, 1)
    
    # 6. Goalkeeping Metrics
    if 'goalkeeper_saves' in data.columns:
        data['save_percentage'] = data['goalkeeper_saves'] / (data['shots_on_goal'] + data['goalkeeper_saves'] + eps)
        data['xg_prevented'] = data['goalkeeper_saves'] - data['expected_goals']
    
    # 7. Style Metrics
    data['directness_index'] = (data.get('shots_outsidebox', 0) / (data['total_shots'] + eps)) * (1 - data['ball_possession'])
    data['possession_efficiency'] = data['expected_goals'] * data['ball_possession']
    data['verticality'] = data.get('shots_insidebox', 0) / (data.get('total_passes', 1) + eps)
    
    # 8. Create Normalized Composite Metrics
    metrics_to_scale = [
        'expected_goals', 'passing_efficiency', 'shot_accuracy',
        'defensive_engagement', 'discipline_score', 'save_percentage',
        'possession_efficiency', 'xg_per_shot'
    ]
    
    scaler = MinMaxScaler()
    for metric in metrics_to_scale:
        if metric in data.columns:
            data[f'scaled_{metric}'] = scaler.fit_transform(data[[metric]])
    
    # 9. Weighted Overall Performance Score
    weights = {
        'scaled_expected_goals': 0.25,
        'scaled_passing_efficiency': 0.2,
        'scaled_shot_accuracy': 0.15,
        'scaled_defensive_engagement': 0.15,
        'scaled_discipline_score': 0.1,
        'scaled_save_percentage': 0.1,
        'scaled_possession_efficiency': 0.05
    }
    
    data['overall_team_performance'] = sum(
        data[col] * weight for col, weight in weights.items() if col in data.columns
    )
    
    # 10. Play Style Classification
    conditions = [
        (data['ball_possession'] > 0.6) & (data['pass_accuracy'] > 0.85),
        (data['ball_possession'] < 0.4) & (data['directness_index'] > 0.5),
        (data['box_penetration'] > 0.7),
        (data['defensive_engagement'] > 0.5)
    ]
    choices = ['Possession', 'Direct', 'Penetrative', 'Defensive']
    data['play_style'] = np.select(conditions, choices, default='Balanced')
    
    # Fill any remaining NA values
    data.fillna(0, inplace=True)

    data.drop(columns=['shots_on_goal', 'shots_off_goal', 'shots_insidebox', 'shots_outsidebox',
                    'expected_goals', 'passes_accurate', 'total_passes', 'passes_accurate',
                    ], inplace=True, errors='ignore')
    
    return data