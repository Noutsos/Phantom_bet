import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
from sklearn.preprocessing import OneHotEncoder
from itertools import groupby  # Need to add at top


class FootballDataPipeline:
    """
    Enhanced pipeline that automatically discovers all leagues and seasons
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dictionary containing pipeline configuration
                   (extracted_dir, processed_dir, merged_dir, etc.)
        """
        # Default configuration
        self.config = {
            'extracted_dir': 'data/extracted',
            'processed_dir': 'data/processed',
            'merged_dir': 'data/merged',
            'final_output': 'data/final_merged_dataset.csv',
            'data_types': None,  # None means process all types
            'verbose': True,
            'window': 5,
            'window_short': 7,  # Short-term rolling window for injuries
            'window_long': 30,  # Long-term rolling window for injuries
            'league_avg_goals': 1.5,
            'min_fixtures': 5
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
            
        # Processing function mappings
        self.process_functions = {
            "fixtures": self._process_fixture_events,
            #"injuries": self._process_injuries,
            "lineups": self._process_lineups,
            #"player_stats": self._process_player_statistics,
            "standings": self._generate_team_standings,
            "team_stats": self._process_team_statistics,
        }
        
        # File name mappings
        self.file_patterns = {
            "fixtures": "fixture_events.csv",
           #"injuries": "injuries.csv",
           "lineups": "lineups.csv",
           # "player_stats": "player_statistics.csv",
            "standings": "fixture_events.csv",
            "team_stats": "team_statistics.csv",
        }
        
        # Processed file name suffixes
        self.processed_suffix = "_processed.csv"
        
        # Cache for fixtures data needed by other processors
        self._fixtures_cache = {}

    def _process_fixture_events(self, df):
        """
        Enhanced fixture processing with syntax corrections and improvements.
        """
        # Configuration parameters
        window = self.config.get('window', 5)
        league_avg_goals = self.config.get('league_avg_goals', 1.5)
        min_matches_for_form = 3
        
        # Data preparation
        df = df.copy()

        df = df.fillna(0)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['home_team_id', 'date'])
        
        # --- Outcome Column ---
        df['outcome'] = np.select(
            condlist=[
                df['home_goals'] > df['away_goals'],
                df['home_goals'] < df['away_goals']
            ],
            choicelist=[1, 2],
            default=0
        )
        
        # --- Temporal Features ---

        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        #df['season_progress'] = df.groupby('season')['date'].rank(pct=True)
        
        # --- Team Form Features ---
        for team_type in ['home', 'away']:
            team_id = f'{team_type}_team_id'
            opp_type = 'away' if team_type == 'home' else 'home'
            
            # Goals metrics
            for metric in ['goals', 'goals_conceded']:
                opp_col = f'{opp_type}_goals' if metric == 'goals_conceded' else f'{team_type}_goals'
                
                # Rolling averages
                df[f'{team_type}_{metric}_rol'] = (df.groupby(team_id)[opp_col]
                                                    .shift(1)
                                                    .rolling(window=window, min_periods=min_matches_for_form)
                                                    .mean()
                                                    .fillna(league_avg_goals))
                
                # Exponential moving average
                df[f'{team_type}_{metric}_ewm'] = (df.groupby(team_id)[opp_col]
                                                .shift(1)
                                                .ewm(span=window, min_periods=min_matches_for_form)
                                                .mean()
                                                .fillna(league_avg_goals))
            
            # Points calculation
            df[f'{team_type}_points'] = np.select(
                [
                    df[f'{team_type}_goals'] > df[f'{opp_type}_goals'],
                    df[f'{team_type}_goals'] == df[f'{opp_type}_goals']
                ],
                [3, 1],
                default=0
            )
            
            # Win/loss stats
            df[f'{team_type}_winner'] = (df[f'{team_type}_goals'] > df[f'{opp_type}_goals']).astype(int)
            df[f'{team_type}_win_streak'] = (df.groupby(team_id)[f'{team_type}_winner']
                                            .shift(1)
                                            .rolling(window, min_periods=1)
                                            .sum()
                                            .fillna(0))
            
            # Clean sheets
            df[f'{team_type}_clean_sheets'] = (df.groupby(team_id)[f'{opp_type}_goals']
                                            .shift(1)
                                            .rolling(window=5, min_periods=1)
                                            .apply(lambda x: sum(x == 0))
                                            .fillna(0))
        
        # --- Opponent-Specific Features ---
        df = df.sort_values(['home_team_id', 'away_team_id', 'date'])
        
        # --- Head-to-Head Features ---
        df = df.sort_values(['home_team_id', 'away_team_id', 'date'])
        df['h2h_home_wins'] = (df.groupby(['home_team_id', 'away_team_id'])['home_winner']
                            .shift(1)
                            .rolling(5, min_periods=1)
                            .sum()
                            .fillna(0))
        
        # --- Game Context Features ---
        df['home_league_position'] = df.groupby(['season', 'home_team_id'])['date'].rank()
        df['away_league_position'] = df.groupby(['season', 'away_team_id'])['date'].rank()
        df['position_difference'] = df['home_league_position'] - df['away_league_position']
        df['is_derby'] = df.apply(lambda x: x['venue_city'] == x.get('away_city', ''), axis=1)
        
        # --- First Match Handling ---
        df['is_home_first_season_match'] = ~df.duplicated(['season', 'home_team_id'], keep='first')
        df['is_away_first_season_match'] = ~df.duplicated(['season', 'away_team_id'], keep='first')
        
        # --- Cleanup ---
        to_drop = [
            'home_winner', 'away_winner', 'home_points', 'away_points',
            'home_league_position', 'away_league_position'
        ]
        return df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')
    
    def _process_lineups(self, lineup_df):
        """
        Processes lineup data into comprehensive team-level features while preventing data leakage.
        Returns engineered features at (fixture_id × team_id) level with enhanced metrics.
        
        Parameters:
            lineup_df (pd.DataFrame): Raw lineup data with shape (17653, 20)
            
        Returns:
            pd.DataFrame: Engineered features ready for modeling with shape (n_fixtures × n_teams, n_features)
        """
        df = lineup_df.copy()
        
        # Validate required columns
        required_cols = ['fixture_id', 'team_id', 'player_id', 'player_pos', 'is_substitute', 'formation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # --- Feature 1: Core Team Metrics ---
        team_features = df.groupby(['fixture_id', 'team_id']).agg({
            'player_id': ['count', 'nunique'],  # Squad size and unique players (handles duplicates)
            'is_substitute': 'sum',
            'formation': 'first',
            'coach_id': 'first',
            'coach_name': 'first'
        })
        team_features.columns = ['_'.join(col).strip() for col in team_features.columns.values]
        team_features = team_features.rename(columns={
            'player_id_count': 'squad_size',
            'player_id_nunique': 'unique_players',
            'is_substitute_sum': 'substitutes_count',
            'formation_first': 'formation',
            'coach_id_first': 'coach_id',
            'coach_name_first': 'coach_name'
        }).reset_index()

        # --- Feature 2: Advanced Position Analysis ---
        # Standardize position categories
        df['position_category'] = df['player_pos'].str[0].replace({
            'G': 'GK', 'D': 'DF', 'M': 'MF', 'F': 'FW'
        }).fillna('Other')
        
        # Starting XI position distribution
        pos_features = (
            df[~df['is_substitute']]
            .groupby(['fixture_id', 'team_id', 'position_category'])
            .size()
            .unstack(fill_value=0)
            .add_prefix('start_')
            .reset_index()
        )
        pos_features = pos_features.drop(columns=['start_Other'], errors='ignore')  # Remove 'Other' if exists
        
        # Substitute position distribution
        sub_pos_features = (
            df[df['is_substitute']]
            .groupby(['fixture_id', 'team_id', 'position_category'])
            .size()
            .unstack(fill_value=0)
            .add_prefix('sub_')
            .reset_index()
        )
        sub_pos_features = sub_pos_features.drop(columns=['sub_Other'], errors='ignore')  # Remove 'Other' if exists

        # --- Feature 3: Formation Intelligence ---
        # Clean and standardize formations
        team_features['formation_clean'] = (
            team_features['formation']
            .str.replace('-', '')
            .str.extract(r'(\d{3,4})')[0]  # Extract core formation numbers
            .fillna('000')
        )
        
        # Formation one-hot encoding
        top_formations = ['442', '433', '352', '4231', '343', '532']
        for form in top_formations:
            team_features[f'formation_{form}'] = (team_features['formation_clean'] == form).astype(int)
        
        # Formation flexibility metric
        team_features['formation_flexibility'] = (
            team_features['formation_clean']
            .apply(lambda x: len(set(x)))  # Count unique digits
        )

        # --- Feature 4: Squad Balance Metrics ---
        # Positional balance indicators
        pos_features['defensive_ratio'] = pos_features['start_DF'] / (pos_features['start_MF'] + 1e-6)
        pos_features['attack_ratio'] = pos_features['start_FW'] / (pos_features['start_MF'] + 1e-6)
        
        # Substitute quality metrics
        pos_strength = {
            'GK': 1, 'DF': 2, 'MF': 3, 'FW': 4, 'Other': 0
        }
        df['pos_value'] = df['position_category'].map(pos_strength)
        
        sub_quality = (
            df[df['is_substitute']]
            .groupby(['fixture_id', 'team_id'])
            .agg({
                'pos_value': ['mean', 'sum', 'max'],
                'player_id': 'count'
            })
        )
        sub_quality.columns = ['_'.join(col).strip() for col in sub_quality.columns.values]
        sub_quality = sub_quality.rename(columns={
            'pos_value_mean': 'sub_quality_avg',
            'pos_value_sum': 'sub_quality_total',
            'pos_value_max': 'sub_quality_max',
            'player_id_count': 'sub_count'
        }).reset_index()

        # --- Feature 5: Player Grid Analysis ---
        if 'player_grid' in df.columns:
            try:
                # Extract grid positions if available
                df[['grid_x', 'grid_y']] = (
                    df['player_grid']
                    .str.split(',', expand=True)
                    .astype(float)
                )
                
                grid_features = (
                    df[~df['is_substitute']]  # Only starting players
                    .groupby(['fixture_id', 'team_id'])
                    .agg({
                        'grid_x': ['mean', 'std', 'min', 'max'],
                        'grid_y': ['mean', 'std', 'min', 'max']
                    })
                )
                grid_features.columns = ['_'.join(col).strip() for col in grid_features.columns.values]
                grid_features = grid_features.reset_index()
            except:
                pass

        # --- Feature Merging ---
        # Combine all features
        final_features = team_features.merge(
            pos_features,
            on=['fixture_id', 'team_id'],
            how='left'
        ).merge(
            sub_pos_features,
            on=['fixture_id', 'team_id'],
            how='left'
        ).merge(
            sub_quality,
            on=['fixture_id', 'team_id'],
            how='left'
        )

        # Add grid features if they exist
        if 'grid_features' in locals():
            final_features = final_features.merge(
                grid_features,
                on=['fixture_id', 'team_id'],
                how='left'
            )

        # Fill NA values
        numeric_cols = final_features.select_dtypes(include=np.number).columns
        final_features[numeric_cols] = final_features[numeric_cols].fillna(0)
        
        # Drop intermediate columns
        final_features = final_features.drop(columns=['formation_clean'], errors='ignore')
       
        
        return final_features
    
    def _process_injuries(self, injuries_df):
        """
        Processes injury data with comprehensive feature engineering.
        Returns engineered features at (fixture_id × team_id) level with:
        - Current injury status
        - Historical injury patterns
        - Injury type analysis
        - Player importance weighting (if available)
        
        Args:
            injuries_df: DataFrame containing injury data (shape: 2660, 12)
            
        Returns:
            DataFrame with enhanced injury features
        """
        # Configuration
        window_short = self.config.get('window_short', 7)
        window_long = self.config.get('window_long', 30)
        importance_threshold = self.config.get('importance_threshold', 0.7)
        
        df = injuries_df.copy()
        
        # Validate required columns
        required_cols = ['fixture_id', 'date', 'team_id', 'player_id', 'type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert and sort data
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['team_id', 'date'])

        # --- Feature 1: Current Match Injury Status ---
        current_features = df.groupby(['fixture_id', 'team_id', 'date']).agg({
            'player_id': 'count',
            'type': [
                ('current_serious', lambda x: x.str.contains('Fracture|ACL|Rupture|Surgery', case=False, na=False).sum()),
                ('current_muscle', lambda x: x.str.contains('Muscle|Strain|Tear', case=False, na=False).sum()),
                ('current_minor', lambda x: x.str.contains('Illness|Knock|Fatigue', case=False, na=False).sum())
            ]
        })
        current_features.columns = ['_'.join(col).strip() for col in current_features.columns.values]
        current_features = current_features.rename(columns={
            'player_id_count': 'current_injuries',
            'type_current_serious': 'current_serious_injuries',
            'type_current_muscle': 'current_muscle_injuries',
            'type_current_minor': 'current_minor_injuries'
        }).reset_index()

        # --- Feature 2: Historical Injury Patterns ---
        daily_counts = (
            df.groupby(['team_id', pd.Grouper(key='date', freq='D')])
            .agg({
                'player_id': 'count',
                'type': [
                    ('serious', lambda x: x.str.contains('Fracture|ACL|Rupture|Surgery', case=False, na=False).sum()),
                    ('muscle', lambda x: x.str.contains('Muscle|Strain|Tear', case=False, na=False).sum())
                ]
            })
        )
        daily_counts.columns = ['_'.join(col).strip() for col in daily_counts.columns.values]
        daily_counts = daily_counts.rename(columns={
            'player_id_count': 'daily_injuries',
            'type_serious': 'daily_serious',
            'type_muscle': 'daily_muscle'
        }).reset_index()

        def calculate_rolling_metrics(group):
            group = group.set_index('date').sort_index()
            
            for window in [window_short, window_long]:
                group[f'injuries_{window}d'] = group['daily_injuries'].rolling(f'{window}D').sum()
                group[f'serious_{window}d'] = group['daily_serious'].rolling(f'{window}D').sum()
                group[f'muscle_{window}d'] = group['daily_muscle'].rolling(f'{window}D').sum()
                group[f'injury_rate_{window}d'] = group[f'injuries_{window}d'] / float(window)
                group[f'serious_rate_{window}d'] = group[f'serious_{window}d'] / float(window)
            
            group['injury_trend'] = group[f'injuries_{window_short}d'] / (group[f'injuries_{window_long}d'] + 1e-6)
            return group.reset_index()

        rolling_features = (
            daily_counts
            .groupby('team_id', group_keys=False)
            .apply(calculate_rolling_metrics)
            .reset_index(drop=True)
        )

        # --- Feature 3: Player Importance Weighting ---
        if 'player_importance' in df.columns or hasattr(self, 'player_importance_dict'):
            if 'player_importance' not in df.columns:
                df['player_importance'] = df['player_id'].map(
                    getattr(self, 'player_importance_dict', defaultdict(lambda: 0.5)))
            
            weighted_injuries = (
                df.groupby(['team_id', pd.Grouper(key='date', freq='D')])
                .apply(lambda x: pd.Series({
                    'weighted_injuries': x['player_importance'].sum(),
                    'key_players_injured': (x['player_importance'] > importance_threshold).sum()
                }))
                .reset_index()
            )
            
            rolling_features = rolling_features.merge(
                weighted_injuries,
                on=['team_id', 'date'],
                how='left'
            ).fillna(0)

            def add_weighted_rolling(group):
                group = group.set_index('date').sort_index()
                group[f'weighted_{window_long}d'] = group['weighted_injuries'].rolling(f'{window_long}D').sum()
                group[f'key_players_{window_long}d'] = group['key_players_injured'].rolling(f'{window_long}D').sum()
                return group.reset_index()
            
            rolling_features = (
                rolling_features
                .groupby('team_id', group_keys=False)
                .apply(add_weighted_rolling)
                .reset_index(drop=True)
            )

        # --- Feature 4: Injury Clustering Analysis ---
        injury_spikes = (
            daily_counts[daily_counts['daily_injuries'] >= 3]
            .groupby('team_id')
            .apply(lambda x: x.set_index('date')['daily_injuries'])
        )
        
        def days_since_last_spike(group):
            last_spike = injury_spikes.get(group.name, pd.Series()).index
            if len(last_spike) > 0:
                group['days_since_spike'] = (group['date'] - last_spike[-1]).dt.days
            else:
                group['days_since_spike'] = 365
            return group
        
        rolling_features = (
            rolling_features
            .groupby('team_id', group_keys=False)
            .apply(days_since_last_spike)
            .fillna(365)
        )

        # --- Combine All Features ---
        features = current_features.merge(
            rolling_features,
            on=['team_id', 'date'],
            how='left'
        ).fillna(0)

        # Calculate composite injury burden score
        features['injury_burden'] = (
            0.3 * features['current_serious_injuries'] +
            0.25 * features[f'serious_{window_long}d'] +
            0.2 * features[f'muscle_{window_long}d'] +
            0.15 * features.get('weighted_injuries', 0) +
            0.1 * features['injury_trend']
        )

        # Final feature selection
        final_features = [
            'fixture_id', 'team_id', 'date',
            'current_injuries', 'current_serious_injuries', 
            'current_muscle_injuries', 'current_minor_injuries',
            f'injuries_{window_short}d', f'injuries_{window_long}d',
            f'serious_{window_short}d', f'serious_{window_long}d',
            f'muscle_{window_short}d', f'muscle_{window_long}d',
            'injury_rate_7d', 'serious_rate_30d',
            'injury_trend', 'days_since_spike',
            'injury_burden'
        ]
        
        if 'weighted_injuries' in features.columns:
            final_features.extend([
                'weighted_injuries', f'weighted_{window_long}d',
                'key_players_injured', f'key_players_{window_long}d'
            ])

        return features[final_features].drop_duplicates()
    
    def _process_team_statistics_2(self, stats_df, fixtures_df):
        """
        Processes team stats, keeping ONLY rolling averages (prefixed with 'rol_').
        Removes original columns used for aggregation.
        """
        window = self.config.get('window', 5)
        
        # --- Input Validation (unchanged) ---
        required_stats = ['fixture_id', 'team_id']
        missing = [col for col in required_stats if col not in stats_df.columns]
        if missing:
            raise ValueError(f"Missing columns in stats_df: {missing}")
        
        required_fixtures = ['fixture_id', 'date']
        missing = [col for col in required_fixtures if col not in fixtures_df.columns]
        if missing:
            raise ValueError(f"Missing columns in fixtures_df: {missing}")

        # --- Data Cleaning (unchanged) ---

        stats_df = stats_df.fillna(0)
        df = stats_df.copy()
        df = df.fillna(0)
        fixtures_df = fixtures_df.copy()
        
        def clean_column_names(df):
            df.columns = (
                df.columns.str.lower()
                .str.replace(r'[^a-z0-9_]', '_', regex=True)
                .str.replace(r'_+', '_', regex=True)
                .str.strip('_')
            )
            return df
        
        df = clean_column_names(df)
        fixtures_df = clean_column_names(fixtures_df)

        # Explicit removal of specific columns
        columns_to_remove = ['expected_goals', 'goals_prevented']
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
        
        # --- Type Conversion (unchanged) ---
        df['fixture_id'] = df['fixture_id'].astype(str).str.strip()
        df['team_id'] = df['team_id'].astype(str).str.strip()
        fixtures_df['fixture_id'] = fixtures_df['fixture_id'].astype(str).str.strip()
        
        # --- Numeric Columns Handling (unchanged) ---
        numeric_cols = []
        for col in df.columns:
            if col not in ['fixture_id', 'team_id']:
                if df[col].dtype == 'object' and df[col].astype(str).str.contains('%').any():
                    df[col] = (
                        df[col].astype(str)
                        .str.replace('%', '')
                        .str.replace(',', '.')
                        .astype(float) / 100
                    )
                    numeric_cols.append(col)
                else:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except (ValueError, TypeError):
                        continue

        # --- Merge with Fixtures (unchanged) ---
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'], errors='coerce', utc=True)
        if fixtures_df['date'].isnull().any():
            invalid_dates = fixtures_df[fixtures_df['date'].isnull()]
            raise ValueError(f"Invalid date values found in rows: {invalid_dates.index.tolist()}")
        
        df = df.merge(
            fixtures_df[['fixture_id', 'date']],
            on='fixture_id',
            how='left',
            validate='many_to_one'
        )
        
        # --- Sort Data (unchanged) ---
        df = df.sort_values(['team_id', 'date'])

                # --- Derived Features (unchanged) ---
        if all(col in df.columns for col in ['shots_on_goal', 'total_shots']):
            df['shot_accuracy'] = (df['shots_on_goal'] / df['total_shots'].replace(0, 1)).clip(0, 1)

        if all(col in df.columns for col in ['goals', 'total_shots']):
            df['shot_efficiency'] = df['goals'] / df['total_shots'].replace(0, 1)

        if all(col in df.columns for col in ['goals_conceded', 'shots_on_target_against']):
            df['defensive_efficiency'] = 1 - (df['goals_conceded'] / df['shots_on_target_against'].replace(0, 1))

        if all(col in df.columns for col in ['goals', 'ball_possession']):
            df['goals_per_possession'] = df['goals'] / (df['ball_possession'] / 100).replace(0, 1)
        
        # --- Rolling Averages (MODIFIED TO REMOVE ORIGINAL COLS) ---
        rolling_features = []
        for team in df['team_id'].unique():
            team_data = df[df['team_id'] == team].copy()
            
            for col in numeric_cols:
                # Calculate rolling average
                team_data[f'rol_tm_{col}'] = (
                    team_data[col].shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
                # Remove the original column
                team_data.drop(col, axis=1, inplace=True)
            
            rolling_features.append(team_data)
        
        # Combine all teams
        df = pd.concat(rolling_features).sort_values(['team_id', 'date'])
        

        
        defensive_cols = [c for c in ['fouls', 'blocked_shots', 'goalkeeper_saves'] if c in df.columns]
        if defensive_cols:
            df['defensive_actions'] = df[defensive_cols].sum(axis=1)
        
        if 'ball_possession' in df.columns:
            df['playing_style'] = pd.cut(
                df['ball_possession'],
                bins=[0, 40, 60, 100],
                labels=['counter', 'balanced', 'possession']
            ).astype(object)
        
        # --- Select Only Rolling + Derived Features ---
        keep_cols = ['fixture_id', 'team_id'] + [f'rol_tm_{col}' for col in numeric_cols]
        derived_features = ['shot_accuracy', 'defensive_actions', 'playing_style']
        keep_cols += [f for f in derived_features if f in df.columns]
        
        return df[keep_cols].fillna(0)
 
    def _process_team_statistics(self, stats_df, fixtures_df):
        """
        Processes team stats, keeping ONLY rolling averages (prefixed with 'rol_').
        Removes original columns used for aggregation after all calculations are done.
        """
        window = self.config.get('window', 5)
        
        # --- Input Validation ---
        required_stats = ['fixture_id', 'team_id']
        missing = [col for col in required_stats if col not in stats_df.columns]
        if missing:
            raise ValueError(f"Missing columns in stats_df: {missing}")
        
        required_fixtures = ['fixture_id', 'date']
        missing = [col for col in required_fixtures if col not in fixtures_df.columns]
        if missing:
            raise ValueError(f"Missing columns in fixtures_df: {missing}")

        # --- Data Cleaning ---
        stats_df = stats_df.fillna(0)
        df = stats_df.copy()
        fixtures_df = fixtures_df.copy()
        
        def clean_column_names(df):
            df.columns = (
                df.columns.str.lower()
                .str.replace(r'[^a-z0-9_]', '_', regex=True)
                .str.replace(r'_+', '_', regex=True)
                .str.strip('_')
            )
            return df
        
        df = clean_column_names(df)
        fixtures_df = clean_column_names(fixtures_df)

        # Explicit removal of specific columns
        columns_to_remove = ['expected_goals', 'goals_prevented']
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
        
        # --- Type Conversion ---
        df['fixture_id'] = df['fixture_id'].astype(str).str.strip()
        df['team_id'] = df['team_id'].astype(str).str.strip()
        fixtures_df['fixture_id'] = fixtures_df['fixture_id'].astype(str).str.strip()
        
        # --- Numeric Columns Handling ---
        numeric_cols = []
        for col in df.columns:
            if col not in ['fixture_id', 'team_id']:
                if df[col].dtype == 'object' and df[col].astype(str).str.contains('%').any():
                    df[col] = (
                        df[col].astype(str)
                        .str.replace('%', '')
                        .str.replace(',', '.')
                        .astype(float) / 100
                    )
                    numeric_cols.append(col)
                else:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except (ValueError, TypeError):
                        continue

        # --- Merge with Fixtures ---
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'], errors='coerce', utc=True)
        if fixtures_df['date'].isnull().any():
            invalid_dates = fixtures_df[fixtures_df['date'].isnull()]
            raise ValueError(f"Invalid date values found in rows: {invalid_dates.index.tolist()}")
        
        df = df.merge(
            fixtures_df[['fixture_id', 'date']],
            on='fixture_id',
            how='left',
            validate='many_to_one'
        )
        
        # --- Sort Data ---
        df = df.sort_values(['team_id', 'date'])

        # --- Create ALL Derived Features FIRST ---
        derived_features = []
        
        if all(col in df.columns for col in ['shots_on_goal', 'total_shots']):
            df['shot_accuracy'] = (df['shots_on_goal'] / df['total_shots'].replace(0, 1)).clip(0, 1)
            derived_features.append('shot_accuracy')

        if all(col in df.columns for col in ['goals', 'total_shots']):
            df['shot_efficiency'] = df['goals'] / df['total_shots'].replace(0, 1)
            derived_features.append('shot_efficiency')

        if all(col in df.columns for col in ['goals_conceded', 'shots_on_target_against']):
            df['defensive_efficiency'] = 1 - (df['goals_conceded'] / df['shots_on_target_against'].replace(0, 1))
            derived_features.append('defensive_efficiency')

        if all(col in df.columns for col in ['goals', 'ball_possession']):
            df['goals_per_possession'] = df['goals'] / (df['ball_possession'] / 100).replace(0, 1)
            derived_features.append('goals_per_possession')
        
        defensive_cols = [c for c in ['fouls', 'blocked_shots', 'goalkeeper_saves'] if c in df.columns]
        if defensive_cols:
            df['defensive_actions'] = df[defensive_cols].sum(axis=1)
            derived_features.append('defensive_actions')
        
        if 'ball_possession' in df.columns:
            df['playing_style'] = pd.cut(
                df['ball_possession'],
                bins=[0, 40, 60, 100],
                labels=['counter', 'balanced', 'possession']
            ).astype(object)
            derived_features.append('playing_style')

        # --- Calculate Rolling Averages ---
        rolling_features = []
        for team in df['team_id'].unique():
            team_data = df[df['team_id'] == team].copy()
            
            # Calculate rolling for both original and derived features
            for col in numeric_cols + derived_features:
                if col in team_data.columns:
                    team_data[f'rol_tm_{col}'] = (
                        team_data[col].shift(1)
                        .rolling(window, min_periods=1)
                        .mean()
                    )
            
            rolling_features.append(team_data)
        
        # Combine all teams
        df = pd.concat(rolling_features).sort_values(['team_id', 'date'])
        
        # --- Final Column Selection ---
        # Keep only rolling versions of all features (original + derived)
        keep_cols = ['fixture_id', 'team_id'] + [f'rol_tm_{col}' for col in numeric_cols + derived_features]
        
        return df[keep_cols].fillna(0)
    
    def _process_player_statistics(self, player_stats_df, fixtures_df):
        """
        Processes player stats with comprehensive rolling features and performance metrics.
        Returns enhanced player features at (fixture_id × player_id) level.
        
        Parameters:
            player_stats_df: DataFrame with player match stats (shape: 17650, 39)
            fixtures_df: DataFrame with fixture dates
            
        Returns:
            DataFrame with rolling features and performance metrics
        """
        # Configuration
        window = self.config.get('window', 5)
        
        # Validate inputs
        required_player_cols = ['fixture_id', 'team_id', 'player_id', 'games_rating']
        missing = [col for col in required_player_cols if col not in player_stats_df.columns]
        if missing:
            raise ValueError(f"Missing columns in player_stats_df: {missing}")
        
        if not all(col in fixtures_df.columns for col in ['fixture_id', 'date']):
            raise ValueError("fixtures_df must contain 'fixture_id' and 'date'")

        # Create working copies
        df = player_stats_df.copy()
        fixtures_df = fixtures_df.copy()
        
        # --- Data Cleaning ---
        player_stats_df = player_stats_df.fillna(0)

        # Convert IDs to consistent string type
        for col in ['fixture_id', 'team_id', 'player_id']:
            df[col] = df[col].astype(str).str.strip()
            if col in fixtures_df.columns:
                fixtures_df[col] = fixtures_df[col].astype(str).str.strip()
        
        # Convert percentages and numeric columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].astype(str).str.contains('%').any():
                try:
                    df[col] = (
                        df[col].astype(str)
                        .str.replace('%', '')
                        .str.replace(',', '.')
                        .astype(float) / 100)
                except ValueError:
                    continue

        # --- Data Merging ---
        # Convert and validate date column
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'], errors='coerce')
        if fixtures_df['date'].isnull().any():
            invalid_dates = fixtures_df[fixtures_df['date'].isnull()]
            raise ValueError(f"Invalid date values found in rows: {invalid_dates.index.tolist()}")
        
        # Merge with fixtures
        try:
            df = df.merge(
                fixtures_df[['fixture_id', 'date']],
                on='fixture_id',
                how='left',
                validate='many_to_one'
            )
        except pd.errors.MergeError as e:
            mismatches = set(df['fixture_id']) - set(fixtures_df['fixture_id'])
            raise ValueError(f"Merge failed: {str(e)}. Missing fixture_ids: {list(mismatches)[:10]}...")

        # Sort chronologically by player
        df = df.sort_values(['player_id', 'date'])

        # --- Feature Engineering ---
        # Select numeric performance columns
        exclude_cols = ['fixture_id', 'team_id', 'player_id', 'games_number', 'games_minutes']
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                        if col not in exclude_cols]
        
        # Initialize list for storing processed player data
        rolling_features = []
        
        # Process each player separately
        for player in df['player_id'].unique():
            player_data = df[df['player_id'] == player].copy()
            
            # Calculate rolling stats (excluding current match)
            for col in numeric_cols:
                player_data[f'rol_pl_{col}'] = (
                    player_data[col].shift(1)  # Exclude current match
                    .rolling(window, min_periods=1)
                    .mean()
                )
            
            # Add minutes-weighted features for key metrics
            if 'games_minutes' in player_data.columns:
                for metric in ['games_rating', 'passes_accuracy', 'shots_on']:
                    if metric in numeric_cols:
                        player_data[f'weighted_pl_{metric}'] = (
                            (player_data[metric] * player_data['games_minutes']).shift(1)
                            .rolling(window, min_periods=1)
                            .sum() / 
                            player_data['games_minutes'].shift(1)
                            .rolling(window, min_periods=1)
                            .sum()
                        ).fillna(0)
            
            # Add recent form indicators
            if 'games_rating' in numeric_cols:
                player_data['form_rating'] = (
                    player_data['games_rating'].shift(1)
                    .rolling(3, min_periods=1)
                    .mean()
                )
            
            rolling_features.append(player_data)

        # Combine all players
        df = pd.concat(rolling_features).sort_values(['player_id', 'date'])

        # --- Enhanced Derived Features ---
        # Performance metrics
        if all(col in df.columns for col in ['shots_on', 'shots_total']):
            df['shot_efficiency'] = (
                df['shots_on'] / df['shots_total'].replace(0, np.nan)
            ).fillna(0)
            df['rol_shot_efficiency'] = (
                df['shot_efficiency'].shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        
        # Defensive metrics
        defensive_cols = ['tackles_total', 'tackles_interceptions', 'tackles_blocks']
        defensive_cols = [col for col in defensive_cols if col in df.columns]
        if defensive_cols:
            df['defensive_impact'] = df[defensive_cols].sum(axis=1)
            df['rol_defensive_impact'] = (
                df['defensive_impact'].shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        
        # Creative metrics
        if all(col in df.columns for col in ['passes_key', 'goals_assists']):
            df['creative_impact'] = df['passes_key'] + df['goals_assists']
            df['rol_creative_impact'] = (
                df['creative_impact'].shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        
        # Positional features
        if 'games_position' in df.columns:
            # Simplify positions
            df['position_group'] = df['games_position'].str.extract(r'([A-Za-z]+)')[0]
            position_dummies = pd.get_dummies(df['position_group'], prefix='pos')
            df = pd.concat([df, position_dummies], axis=1)
        
        # Consistency metrics
        if 'games_rating' in df.columns:
            df['rating_consistency'] = (
                df['games_rating'].shift(1)
                .rolling(window, min_periods=3)
                .std()
            ).fillna(0)
        
        # --- Final Feature Selection ---
        base_cols = ['fixture_id', 'team_id', 'player_id', 'date']
        rolling_cols = [col for col in df.columns if col.startswith('rol_pl_')]
        weighted_cols = [col for col in df.columns if col.startswith('weighted_pl_')]
        derived_cols = [
            col for col in df.columns if col in [
                'shot_efficiency', 'defensive_impact', 
                'creative_impact', 'form_rating',
                'rating_consistency'
            ]
        ]
        positional_cols = [col for col in df.columns if col.startswith('pos_')]
        
        keep_cols = base_cols + rolling_cols + weighted_cols + derived_cols + positional_cols
        
        return df[keep_cols].fillna(0)
    
    def _generate_team_standings(self, fixtures_df):
        # Sort fixtures by date to process in chronological order
        fixtures_df = fixtures_df.sort_values('date')
        
        # Get all unique teams in the league
        all_teams = set(fixtures_df['home_team_id'].unique()).union(set(fixtures_df['away_team_id'].unique()))
        
        # Initialize team standings dictionaries with all required keys
        team_standings = defaultdict(lambda: {
            'fixture_id': None,
            'points': 0,
            'goals_diff': 0,
            'goals_for': 0,
            'goals_against': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'home_played': 0,
            'home_wins': 0,
            'home_draws': 0,
            'home_losses': 0,
            'home_goals_for': 0,
            'home_goals_against': 0,
            'away_played': 0,
            'away_wins': 0,
            'away_draws': 0,
            'away_losses': 0,
            'away_goals_for': 0,
            'away_goals_against': 0,
            'form': '',
            'matches_history': [],
            'team_name': ''
    
        })
        
        # Initialize team names and ensure all teams are in standings
        for _, row in fixtures_df.iterrows():
            team_standings[row['home_team_id']]['team_name'] = row['home_team']
            team_standings[row['away_team_id']]['team_name'] = row['away_team']
        
        # Prepare output dataframe
        output_rows = []
        
        for _, row in fixtures_df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            
            # Calculate rankings before this match - include ALL teams
            teams_list = []
            for team_id in all_teams:
                stats = team_standings[team_id]
                teams_list.append({
                    'fixture_id': stats['fixture_id'],  # Use fixture_id from the first match
                    'team_id': team_id,
                    'team_name': stats['team_name'],
                    'points': stats['points'],
                    'goals_diff': stats['goals_diff'],
                    'goals_for': stats['goals_for'],
                    'goals_against': stats['goals_against'],
                    'played': stats['played']
                })
            
            # Sort teams according to standard football ranking criteria
            teams_sorted = sorted(teams_list, 
                                key=lambda x: (-x['points'], -x['goals_diff'], -x['goals_for'], x['team_name']))
            
            # Create rank mapping - handle ties properly
            rank_dict = {}
            current_rank = 1
            for i, team in enumerate(teams_sorted):
                if i > 0 and (teams_sorted[i]['points'] == teams_sorted[i-1]['points'] and
                            teams_sorted[i]['goals_diff'] == teams_sorted[i-1]['goals_diff'] and
                            teams_sorted[i]['goals_for'] == teams_sorted[i-1]['goals_for']):
                    rank_dict[team['team_id']] = current_rank
                else:
                    current_rank = i + 1
                    rank_dict[team['team_id']] = current_rank
            
            # Get pre-match standings for both teams (including rank)
            home_pre = {**team_standings[home_id].copy(), 'rank': rank_dict.get(home_id, None)}
            away_pre = {**team_standings[away_id].copy(), 'rank': rank_dict.get(away_id, None)}
            
            # Create separate rows for home and away teams
            # Home team row
            output_rows.append({
                'fixture_id': row['fixture_id'],
                'rank': home_pre['rank'],
                'team_id': home_id,
                'team_name': home_pre['team_name'],
                'points': home_pre['points'],
                'goals_diff': home_pre['goals_diff'],
                'form': home_pre['form'],
                'played': home_pre['played'],
                'wins': home_pre['wins'],
                'draws': home_pre['draws'],
                'losses': home_pre['losses'],
                'goals_for': home_pre['goals_for'],
                'goals_against': home_pre['goals_against'],
                'home_played': home_pre['home_played'],
                'home_wins': home_pre['home_wins'],
                'home_draws': home_pre['home_draws'],
                'home_losses': home_pre['home_losses'],
                'home_goals_for': home_pre['home_goals_for'],
                'home_goals_against': home_pre['home_goals_against'],
                'away_played': home_pre['away_played'],
                'away_wins': home_pre['away_wins'],
                'away_draws': home_pre['away_draws'],
                'away_losses': home_pre['away_losses'],
                'away_goals_for': home_pre['away_goals_for'],
                'away_goals_against': home_pre['away_goals_against'],
                
            })
            
            # Away team row
            output_rows.append({
                'fixture_id': row['fixture_id'],
                'rank': away_pre['rank'],
                'team_id': away_id,
                'team_name': away_pre['team_name'],
                'points': away_pre['points'],
                'goals_diff': away_pre['goals_diff'],
                'form': away_pre['form'],
                'played': away_pre['played'],
                'wins': away_pre['wins'],
                'draws': away_pre['draws'],
                'losses': away_pre['losses'],
                'goals_for': away_pre['goals_for'],
                'goals_against': away_pre['goals_against'],
                'home_played': away_pre['home_played'],
                'home_wins': away_pre['home_wins'],
                'home_draws': away_pre['home_draws'],
                'home_losses': away_pre['home_losses'],
                'home_goals_for': away_pre['home_goals_for'],
                'home_goals_against': away_pre['home_goals_against'],
                'away_played': away_pre['away_played'],
                'away_wins': away_pre['away_wins'],
                'away_draws': away_pre['away_draws'],
                'away_losses': away_pre['away_losses'],
                'away_goals_for': away_pre['away_goals_for'],
                'away_goals_against': away_pre['away_goals_against'],
            })
            
            # Process match result and update standings (same as before)
            home_goals = row['home_goals']
            away_goals = row['away_goals']
            
            # Update home team stats
            team_standings[home_id]['played'] += 1
            team_standings[home_id]['home_played'] += 1
            team_standings[home_id]['goals_for'] += home_goals
            team_standings[home_id]['goals_against'] += away_goals
            team_standings[home_id]['home_goals_for'] += home_goals
            team_standings[home_id]['home_goals_against'] += away_goals
            team_standings[home_id]['goals_diff'] = team_standings[home_id]['goals_for'] - team_standings[home_id]['goals_against']
            
            # Update away team stats
            team_standings[away_id]['played'] += 1
            team_standings[away_id]['away_played'] += 1
            team_standings[away_id]['goals_for'] += away_goals
            team_standings[away_id]['goals_against'] += home_goals
            team_standings[away_id]['away_goals_for'] += away_goals
            team_standings[away_id]['away_goals_against'] += home_goals
            team_standings[away_id]['goals_diff'] = team_standings[away_id]['goals_for'] - team_standings[away_id]['goals_against']
            
            # Update points and win/draw/loss stats
            if home_goals > away_goals:
                # Home win
                team_standings[home_id]['points'] += 3
                team_standings[home_id]['wins'] += 1
                team_standings[home_id]['home_wins'] += 1
                team_standings[away_id]['losses'] += 1
                team_standings[away_id]['away_losses'] += 1
                home_result = 'W'
                away_result = 'L'
            elif home_goals == away_goals:
                # Draw
                team_standings[home_id]['points'] += 1
                team_standings[away_id]['points'] += 1
                team_standings[home_id]['draws'] += 1
                team_standings[away_id]['draws'] += 1
                team_standings[home_id]['home_draws'] += 1
                team_standings[away_id]['away_draws'] += 1
                home_result = 'D'
                away_result = 'D'
            else:
                # Away win
                team_standings[away_id]['points'] += 3
                team_standings[away_id]['wins'] += 1
                team_standings[away_id]['away_wins'] += 1
                team_standings[home_id]['losses'] += 1
                team_standings[home_id]['home_losses'] += 1
                home_result = 'L'
                away_result = 'W'
            
            # Update form (last 5 matches)
            team_standings[home_id]['matches_history'].append(home_result)
            team_standings[away_id]['matches_history'].append(away_result)
            
            # Keep only last 5 matches for form
            if len(team_standings[home_id]['matches_history']) > 5:
                team_standings[home_id]['matches_history'] = team_standings[home_id]['matches_history'][-5:]
            if len(team_standings[away_id]['matches_history']) > 5:
                team_standings[away_id]['matches_history'] = team_standings[away_id]['matches_history'][-5:]
            
            team_standings[home_id]['form'] = ''.join(team_standings[home_id]['matches_history'])
            team_standings[away_id]['form'] = ''.join(team_standings[away_id]['matches_history'])
        
        # Create dataframe from collected rows
        output_df = pd.DataFrame(output_rows)
        
        # Reorder columns for better organization
        base_cols = [ 'team_id', 'team_name', 
                    'rank']
        stat_cols = [col for col in output_df.columns if col not in base_cols]
        output_df = output_df[base_cols + stat_cols]
        
        return output_df

    def discover_leagues_seasons(self) -> List[Tuple[str, str]]:
        """
        Automatically discover all league-season combinations in extracted_dir
        
        Returns:
            List of (league, season) tuples
        """
        leagues_seasons = []
        extracted_path = Path(self.config['extracted_dir'])
        
        # Iterate through all league directories
        for league_dir in extracted_path.iterdir():
            if league_dir.is_dir():
                league = league_dir.name
                
                # Iterate through all season directories
                for season_dir in league_dir.iterdir():
                    if season_dir.is_dir():
                        season = season_dir.name
                        leagues_seasons.append((league, season))
        
        if self.config['verbose']:
            print(f"Discovered {len(leagues_seasons)} league-season combinations")
        
        return leagues_seasons

    def process_all_data(self) -> None:
        """
        Process all discovered data automatically
        """
        leagues_seasons = self.discover_leagues_seasons()
        
        if self.config['verbose']:
            print(f"Starting processing for {len(leagues_seasons)} league-season combinations")
        
        for league, season in leagues_seasons:
            if self.config['verbose']:
                print(f"\nProcessing {league} - {season}")
            
            # Process each data type
            for data_type, process_func in self.process_functions.items():
                self._process_league_season_data(league, season, data_type, process_func)

    def _process_league_season_data(self, league: str, season: str, 
                                  data_type: str, process_func: callable) -> None:
        """
        Process specific data type for a league-season combination
        """
        # Construct file paths
        input_path = Path(self.config['extracted_dir']) / league / season / self.file_patterns[data_type]
        output_dir = Path(self.config['processed_dir']) / league / season
        output_path = output_dir / f"{data_type}_processed.csv"
        
        # Skip if input doesn't exist
        if not input_path.exists():
            if self.config['verbose']:
                print(f"⚠️ File not found: {input_path}")
            return
        
        try:
            # Create output directory if needed
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data
            df = pd.read_csv(input_path)
            
            # Special handling for data types that need fixtures
            kwargs = {}
            if data_type in ['team_stats', 'player_stats']:
                fixtures_path = Path(self.config['processed_dir']) / league / season / "fixtures_processed.csv"
                if fixtures_path.exists():
                    kwargs['fixtures_df'] = pd.read_csv(fixtures_path)
            
            # Process data
            processed_df = process_func(df, **kwargs)
            
            # Save results
            processed_df.to_csv(output_path, index=False)
            
            if self.config['verbose']:
                print(f"✅ Processed {data_type} -> {output_path}")
                
        except Exception as e:
            if self.config['verbose']:
                print(f"❌ Error processing {input_path}: {str(e)}")

    def _get_fixtures_for_stats(self, season_path: str) -> Optional[pd.DataFrame]:
        """
        Helper method to get fixtures data for other processors that need it.
        Caches the result to avoid repeated file reads.
        """
        if season_path in self._fixtures_cache:
            return self._fixtures_cache[season_path]
        
        fixtures_path = os.path.join(season_path, self.file_names['fixtures'])
        if not os.path.exists(fixtures_path):
            if self.config['verbose']:
                print(f"⚠️ Could not find fixtures data at {fixtures_path}")
            return None
        
        try:
            fixtures = pd.read_csv(fixtures_path)
            self._fixtures_cache[season_path] = fixtures
            return fixtures
        except Exception as e:
            if self.config['verbose']:
                print(f"⚠️ Error loading fixtures data: {str(e)}")
            return None
    
    
    def merge_data_for_season(self, league: str, season: str) -> pd.DataFrame:
        """
        Merge all processed data for a specific league and season.
        
        Args:
            league: League name (e.g., "La Liga")
            season: Season year (e.g., "2022")
            
        Returns:
            Merged DataFrame for the season
        """
        if self.config['verbose']:
            print(f"\n🔗 Merging data for {league} {season}...")
        
        # Path to processed data directory
        processed_dir = os.path.join(self.config['processed_dir'], league, season)
        
        # File paths
        file_paths = {
            'fixtures': os.path.join(processed_dir, f"fixtures{self.processed_suffix}"),
            'lineups': os.path.join(processed_dir, f"lineups{self.processed_suffix}"),
           # 'injuries': os.path.join(processed_dir, f"injuries{self.processed_suffix}"),
            'team_stats': os.path.join(processed_dir, f"team_stats{self.processed_suffix}"),
           # 'player_stats': os.path.join(processed_dir, f"player_stats{self.processed_suffix}"),
            'standings': os.path.join(processed_dir, f"standings{self.processed_suffix}"),
        }
        
        # Check all required files exist
        missing_files = [ftype for ftype, path in file_paths.items() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing processed files for {league}/{season}: {missing_files}")
        
        # Load all datasets
        data = {}
        for ftype, path in file_paths.items():
            try:
                data[ftype] = pd.read_csv(path)
                if self.config['verbose']:
                    print(f"📁 Loaded {ftype} data ({len(data[ftype])} records)")
            except Exception as e:
                raise ValueError(f"Error loading {ftype} data: {str(e)}")
        
        # Merge data using the optimized method
        merged = self._merge_datasets_optimized(**data)
        merged.sort_values(by='date', inplace=True)
        
        # Add league and season identifiers
        #merged['league'] = league
        #merged['season'] = season
        
        # Save merged data for this season
        season_output_path = os.path.join(
            self.config['merged_dir'],
            league,
            season,
            "merged_data.csv"
        )
        os.makedirs(os.path.dirname(season_output_path), exist_ok=True)
        merged.to_csv(season_output_path, index=False)
        
        if self.config['verbose']:
            print(f"💾 Saved merged data for {league} {season} to {season_output_path}")
        
        return merged
    
    def _merge_datasets_optimized(self, fixtures, lineups, team_stats, standings):
        """
        Optimized merge process for standings structured like lineups
        """
        if self.config['verbose']:
            print("\n🔄 Starting optimized merge process...")
        
        # 1. Verify and prepare base data
        REQUIRED_COLS = ['fixture_id', 'home_team_id', 'away_team_id']
        missing = [col for col in REQUIRED_COLS if col not in fixtures.columns]
        if missing:
            raise ValueError(f"Fixtures missing required columns: {missing}")
        
        # 2. Process lineups and standings (both have team_id per fixture)
        team_ref = fixtures[REQUIRED_COLS].copy()
        
        # Merge team references into both lineups and standings
        lineups = lineups.merge(team_ref, on='fixture_id', how='left')
        standings = standings.merge(team_ref, on='fixture_id', how='left')
   
        
        # 3. Prepare team data (combine stats, injuries, lineups)
        team_data = (
            team_stats
           # .merge(injuries, on=['fixture_id', 'team_id'], how='left')
            #.merge(player_stats.select_dtypes(include=['number']).groupby(['fixture_id', 'team_id']).mean(), 
                #on=['fixture_id', 'team_id'], how='left')
           .merge(lineups, on=['fixture_id', 'team_id'], how='left')
        )

        
        # 4. Split into home and away data
        home_data = (
            team_data[team_data['team_id'] == team_data['home_team_id']]
            .drop(columns=['home_team_id', 'away_team_id'])
            .rename(columns=lambda x: f'home_{x}' if x != 'fixture_id' else x)
        )

        
        away_data = (
            team_data[team_data['team_id'] == team_data['away_team_id']]
            .drop(columns=['home_team_id', 'away_team_id'])
            .rename(columns=lambda x: f'away_{x}' if x != 'fixture_id' else x)
        )

        
        # 5. Merge home and away data
        merged = (
            fixtures
            .merge(home_data, on='fixture_id', how='left')
            .merge(away_data, on='fixture_id', how='left')
        )
   
        
        # 6. Process standings (similar to lineups)
        home_standings = (
            standings[standings['team_id'] == standings['home_team_id']]
            .drop(columns=['home_team_id', 'away_team_id'])
            .rename(columns=lambda x: f'home_{x}' if x != 'fixture_id' else x)
        )

        
        away_standings = (
            standings[standings['team_id'] == standings['away_team_id']]
            .drop(columns=['home_team_id', 'away_team_id'])
            .rename(columns=lambda x: f'away_{x}' if x != 'fixture_id' else x)
        )

        
        # 7. Merge standings
        merged = (
            merged
            .merge(home_standings, on='fixture_id', how='left')
            .merge(away_standings, on='fixture_id', how='left')
        )

        
        # 8. Cleanup
         # Add time-based features
        merged['date'] = pd.to_datetime(merged['date'])
        merged['days_since_last_home'] = merged.groupby('home_team_id')['date'].diff().dt.days
        merged['days_since_last_away'] = merged.groupby('away_team_id')['date'].diff().dt.days
        # Fill NaN values (for first game of each team) with 0 or some default value
        merged['days_since_last_home'] = merged['days_since_last_home'].fillna(0)
        merged['days_since_last_away'] = merged['days_since_last_away'].fillna(0)
        num_cols = merged.select_dtypes(include=np.number).columns
        merged[num_cols] = merged[num_cols].fillna(0)
        
        def clean_column_name(name):
            """Remove _x/_y suffixes while preserving other underscores"""
            if name.endswith('_x') or name.endswith('_y'):
                return name[:-2]  # Remove last 2 characters
            return name
        
       # merged.columns = [clean_column_name(col) for col in merged.columns]
        #merged['home_team_id'] = merged['home_team_id_x']
        #merged['away_team_id'] = merged['away_team_id_x']
        cols_to_drop = [
            'home_standings_team_id', 'away_standings_team_id', 'league',
            'home_home_team_id', 'away_away_team_id', 'away_team_id_x', 'home_team_id_x',
            'date_away', 'home_team_id_y', 'away_team_id_y', 'home_date', 'away_date',
            'home_team_name', 'away_team_name',
        ]
        merged.drop(columns=[col for col in cols_to_drop if col in merged.columns], inplace=True)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        if self.config['verbose']:
            print("✅ Merge completed successfully!")
            print(f"📊 Dataset contains {len(merged)} records and {len(merged.columns)} features.")
            print(f"Memory usage: {merged.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
        
        
        return merged
        
    def _merge_datasets_optimized_2(self, fixtures, lineups, injuries, team_stats, player_stats, standings):
        """
        Optimized merge process with robust type handling and error checking.
        Returns merged dataset with home/away features and cleaned columns.
        """
        if self.config['verbose']:
            print("\n🔄 Starting optimized merge process...")
        
        # 1. Verify and prepare base data with type conversion
        REQUIRED_COLS = ['fixture_id', 'home_team_id', 'away_team_id']
        missing = [col for col in REQUIRED_COLS if col not in fixtures.columns]
        if missing:
            raise ValueError(f"Fixtures missing required columns: {missing}")
        

        
        # 2. Process lineups and standings with team references
        team_ref = fixtures[REQUIRED_COLS].copy()
        
        # Merge team references with validation
        def safe_merge(left, right, on_cols):
            try:
                return left.merge(right, on=on_cols, how='left', validate='many_to_one')
            except pd.errors.MergeError as e:
                print(f"Merge conflict on columns: {on_cols}")
                print(f"Left types: {left[on_cols].dtypes}")
                print(f"Right types: {right[on_cols].dtypes}")
                raise
        
        lineups = safe_merge(lineups, team_ref, ['fixture_id'])
        standings = safe_merge(standings, team_ref, ['fixture_id'])
        
        # 3. Prepare team data with numeric aggregation
        numeric_player_stats = player_stats.select_dtypes(include=['number'])
        player_agg = numeric_player_stats.groupby(['fixture_id', 'team_id']).mean()
        
        team_data = (
            team_stats
            .pipe(safe_merge, injuries, ['fixture_id', 'team_id'])
            .pipe(safe_merge, player_agg, ['fixture_id', 'team_id'])
            .pipe(safe_merge, lineups, ['fixture_id', 'team_id'])
        )
        
        # 4. Split into home and away data with prefixing
        def split_and_prefix(df, team_type):
            prefix = f'{team_type}_'
            return (
                df[df['team_id'] == df[f'{team_type}_team_id']]
                .drop(columns=['home_team_id', 'away_team_id'])
                .rename(columns=lambda x: prefix + x if x != 'fixture_id' else x)
            )
        
        home_data = split_and_prefix(team_data, 'home')
        away_data = split_and_prefix(team_data, 'away')
        
        # 5. Core merge with fixtures
        merged = (
            fixtures
            .pipe(safe_merge, home_data, ['fixture_id'])
            .pipe(safe_merge, away_data, ['fixture_id'])
        )
        
        # 6. Merge standings data
        home_standings = split_and_prefix(standings, 'home')
        away_standings = split_and_prefix(standings, 'away')
        
        merged = (
            merged
            .pipe(safe_merge, home_standings, ['fixture_id'])
            .pipe(safe_merge, away_standings, ['fixture_id'])
        )
        
        # 7. Add temporal features with proper null handling
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        for team_type in ['home', 'away']:
            col = f'days_since_last_{team_type}'
            merged[col] = (
                merged.groupby(f'{team_type}_team_id')['date']
                .diff()
                .dt.total_seconds()
                .div(86400)  # Convert to days
                .fillna(0)  # Fill first occurrence for each team
            )
        
        # 8. Cleanup and final processing
        cols_to_drop = [
            'home_standings_team_id', 'away_standings_team_id', 'league',
            'home_home_team_id', 'away_away_team_id', 
            'home_team_name', 'away_team_name',
        ] + [col for col in merged.columns if '_x' in col or '_y' in col]
        
        merged = (
            merged
            .drop(columns=[col for col in cols_to_drop if col in merged.columns])
            .loc[:, ~merged.columns.duplicated()]
        )
        
        # Fill numeric NA values
        num_cols = merged.select_dtypes(include=np.number).columns
        merged[num_cols] = merged[num_cols].fillna(0)
        
        if self.config['verbose']:
            print("✅ Merge completed successfully!")
            print(f"📊 Final dataset contains {len(merged)} records and {len(merged.columns)} features.")
            print(f"Memory usage: {merged.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
        
        return merged
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        Merge all processed data across all leagues and seasons.
        
        Returns:
            Final merged DataFrame
        """
        if self.config['verbose']:
            print("\n🔗 Starting final merge of all processed data...")
        
        # Find all league-season combinations
        processed_dir = Path(self.config['processed_dir'])
        
        # Get all leagues (if not specified in config)
        leagues = []
        if 'leagues' in self.config and self.config['leagues'] is not None:
            leagues = self.config['leagues']
        else:
            leagues = [
                d.name for d in processed_dir.iterdir() 
                if d.is_dir()
            ]
        
        all_merged = []
        
        for league in leagues:
            league_path = processed_dir / league
            
            # Get all seasons (if not specified in config)
            seasons = []
            if 'seasons' in self.config and self.config['seasons'] is not None:
                seasons = self.config['seasons']
            else:
                seasons = [
                    d.name for d in league_path.iterdir()
                    if d.is_dir()
                ]
            
            for season in seasons:
                try:
                    merged = self.merge_data_for_season(league, season)
                    all_merged.append(merged)
                    if self.config['verbose']:
                        print(f"✅ Successfully merged {league} {season}")
                except Exception as e:
                    if self.config['verbose']:
                        print(f"❌ Error merging {league} {season}: {str(e)}")
                    continue
        
        if not all_merged:
            raise ValueError("No data was successfully merged")
        
        # Combine all merged data
        final_data = pd.concat(all_merged, axis=0, ignore_index=True)
        
        # Save final output
        final_output_path = Path(self.config['merged_dir']) / 'final_merged_dataset.csv'
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_data.to_csv(final_output_path, index=False)
        
        if self.config['verbose']:
            print(f"\n💾 Final merged dataset saved to {final_output_path}")
            print(f"📊 Final Dataset contains {len(final_data)} records and {len(final_data.columns)} features.")
            print(f"Memory usage: {merged.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
           
        
        return final_data
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run complete automated pipeline:
        1. Discover all data
        2. Process all data
        3. Merge all data
        """
        # Process all discovered data
        self.process_all_data()
        
        # Merge all processed data
        final_data = self.merge_all_data()
        
        return final_data


if __name__ == "__main__":
    # Minimal configuration - just point to your directories
    config = {
        'extracted_dir': 'data/extracted',  # Root of your extracted data
        'processed_dir': 'data/processed',
        'merged_dir': 'data/merged',
        'verbose': True
    }
    
    # Create and run pipeline
    pipeline = FootballDataPipeline(config)
    final_dataset = pipeline.run_pipeline()
    
    print(f"\nPipeline completed!")