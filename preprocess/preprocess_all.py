from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings
import pickle
from tqdm import tqdm  # For progress tracking (optional)
import logging
warnings.filterwarnings("ignore")
from src.utils import LEAGUES  # Assuming LEAGUES is defined in utils.py

class FootballDataPipeline:
    """
    Enhanced pipeline that:
    1. First merges all seasons for each league into complete DataFrames
    2. Then performs preprocessing on the complete league data
    3. Finally combines all leagues into one final dataset
    
    Now with incremental processing support for new seasons
    """
    
    def __init__(self, log_dir="logs", config: Optional[Dict] = None):
        # Default configuration
        self.config = {
            'raw_dir': 'data/extracted',
            'merged_dir': 'data/merged',
            'final_output': 'data/final_processed_incement.csv',
            'processed_output': 'data/processed',  # New directory for processed league data
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
            'incremental_mode': False,  # New config for incremental processing
            'current_season': None,  # New config to specify current season for incremental processing
            'h2h_store': 'data/h2h_store.pkl',  # File to store all H2H data

        }
        
        if config:
            self.config.update(config)
        
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load existing H2H data if available
        self.h2h_data = self._load_h2h_data()
        
        Path(self.config['merged_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['processed_output']).mkdir(parents=True, exist_ok=True)  # Create processed dir

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
        log_file = os.path.join(self.log_dir, f"process_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    

    def _load_h2h_data(self) -> Dict:
        """Load stored H2H data from disk"""
        h2h_path = Path(self.config['h2h_store'])
        if h2h_path.exists():
            try:
                with open(h2h_path, 'rb') as f:
                    data = pickle.load(f)
                    self.logger.info(f"Loaded H2H data from {h2h_path} with {len(data)} team pairs")
                    return data
            except Exception as e:
                self.logger.error(f"Failed to load H2H data from {h2h_path}: {str(e)}")
                return {}
        else:
            self.logger.info("No existing H2H data found, starting fresh")
            return {}

    def _save_h2h_data(self):
        """Save current H2H data to disk"""
        try:
            with open(self.config['h2h_store'], 'wb') as f:
                pickle.dump(self.h2h_data, f)
            self.logger.info(f"Saved H2H data with {len(self.h2h_data)} team pairs to {self.config['h2h_store']}")
        except Exception as e:
            self.logger.error(f"Failed to save H2H data: {str(e)}")

    def _update_h2h_data(self, new_matches: pd.DataFrame):
        """
        Update H2H data with new matches while preserving history
        Args:
            new_matches: DataFrame containing new matches to add to H2H stats
        """
        for _, match in new_matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']
            key = frozenset({home_id, away_id})  # Use frozenset for bidirectional lookup
            
            if key not in self.h2h_data:
                self.h2h_data[key] = []
            
            # Store minimal match data needed for H2H calculations
            self.h2h_data[key].append({
                'date': match_date,
                'home_id': home_id,
                'home_goals': match['home_goals'],
                'away_id': away_id,
                'away_goals': match['away_goals'],
                'league_id': match['league_id']
            })    
    
    def _discover_data_structure(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover available countries, leagues and seasons"""
        structure = {}
        raw_path = Path(self.config['raw_dir'])
        
        for country_dir in raw_path.iterdir():
            if country_dir.is_dir():
                country_name = country_dir.name
                structure[country_name] = {}
                
                for league_dir in country_dir.iterdir():
                    if league_dir.is_dir():
                        league_name = league_dir.name
                        seasons = []
                        
                        for season_dir in league_dir.iterdir():
                            if season_dir.is_dir():
                                seasons.append(season_dir.name)
                        
                        if seasons:
                            structure[country_name][league_name] = sorted(seasons)
        
        return structure
    
    def _get_processed_seasons(self, country: str, league: str) -> List[str]:
        """Get list of already processed seasons for a league"""
        processed_path = Path(self.config['processed_output']) / country / league
        if not processed_path.exists():
            return []
        
        return [f.name.replace('.csv', '') for f in processed_path.glob('*.csv') 
                if f.is_file() and f.name != 'all_seasons_merged.csv']

    def _load_processed_league_data(self, country: str, league: str) -> Optional[pd.DataFrame]:
        """Load already processed league data if it exists"""
        processed_path = Path(self.config['processed_output']) / country / league / 'all_seasons_merged.csv'
        if processed_path.exists():
            try:
                return pd.read_csv(processed_path)
            except Exception as e:
                self.logger.error(f"Error loading processed data for {country}/{league}: {str(e)}")
                return None
        return None

    def _save_processed_league_data(self, country: str, league: str, data: pd.DataFrame):
        """Save processed league data"""
        processed_dir = Path(self.config['processed_output']) / country / league
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete merged data
        data.to_csv(processed_dir / 'all_seasons_merged.csv', index=False)
        
        # Also save individual season data for incremental processing
        if 'season' in data.columns:
            for season in data['season'].unique():
                season_data = data[data['season'] == season]
                season_data.to_csv(processed_dir / f'{season}.csv', index=False)

    def _process_single_season(self, country: str, league: str, season: str) -> Optional[pd.DataFrame]:
        """Process and merge data for a single country/league/season
        
        Args:
            country: Country name (directory name)
            league: League name (subdirectory name)
            season: Season name (subdirectory name)
            
        Returns:
            Merged DataFrame or None if processing fails
        """
        # In incremental mode, check if this season is already processed
        if self.config['incremental_mode'] and season in self._get_processed_seasons(country, league):
            self.logger.info(f"Skipping already processed season: {country}/{league}/{season}")
            return None
            
        season_path = Path(self.config['raw_dir']) / country / league / season
        self.logger.info(f"Processing {country}/{league}/{season}")
        
        try:
            # 1. Validate and load files
            fixtures_path = season_path / self.config['data_types']['fixtures']
            team_stats_path = season_path / self.config['data_types']['team_stats']
            
            if not fixtures_path.exists():
                self.logger.warning(f"Missing fixtures file: {fixtures_path}")
                return None
            if not team_stats_path.exists():
                self.logger.warning(f"Missing team stats file: {team_stats_path}")
                return None
                
            # Load with error handling
            fixtures = pd.read_csv(fixtures_path)
            team_stats = pd.read_csv(team_stats_path)

            
            # 2. Validate required columns
            required_fixture_cols = self.config['required_cols']['fixtures']
            missing_fixture_cols = [col for col in required_fixture_cols 
                                if col not in fixtures.columns]
            if missing_fixture_cols:
                raise ValueError(f"Missing required columns in fixtures: {missing_fixture_cols}")

            
            # 4. Prepare team references
            team_ref = fixtures[['fixture_id', 'home_team_id', 'away_team_id']].copy()
            
            # Handle optional columns - only for team_stats
            optional_team_stats_columns = ['expected_goals', 'goals_prevented']
            columns_to_drop = [col for col in optional_team_stats_columns 
                            if col in team_stats.columns]
            
            # Safely filter team stats
            if columns_to_drop:
                team_stats = team_stats.drop(columns=columns_to_drop, errors='ignore')
            
            # Handle penalty columns in fixtures (if they exist but you want to remove them)
            penalty_columns = ['penalty_home', 'penalty_away']
            if any(col in fixtures.columns for col in penalty_columns):
                fixtures = fixtures.drop(columns=penalty_columns, errors='ignore')
            
            # 6. Merge team stats with references
            team_data = team_stats.merge(team_ref, on='fixture_id', how='left')
            
            # 7. Split into home/away data with error handling
            try:
                home_mask = team_data['team_id'] == team_data['home_team_id']
                home_data = (
                    team_data[home_mask]
                    .drop(columns=['home_team_id', 'away_team_id', 'team_id'])
                    .rename(columns=lambda x: f'home_{x}' if x != 'fixture_id' else x)
                )
                
                away_mask = team_data['team_id'] == team_data['away_team_id']
                away_data = (
                    team_data[away_mask]
                    .drop(columns=['home_team_id', 'away_team_id', 'team_id'])
                    .rename(columns=lambda x: f'away_{x}' if x != 'fixture_id' else x)
                )
            except KeyError as e:
                raise ValueError(f"Missing team reference columns: {str(e)}")
            
            # 8. Final merge with validation
            if len(home_data) == 0 or len(away_data) == 0:
                raise ValueError("No home or away data found after splitting")
                
            merged = (
                fixtures
                .merge(home_data, on='fixture_id', how='left')
                .merge(away_data, on='fixture_id', how='left')
            )
            
            # Add season column if not present
            if 'season' not in merged.columns:
                merged['season'] = season
            
            # Validate merge succeeded
            if len(merged) == 0:
                raise ValueError("Final merge resulted in empty DataFrame")
                
            return merged
            
        except Exception as e:
            self.logger.error(f"Error processing {country}/{league}/{season}: {str(e)}")
            if self.config.get('verbose', False):
                import traceback
                self.logger.debug(traceback.format_exc())
            return None

    def _add_new_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced derived features with comprehensive shot analysis and performance metrics.
        Organized into logical sections with better documentation and error handling.
        
        Args:
            df: Input DataFrame containing match statistics
            
        Returns:
            DataFrame with additional calculated metrics
        """
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # 1. String columns handling and cleanup
        string_cols = ['league_name', 'league_flag', 'league_logo', 'season',
                    'home_team', 'home_team_flag', 'away_team', 'away_team_flag',
                    'venue_name', 'venue_city', 'referee', 'status',
                    'home_form', 'away_form', 'round', 'outcome', 'h2h_streak']
        
        # Store string columns before processing
        string_backups = {col: df[col].copy() for col in string_cols if col in df.columns}
        
        
        # Columns to drop
        cols_to_drop = ['home_team_name', 'away_team_name', 'home_winner', 'away_winner',
                    'home_team_logo', 'away_team_logo']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')


        # --- Feature Engineering ---
        self.original_metrics = [
            'shots_on_goal', 'shots_off_goal', 'total_shots', 'blocked_shots', 
            'shots_insidebox', 'shots_outsidebox', 'fouls', 'corner_kicks', 
            'offsides', 'ball_possession', 'yellow_cards', 'red_cards', 
            'goalkeeper_saves', 'total_passes', 'passes_accurate', 'passes',
            'expected_goals', 'goals_prevented'
        ]        
        
        # Finalize new metrics sets
        self.original_metrics = {
            f"{prefix}_{metric}" 
            for metric in self.original_metrics 
            for prefix in ['home', 'away']
        }       
        
        self.new_metrics = set()
        self.diff_metrics = set()
        
        # Helper function to check if required columns exist
        def has_required_cols(prefix, metrics):
            return all(f"{prefix}_{m}" in df.columns for m in metrics)
        
        
        # ===================================
        # 1. ADVANCED METRICS (xG and gp)
        # ===================================
        def calculate_expected_goals(row, team_type):
            """Calculate expected goals based on shot data"""
            prefix = f"{team_type}_"
            shots_on_goal = row.get(f"{prefix}shots_on_goal", 0)
            shots_insidebox = row.get(f"{prefix}shots_insidebox", 0)
            shots_outsidebox = row.get(f"{prefix}shots_outsidebox", 0)
            possession = row.get(f"{prefix}ball_possession", 50)
            
            # Weightings (calibrated for soccer/football)
            inside_box_weight = 0.15
            outside_box_weight = 0.05
            on_target_weight = 0.3
            
            # Basic xG calculation
            xG = (shots_insidebox * inside_box_weight + 
                shots_outsidebox * outside_box_weight) * (1 + on_target_weight if shots_on_goal > 0 else 1)
            
            # Adjust for possession
            xG *= (possession / 50)  # Normalize around average possession
            
            return round(xG, 2)

        def calculate_goals_prevented(row, team_type):
            """Calculate goals prevented by goalkeeper"""
            prefix = f"{team_type}_"
            opp_prefix = "away_" if team_type == "home" else "home_"
            
            goals_conceded = row.get(f"{opp_prefix}goals", 0)
            xG_against = row.get(f"{opp_prefix}expected_goals", 0)
            saves = row.get(f"{prefix}goalkeeper_saves", 0)
            
            goals_prevented = xG_against - goals_conceded
            
            # Adjust for number of saves
            if pd.notna(saves):
                goals_prevented *= (1 + saves / 10)
                
            return round(goals_prevented, 2) if not pd.isna(goals_prevented) else 0

        # Calculate advanced metrics if shot data exists
        if has_required_cols('home', ['shots_insidebox', 'shots_outsidebox', 'shots_on_goal', 'ball_possession']):
            # Calculate expected goals
            df['home_expected_goals'] = df.apply(lambda x: calculate_expected_goals(x, 'home'), axis=1)
            df['away_expected_goals'] = df.apply(lambda x: calculate_expected_goals(x, 'away'), axis=1)
            
            # Calculate goals prevented if goalkeeper data exists
            if 'home_goalkeeper_saves' in df.columns:
                df['home_goals_prevented'] = df.apply(lambda x: calculate_goals_prevented(x, 'home'), axis=1)
                df['away_goals_prevented'] = df.apply(lambda x: calculate_goals_prevented(x, 'away'), axis=1)
            
            self.new_metrics.update(['expected_goals', 'goals_prevented'])        
        
        # ========================
        # 2. SHOT ANALYSIS METRICS
        # ========================
        shot_metrics = ['shots_on_goal', 'total_shots', 'shots_insidebox', 'shots_outsidebox', 'goals']
        
        if (has_required_cols('home', shot_metrics) and 
            has_required_cols('away', shot_metrics)):
            
            for prefix in ['home', 'away']:
                opp_prefix = 'away' if prefix == 'home' else 'home'
                
                # Shot accuracy
                df[f'{prefix}_shot_accuracy'] = (
                    df[f'{prefix}_shots_on_goal'] / 
                    df[f'{prefix}_total_shots'].replace(0, 1))
                
                # Shot quality
                df[f'{prefix}_shot_quality'] = (
                    (df[f'{prefix}_shots_insidebox'] * 1.5 + 
                    df[f'{prefix}_shots_outsidebox'] * 0.5) / 
                    df[f'{prefix}_total_shots'].replace(0, 1))
                
                # Box ratio
                df[f'{prefix}_box_ratio'] = (
                    df[f'{prefix}_shots_insidebox'] / 
                    df[f'{prefix}_total_shots'].replace(0, 1))
                
                # Shot efficiency
                df[f'{prefix}_shot_efficiency'] = (
                    df[f'{prefix}_goals'] / 
                    df[f'{prefix}_total_shots'].replace(0, 1))
                
                # Long-range threat
                df[f'{prefix}_longrange_threat'] = (
                    df[f'{prefix}_shots_outsidebox'] / 
                    df[f'{prefix}_total_shots'].replace(0, 1))

            
            self.new_metrics.update(['shot_accuracy', 'shot_quality', 'box_ratio', 'shot_efficiency', 'longrange_threat'])

        # ========================
        # 3. OFFENSIVE METRICS
        # ========================
        offensive_metrics = ['offsides', 'passes', 'corner_kicks', 'goals']
        
        if (has_required_cols('home', offensive_metrics) and 
            has_required_cols('away', offensive_metrics)):
            
            for prefix in ['home', 'away']:
                opp_prefix = 'away' if prefix == 'home' else 'home'
                
                # Offside per attempt
                df[f'{prefix}_offside_per_attempt'] = (
                    df[f'{opp_prefix}_offsides'] / 
                    df[f'{prefix}_passes'].replace(0, 1))
                
                # Corner efficiency
                df[f'{prefix}_corner_efficiency'] = (
                    df[f'{prefix}_goals'] / 
                    df[f'{prefix}_corner_kicks'].replace(0, 1))
            
            self.new_metrics.update(['offside_per_attempt', 'corner_efficiency'])

        # ========================
        # 4. DEFENSIVE METRICS
        # ========================
        defensive_metrics = ['goals', 'shots_on_goal', 'fouls', 'blocked_shots']
        
        if (has_required_cols('home', defensive_metrics) and 
            has_required_cols('away', defensive_metrics)):
            
            for prefix in ['home', 'away']:
                opp_prefix = 'away' if prefix == 'home' else 'home'
                
                # Defensive efficiency
                df[f'{prefix}_defensive_efficiency'] = (
                    1 - (df[f'{opp_prefix}_goals'] / 
                        df[f'{prefix}_shots_on_goal'].replace(0, 1)))
                
                # Defensive pressure
                df[f'{prefix}_defensive_pressure'] = (
                    df[f'{prefix}_fouls'] + 
                    df[f'{prefix}_blocked_shots'])
                
                # Home clearance efficiency
                df[f'{prefix}_clearance_efficiency'] = (
                    df[f'{prefix}_blocked_shots'] / 
                    (df[f'{prefix}_shots_on_goal'] + 
                     df[f'{prefix}_shots_off_goal']).replace(0, 1))
               
            
            self.new_metrics.update(['defensive_efficiency', 'defensive_pressure', 'clearance_efficiency'])

        # ========================
        # 5. POSSESSION METRICS
        # ========================
        possession_metrics = ['passes_accurate', 'total_passes', 'fouls', 'ball_possession']
        
        if (has_required_cols('home', possession_metrics) and 
            has_required_cols('away', possession_metrics)):
            
            # Possession difference
            df['possession_difference'] = (
                df['home_ball_possession'] - 
                df['away_ball_possession'])
            
            for prefix in ['home', 'away']:
                # Pass accuracy
                df[f'{prefix}_pass_accuracy'] = (
                    df[f'{prefix}_passes_accurate'] / 
                    df[f'{prefix}_total_passes'].replace(0, 1))
                
                # Press resistance
                df[f'{prefix}_press_resistance'] = (
                    df[f'{prefix}_total_passes'] / 
                    (df[f'{prefix}_fouls'] + 1))
                
                # Possession efficiency
                df[f'{prefix}_possession_efficiency'] = (
                    df[f'{prefix}_passes'] / 
                    df[f'{prefix}_ball_possession'].replace(0, 1))
            
            self.new_metrics.update(['pass_accuracy', 'press_resistance', 'possession_efficiency'])
            self.diff_metrics.add('possession_difference')

        # ========================
        # 6. xG METRICS
        # ========================
        if all(col in df.columns for col in ['home_expected_goals', 'away_expected_goals']):
            # xG difference
            df['xg_difference'] = (
                df['home_expected_goals'] - 
                df['away_expected_goals'])
            
            # Total xG
            df['xg_total'] = (
                df['home_expected_goals'] + 
                df['away_expected_goals'])
            
            for prefix in ['home', 'away']:
                opp_prefix = 'away' if prefix == 'home' else 'home'
                
                # xG performance
                df[f'{prefix}_xg_performance'] = (
                    df[f'{prefix}_goals'] - 
                    df[f'{prefix}_expected_goals'])
                
                # xG efficiency
                df[f'{prefix}_xg_efficiency'] = (
                    df[f'{prefix}_goals'] / 
                    df[f'{prefix}_expected_goals'].replace(0, 1))
            
            self.new_metrics.update(['xg_performance', 'xg_efficiency'])
            self.diff_metrics.update(['xg_difference', 'xg_total'])

        # ========================
        # 7. GOALKEEPER METRICS
        # ========================
        if all(col in df.columns for col in ['home_goalkeeper_saves', 'away_shots_on_goal',
                                        'away_goalkeeper_saves', 'home_shots_on_goal']):
            df['home_save_percentage'] = (
                df['home_goalkeeper_saves'] / 
                df['away_shots_on_goal'].replace(0, 1))
            
            df['away_save_percentage'] = (
                df['away_goalkeeper_saves'] / 
                df['home_shots_on_goal'].replace(0, 1))
            
            self.new_metrics.add('save_percentage')

        # ========================
        # 8. MATCH CONTEXT METRICS
        # ========================
        if all(col in df.columns for col in ['home_yellow_cards', 'home_red_cards',
                                        'away_yellow_cards', 'away_red_cards']):
            for prefix in ['home', 'away']:
                df[f'{prefix}_total_cards'] = (
                    df[f'{prefix}_yellow_cards'] + 
                    df[f'{prefix}_red_cards'] * 2)
            
            df['discipline_difference'] = (
                df['home_total_cards'] - 
                df['away_total_cards'])
            
            # Home pressing intensity
            df['home_pressing_intensity'] = df['away_total_cards'] / df['away_ball_possession'].replace(0, 1)
            df['away_pressing_intensity'] = df['home_total_cards'] / df['home_ball_possession'].replace(0, 1)

            
            self.new_metrics.update(['total_cards', 'pressing_intensity'])
            self.diff_metrics.add('discipline_difference')

        # ========================
        # 9. MATCH MOMENTUM METRICS
        # ========================
        if all(col in df.columns for col in ['halftime_home', 'halftime_away', 
                                            'home_goals', 'away_goals']):
            
            # Comeback wins
            df['home_comeback_win'] = (
                (df['halftime_home'] < df['halftime_away']) & 
                (df['home_goals'] > df['away_goals'])
            ).astype(int)
            
            df['away_comeback_win'] = (
                (df['halftime_away'] < df['halftime_home']) & 
                (df['away_goals'] > df['home_goals'])
            ).astype(int)
            
            # Lost leads
            df['home_lost_lead'] = (
                (df['halftime_home'] > df['halftime_away']) & 
                (df['home_goals'] < df['away_goals'])
            ).astype(int)
            
            df['away_lost_lead'] = (
                (df['halftime_away'] > df['halftime_home']) & 
                (df['away_goals'] < df['home_goals'])
            ).astype(int)
            
            # Halftime advantage
            df['ht_lead_difference'] = df['halftime_home'] - df['halftime_away']
            df['ft_ht_swing'] = (df['home_goals'] - df['away_goals']) - df['ht_lead_difference']
            
            self.new_metrics.update(['comeback_win', 'lost_lead'])
            self.diff_metrics.update(['ht_lead_difference', 'ft_ht_swing'])        
        
        
        # Rest advantage
        if all(col in df.columns for col in ['home_days_rest', 'away_days_rest']):
            df['rest_difference'] = (
                df['home_days_rest'] - 
                df['away_days_rest'])
            self.diff_metrics.add('rest_difference')

        # Clean sheets
        df['home_clean_sheet'] = (df['away_goals'] == 0).astype(int)
        df['away_clean_sheet'] = (df['home_goals'] == 0).astype(int)
        self.new_metrics.add('clean_sheet')



        self.new_metrics = {
            f"{prefix}_{metric}" 
            for metric in self.new_metrics 
            if metric not in self.diff_metrics 
            for prefix in ['home', 'away']
        } | self.diff_metrics  # Include differential metrics

        self.combined_metrics = self.original_metrics.union(self.new_metrics)

        if self.config.get('drop_original_metrics', False):
            # Drop original features if configured
            cols_to_drop = [col for col in self.original_metrics if col in df.columns]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            self.logger.info(f"Dropped {len(cols_to_drop)} original metrics")

        # Restore string columns
        for col, data in string_backups.items():
            if col in df.columns:
                df[col] = data
            else:
                df[col] = data
                self.logger.warning(f"Restored missing column: {col}")

        return df
    
    def _calculate_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling averages and optionally drop original features
        
        Args:
            df: Input DataFrame with match data
            
        Returns:
            DataFrame with rolling averages added and original features dropped if configured
        """
        if 'date' not in df.columns:
            return df
        
        # Sort and prepare
        df = df.sort_values(['season', 'date'])
        
        # Calculate rolling features for all available metrics
        for window in self.config['rolling_windows']:
            for feature in self.combined_metrics:
                if feature in df.columns:
                    rolling_col = f"{feature}_rolling_{window}"
                    
                    if feature.startswith('home_'):
                        df[rolling_col] = df.groupby(['season', 'home_team_id'])[feature]\
                                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    elif feature.startswith('away_'):
                        df[rolling_col] = df.groupby(['season', 'away_team_id'])[feature]\
                                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    else:
                        home_vals = df.groupby(['season', 'home_team_id'])[feature]\
                                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                        away_vals = df.groupby(['season', 'away_team_id'])[feature]\
                                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                        df[rolling_col] = (home_vals + away_vals) / 2
        
        # SIMPLE DROP LOGIC - ONLY THIS PART CHANGED
        if self.config.get('drop_non_roll_features', False):
            cols_to_drop = [col for col in self.combined_metrics if col in df.columns]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            self.logger.info(f"Dropped {len(cols_to_drop)} non rolled metrics")


        
        return df




    def _merge_all_seasons(self, country: str, league: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        """Merge seasons with optimized standings/H2H processing"""
        if self.config['incremental_mode']:
            existing_data = self._load_processed_league_data(country, league)
            current_season = self.config['current_season']
            
            if existing_data is not None:
                processed_seasons = set(self._get_processed_seasons(country, league))
                seasons_to_process = [s for s in seasons if s not in processed_seasons]
                
                if not seasons_to_process:
                    self.logger.warning(f"All seasons already processed for {country}/{league}")
                    return existing_data
                
                # Process new seasons
                new_data = pd.DataFrame()
                for season in seasons_to_process:
                    season_data = self._process_single_season(country, league, season)
                    if season_data is not None:
                        new_data = pd.concat([new_data, season_data], ignore_index=True)
                
                if new_data.empty:
                    return existing_data
                
                # Combine old and new data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Preprocess with incremental optimization
                combined_data = self._preprocess_and_feature_engineer(
                    combined_data,
                    is_incremental=True,
                    current_season=current_season
                )
                
                self._save_processed_league_data(country, league, combined_data)
                return combined_data
        
        # Default processing path
        league_data = pd.DataFrame()
        for season in seasons:
            season_data = self._process_single_season(country, league, season)
            if season_data is not None:
                league_data = pd.concat([league_data, season_data], ignore_index=True)
        
        if league_data.empty:
            return None
        
        league_data = self._preprocess_and_feature_engineer(league_data)
        
        if self.config['incremental_mode']:
            self._save_processed_league_data(country, league, league_data)
            
        return league_data




    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive cleaning for football fixture data that:
        - Preserves all original columns
        - Ensures consistent ID types (int64)
        - Handles type conversions correctly
        - Maintains statistical integrity
        """
        if df.empty:
            return df

        # Create a working copy
        df = df.copy()

        # 5. Filter FT matches only if status exists
        if 'status' in df.columns:
            df = df[df['status'].isin(['FT', 'AET', 'PEN'])]
        else:
            self.logger.warning("Status column missing - skipping FT filter")


        # 1. Convert all ID columns to int64 upfront
        id_columns = ['fixture_id', 'league_id', 'home_team_id', 'away_team_id', 'venue_id']
        for col in id_columns:
            if col in df.columns:
                # Use pd.NA for missing values in integer columns
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # 2. Convert date and time columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
        
        # 2. Standardize column names (preserve original as reference)
        original_columns = dict(zip(df.columns, 
            df.columns.str.strip().str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
        ))
        df = df.rename(columns=original_columns)

        # 2. Handle percentage columns with proper error handling
        for side in ['home', 'away']:
            # Ball possession percentage
            poss_col = f"{side}_ball_possession"
            if poss_col in df.columns:
                try:
                    # First ensure we're working with strings
                    df[poss_col] = df[poss_col].astype(str)
                    # Then remove % and convert
                    df[poss_col] = (
                        df[poss_col].str.replace('%', '')
                        .apply(pd.to_numeric, errors='coerce') / 100
                    )
                   
                except Exception as e:
                    self.logger.error(f"Error processing {poss_col}: {e}")
                    df[poss_col] = np.nan  # Set to NaN if conversion fails

            # Pass accuracy percentage
            pass_pct_col = f"{side}_passes"
            if pass_pct_col in df.columns:
                try:
                    df[pass_pct_col] = df[pass_pct_col].astype(str)
                    df[pass_pct_col] = (
                        df[pass_pct_col].str.replace('%', '')
                        .apply(pd.to_numeric, errors='coerce') / 100
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing {pass_pct_col}: {e}")
                    df[pass_pct_col] = np.nan
                
        # Filter rows where either shots column is empty
        df = df.dropna(
            subset=['home_shots_on_goal', 'home_shots_off_goal', 'home_goals', 'away_goals', 'halftime_home', 'halftime_away', 'fulltime_home', 'fulltime_away'],
            how='any'
)
        
        # 4. Convert numeric columns with proper null handling
        numeric_cols = [
            'maintime', 'first_half', 'second_half', 
            'home_goals', 'away_goals', 
            'halftime_home', 'halftime_away',
            'fulltime_home', 'fulltime_away'
        ]
        for col in numeric_cols:
            if col in df.columns:
                # Check if the column exists and has data
                if not df[col].empty:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    except Exception as e:
                        self.logger.warning(f"Error converting column {col}: {e}")
                        # If conversion fails, try a different approach
                        try:
                            df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).astype('Int64')
                        except:
                            df[col] = pd.NA  # Set to NA if all else fails
                else:
                    # If column exists but is empty, initialize with NA
                    df[col] = pd.NA

        # 7. Handle float columns that should stay float
        float_cols = ['extratime']
        for col in float_cols:
            if col in df.columns:
                if not df[col].empty:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = np.nan


        # 7. Handle missing values
        fill_rules = {
            'extratime': 0,
            'home_offsides': 0,
            'away_offsides': 0,
            'home_corners': 0,
            'away_corners': 0,
            'home_fouls': 0,
            'away_fouls': 0,
            'home_yellow_cards': 0,
            'away_yellow_cards': 0,
            'home_red_cards': 0,
            'away_red_cards': 0,
            'home_goalkeeper_saves': 0,
            'away_goalkeeper_saves': 0,
            'referee': 'Unknown',
            'venue_city': 'Unknown',
            'venue_name': 'Unknown'
        }
        
        for col, val in fill_rules.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)



        # 8. Clean text fields
        text_cols = ['home_team', 'away_team', 'league_name', 'round', 'referee']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()

        # 9. Final validation
        required_cols = ['fixture_id', 'date', 'home_team', 'away_team', 'league_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        cols_to_drop = ['home_winner', 'away_winner', 'home_team_name', 'away_team_name', 'home_team_logo', 'away_team_logo']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        # 10. Sort by date and reset index
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def _create_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target column based on match outcome with robust error handling.
        """
        required_cols = ['home_goals', 'away_goals']
        
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns {required_cols} - cannot create target column")
            return df
        
        try:
            # Method 1: Vectorized operation with index alignment
            home_goals = df['home_goals'].values
            away_goals = df['away_goals'].values
            
            # Create outcome array
            outcomes = np.empty(len(df), dtype=object)
            outcomes[home_goals > away_goals] = 'home_win'
            outcomes[home_goals < away_goals] = 'away_win'
            outcomes[home_goals == away_goals] = 'draw'
            
            df['outcome'] = outcomes
            
            # Log distribution
            outcome_counts = df['outcome'].value_counts().to_dict()
            self.logger.info(f"Target column created successfully. Distribution: {outcome_counts}")
            
        except Exception as e:
            self.logger.error(f"Error in vectorized target creation: {str(e)}")
        
        return df
   
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates pure temporal features without any existing streak or form calculations.
        Returns DataFrame with added temporal columns.
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain 'date' column")
            
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 1. Basic date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season_quarter'] = (df['month'] - 1) // 3 + 1  # 1-4 quarters
        
        # Fixed values for first matches
        DEFAULT_DAYS_REST = 7  # Typical weekly schedule
        SEASON_START_MEDIAN_DAYS_REST = 56  # Preseason break
        
        # 2. Days since last match with fixed values for season starters
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team_id'
            days_since_col = f'{team_type}_days_since_last_match'
            
            # Calculate days since last match
            df[days_since_col] = df.groupby(team_col)['date'].diff().dt.days
            
            # Fill first match of each season per team
            season_first_mask = df.groupby([team_col, 'year'])['date'].rank(method='first') == 1
            df.loc[season_first_mask, days_since_col] = SEASON_START_MEDIAN_DAYS_REST
            
            # Fill very first match for new teams (no history at all)
            team_first_mask = df.groupby(team_col)['date'].rank(method='first') == 1
            df.loc[team_first_mask, days_since_col] = SEASON_START_MEDIAN_DAYS_REST
            
            # Fill any remaining NaNs (shouldn't exist but just in case)
            df[days_since_col] = df[days_since_col].fillna(DEFAULT_DAYS_REST)
            
            # Create rest ratio (actual days vs expected days)
            df[f'{team_type}_rest_ratio'] = (
                df[days_since_col] / 
                df.groupby(team_col)[days_since_col].transform('median')
            )
        
            
            # Match density (relative to team's normal schedule)
            df[f'{team_type}_match_density'] = (
                df.groupby(team_col)[days_since_col].transform('mean') / 
                (df[days_since_col] + 0.001))  # Avoid division by zero
        

        
        # 4. Holiday features (example for UK)
        df['is_holiday'] = (
            (df['month'].isin([12, 1])) &  # Christmas/New Year
            (df['day_of_month'].isin([24, 25, 26, 31, 1])) |
            (df['month'] == 3) & (df['day_of_month'].isin([17]))  # St. Patrick's
        ).astype(int)
        
        # 5. Time-based features only (no team performance metrics)
        df['is_night_match'] = ((df['date'].dt.hour >= 18) | (df['date'].dt.hour <= 6)).astype(int)
        df['time_of_day'] = df['date'].dt.hour + df['date'].dt.minute/60
        
        return df

    def _create_standings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes and corrects football fixture data with comprehensive feature engineering.
        Handles all edge cases and produces ML-ready output with accurate standings.
        """
        # Load and prepare data
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['league_id', 'season', 'date']).reset_index(drop=True)
        

        season_start_dates = df.groupby('season')['date'].min().to_dict()

        # Initialize league-aware team tracking
        team_history = {}  # Key: (team_id, league_id, season)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Initializing Team History"):
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            league_id = row['league_id']
            season = row['season']
            
            # Initialize league-specific team records
            for team_id in [home_id, away_id]:
                key = (team_id, league_id, season)
                if key not in team_history:
                    team_history[key] = {
                        'points': 0,
                        'goals_for': 0,
                        'goals_against': 0,
                        'matches_played': 0,
                        'form': []
                    }
        
        # Define features with proper data types
        features = {
            # Numeric features
            'home_rank': 'float64',
            'home_points': 'int64',
            'home_goals_diff': 'int64',
            'home_played': 'int64',
            'home_wins': 'int64',
            'home_draws': 'int64',
            'home_losses': 'int64',
            'home_goals_for': 'int64',
            'home_goals_against': 'int64',
            'home_days_rest': 'int64',
            'home_form_strength': 'float64',
            'away_rank': 'float64',
            'away_points': 'int64',
            'away_goals_diff': 'int64',
            'away_played': 'int64',
            'away_wins': 'int64',
            'away_draws': 'int64',
            'away_losses': 'int64',
            'away_goals_for': 'int64',
            'away_goals_against': 'int64',
            'away_days_rest': 'int64',
            'away_form_strength': 'float64',
            # Categorical features

        }

        # Initialize columns carefully (don't overwrite existing data)
        for col, dtype in features.items():
            if col not in df.columns:
                df[col] = pd.Series(dtype=dtype)
            elif df[col].dtype != dtype and dtype != 'object':  # Don't overwrite strings
                try:
                    df[col] = df[col].astype(dtype)
                except:
                    df[col] = pd.Series(dtype=dtype)

        # Process each match
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Standings"):
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            match_date = row['date']
            season = row['season']
            round_num = row['round']
            
            # Initialize team data if new season or new team
            for team_id in [home_id, away_id]:
                if team_id not in team_history or season != team_history[team_id]['current_season']:
                    team_history[team_id] = {
                        'current_season': season,
                        'points': 0,
                        'goals_for': 0,
                        'goals_against': 0,
                        'wins': 0,
                        'draws': 0,
                        'losses': 0,
                        'form': [],  # Stores numeric form (1=win, 0.5=draw, 0=loss)
                        'last_match_date': season_start_dates.get(season, match_date) - timedelta(days=30),
                        'matches_played': 0,
                        'team_name': row['home_team'] if team_id == home_id else row['away_team']
                    }
            
            # Calculate days rest (capped at 30 days)
            home_days_rest = (match_date - team_history[home_id]['last_match_date']).days
            away_days_rest = (match_date - team_history[away_id]['last_match_date']).days
            home_days_rest = min(30, home_days_rest) if home_days_rest > 0 else 30  # Default for first match
            away_days_rest = min(30, away_days_rest) if away_days_rest > 0 else 30  # Default for first match
            
            # Calculate form strength with additive smoothing
            def calculate_enhanced_form_strength(team_id, match_date, team_history, df):
                """Calculate form strength with exponential weighting and goal difference consideration"""
                # Get last 5 matches (excluding current match)
                past_matches = df[(df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)]
                past_matches = past_matches[past_matches['date'] < match_date].sort_values('date', ascending=False).head(5)
                
                if past_matches.empty:
                    return 0.5  # Neutral value
                
                total_weight = 0
                form_score = 0
                
                for i, (_, match) in enumerate(past_matches.iterrows()):
                    # Determine if home or away
                    is_home = match['home_team_id'] == team_id
                    weight = 0.8 ** i  # Exponential decay (most recent match has weight 1)
                    
                    # Get match result
                    if is_home:
                        goals_for = match['home_goals']
                        goals_against = match['away_goals']
                    else:
                        goals_for = match['away_goals']
                        goals_against = match['home_goals']
                        
                    # Calculate result score (0-1)
                    if goals_for > goals_against:  # Win
                        result_score = 1.0
                    elif goals_for == goals_against:  # Draw
                        result_score = 0.5
                    else:  # Loss
                        result_score = 0.0
                        
                    # Add goal difference factor (capped at ±3)
                    gd_factor = min(3, max(-3, goals_for - goals_against)) / 12  # ±0.25 max effect
                    
                    form_score += (result_score + gd_factor) * weight
                    total_weight += weight
                
                # Normalize and apply bounds
                form_strength = form_score / total_weight
                return max(0, min(1, round(form_strength, 4)))
            
            home_form_strength = calculate_enhanced_form_strength(home_id, match_date, team_history, df)
            away_form_strength = calculate_enhanced_form_strength(away_id, match_date, team_history, df)
            
            # Generate form string (last 5 matches)
            def get_form_string(form_history):
                if not form_history:
                    return 'N'
                return ''.join(['W' if x == 1 else 'D' if x == 0.5 else 'L' for x in form_history[-5:]])
            
            home_form_str = get_form_string(team_history[home_id]['form'])
            away_form_str = get_form_string(team_history[away_id]['form'])
            
            # Set PRE-MATCH features
            df.at[idx, 'home_points'] = team_history[home_id]['points']
            df.at[idx, 'home_goals_for'] = team_history[home_id]['goals_for']
            df.at[idx, 'home_goals_against'] = team_history[home_id]['goals_against']
            df.at[idx, 'home_goals_diff'] = team_history[home_id]['goals_for'] - team_history[home_id]['goals_against']
            df.at[idx, 'home_played'] = team_history[home_id]['matches_played']
            df.at[idx, 'home_wins'] = team_history[home_id]['wins']
            df.at[idx, 'home_draws'] = team_history[home_id]['draws']
            df.at[idx, 'home_losses'] = team_history[home_id]['losses']
            df.at[idx, 'home_days_rest'] = home_days_rest
            df.at[idx, 'home_form_strength'] = home_form_strength
            #df.at[idx, 'home_form'] = home_form_str
            
            df.at[idx, 'away_points'] = team_history[away_id]['points']
            df.at[idx, 'away_goals_for'] = team_history[away_id]['goals_for']
            df.at[idx, 'away_goals_against'] = team_history[away_id]['goals_against']
            df.at[idx, 'away_goals_diff'] = team_history[away_id]['goals_for'] - team_history[away_id]['goals_against']
            df.at[idx, 'away_played'] = team_history[away_id]['matches_played']
            df.at[idx, 'away_wins'] = team_history[away_id]['wins']
            df.at[idx, 'away_draws'] = team_history[away_id]['draws']
            df.at[idx, 'away_losses'] = team_history[away_id]['losses']
            df.at[idx, 'away_days_rest'] = away_days_rest
            df.at[idx, 'away_form_strength'] = away_form_strength
            #df.at[idx, 'away_form'] = away_form_str
            
            # Update POST-MATCH stats (using actual match results)
            home_goals = int(row['home_goals'])
            away_goals = int(row['away_goals'])
            
            # Update goal metrics
            team_history[home_id]['goals_for'] += home_goals
            team_history[home_id]['goals_against'] += away_goals
            team_history[away_id]['goals_for'] += away_goals
            team_history[away_id]['goals_against'] += home_goals
            
            # Update points and form based on result
            if home_goals > away_goals:  # Home win
                team_history[home_id]['points'] += 3
                team_history[home_id]['wins'] += 1
                team_history[home_id]['form'].append(1)
                team_history[away_id]['losses'] += 1
                team_history[away_id]['form'].append(0)
            elif home_goals == away_goals:  # Draw
                team_history[home_id]['points'] += 1
                team_history[home_id]['draws'] += 1
                team_history[home_id]['form'].append(0.5)
                team_history[away_id]['points'] += 1
                team_history[away_id]['draws'] += 1
                team_history[away_id]['form'].append(0.5)
            else:  # Away win
                team_history[away_id]['points'] += 3
                team_history[away_id]['wins'] += 1
                team_history[away_id]['form'].append(1)
                team_history[home_id]['losses'] += 1
                team_history[home_id]['form'].append(0)
            
            # Update match counts and dates
            team_history[home_id]['matches_played'] += 1
            team_history[away_id]['matches_played'] += 1
            team_history[home_id]['last_match_date'] = match_date
            team_history[away_id]['last_match_date'] = match_date
        
        def calculate_ranks(group):
            """
            Properly calculates ranks within each season/round group using:
            1. Points (descending)
            2. Goal Difference (descending)
            3. Goals For (descending)
            """
            # Create temporary dataframe for ranking calculations
            teams = {}
            for _, row in group.iterrows():
                for team_type in ['home', 'away']:
                    team_id = row[f'{team_type}_team_id']
                    if team_id not in teams:
                        teams[team_id] = {
                            'points': row[f'{team_type}_points'],
                            'gd': row[f'{team_type}_goals_diff'],
                            'gf': row[f'{team_type}_goals_for'],
                            'name': row[f'{team_type}_team']
                        }
            
            # Convert to DataFrame for ranking
            ranking_df = pd.DataFrame.from_dict(teams, orient='index')
            ranking_df['rank_key'] = ranking_df.apply(lambda x: (-x['points'], -x['gd'], -x['gf']), axis=1)
            ranking_df['rank'] = ranking_df['rank_key'].rank(method='min').astype(int)
            
            # Map ranks back to original dataframe
            for idx, row in group.iterrows():
                home_id = row['home_team_id']
                away_id = row['away_team_id']
                group.at[idx, 'home_rank'] = ranking_df.at[home_id, 'rank']
                group.at[idx, 'away_rank'] = ranking_df.at[away_id, 'rank']
            
            return group

        # Apply ranking calculation
        df = df.groupby(['season', 'round'], group_keys=False).apply(calculate_ranks)
    
        
        # Select output columns
        output_cols = [
            'fixture_id', 'date', 'season', 'round',
            'league_name', 'league_flag', 'league_logo',
            'venue_name', 'venue_id', 'referee', 'status',
            'home_team', 'home_team_id', 'home_team_flag', 
            'away_team', 'away_team_id', 'away_team_flag',
            'home_goals', 'away_goals'
        ] + list(features.keys())

      
        return df

    def _create_h2h_features(self, df: pd.DataFrame, LEAGUES: dict) -> pd.DataFrame:
        """Enhanced H2H calculator that works with incremental processing"""
        df = df.copy()
        
        # Add competition_type from LEAGUES dictionary
        def get_competition_type(league_id):
            for country, competitions in LEAGUES.items():
                for comp_id, comp_data in competitions.items():
                    if str(league_id) == comp_id:
                        return 'cup' if 'cup' in comp_data['name'].lower() else 'league'
            return 'league'
        
        df['competition_type'] = df['league_id'].apply(get_competition_type)
        
        # Initialize H2H columns
        h2h_features = [
            'h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
            'h2h_home_win_pct', 'h2h_away_win_pct',
            'h2h_recent_home_wins_last5', 'h2h_recent_away_wins_last5',
            'h2h_recent_draws_last5', 'h2h_recent_avg_goals_last5',
            'h2h_streak', 'h2h_avg_goals',
            'h2h_league_matches', 'h2h_cup_matches',
            'h2h_cup_home_wins', 'h2h_cup_away_wins',
            'h2h_same_country', 'h2h_win_streak', 'h2h_loss_streak'
        ]
        
        for col in h2h_features:
            df[col] = 0.0 if not col.endswith('_streak') else 0
        
        # Calculate competition averages from current data
        comp_avg_goals = {}
        comp_home_win_rates = {}
        
        for country in df['country'].unique():
            country_comps = df[df['country'] == country]['league_id'].unique()
            for comp_id in country_comps:
                comp_matches = df[
                    (df['league_id'] == comp_id) & 
                    (df['country'] == country)
                ]
                
                if len(comp_matches) > 0:
                    comp_key = f"{country}_{comp_id}"
                    comp_avg_goals[comp_key] = (comp_matches['home_goals'].mean() + 
                                            comp_matches['away_goals'].mean()) / 2
                    comp_home_win_rates[comp_key] = (comp_matches['home_goals'] > comp_matches['away_goals']).mean()

        # Calculate H2H features using stored data + new matches
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating H2H"):
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            current_date = row['date']
            competition_id = row['league_id']
            country = row['country']
            comp_type = row['competition_type']
            comp_key = f"{country}_{competition_id}"
            
            # Get historical matches from stored H2H data
            key = frozenset({home_id, away_id})
            stored_matches = self.h2h_data.get(key, [])
            
            # Convert stored matches to DataFrame format
            past_matches_list = []
            for match in stored_matches:
                if match['date'] < current_date:
                    past_matches_list.append({
                        'date': match['date'],
                        'home_team_id': match['home_id'],
                        'away_team_id': match['away_id'],
                        'home_goals': match['home_goals'],
                        'away_goals': match['away_goals'],
                        'league_id': match['league_id'],
                        'country': country,  # Assuming country is same for all stored matches
                        'competition_type': 'league'  # Default, adjust if you store this in h2h_data
                    })
            
            # Create DataFrame from stored matches
            past_matches = pd.DataFrame(past_matches_list) if past_matches_list else pd.DataFrame()
            
            # Teams are from same country
            df.at[idx, 'h2h_same_country'] = 1.0
            
            if len(past_matches) == 0:
                # Use competition averages if no history
                avg_goals = comp_avg_goals.get(comp_key, 2.5)
                home_win_rate = comp_home_win_rates.get(comp_key, 0.45)
                
                df.at[idx, 'h2h_avg_goals'] = avg_goals
                df.at[idx, 'h2h_home_win_pct'] = home_win_rate
                df.at[idx, 'h2h_away_win_pct'] = 1 - home_win_rate - 0.25
                continue
                
            # Split into league and cup matches
            league_matches = past_matches[past_matches['competition_type'] == 'league']
            cup_matches = past_matches[past_matches['competition_type'] == 'cup']
            
            # Store counts
            df.at[idx, 'h2h_league_matches'] = len(league_matches)
            df.at[idx, 'h2h_cup_matches'] = len(cup_matches)
            
            # Calculate stats for all matches combined
            self._calculate_h2h_stats(df, idx, past_matches, home_id, away_id, 'h2h_')
            
            # Calculate cup-specific stats if available
            if len(cup_matches) > 0:
                self._calculate_h2h_stats(df, idx, cup_matches, home_id, away_id, 'h2h_cup_')
            
            # Calculate streak using your enhanced method
            streak_value = self._calculate_streak(past_matches, home_id)
            df.at[idx, 'h2h_streak'] = streak_value
            
            # Additional streak features
            df.at[idx, 'h2h_win_streak'] = max(streak_value, 0)
            df.at[idx, 'h2h_loss_streak'] = abs(min(streak_value, 0))
        
        return df

    def _calculate_h2h_stats(self, df, idx, matches, home_id, away_id, prefix):
        """Helper method to calculate H2H stats for a match subset"""
        total_matches = len(matches)
        if total_matches == 0:
            return
            
        # Home wins (from perspective of current home team)
        home_wins = len(matches[
            ((matches['home_team_id'] == home_id) & 
            (matches['home_goals'] > matches['away_goals'])) |
            ((matches['away_team_id'] == home_id) & 
            (matches['away_goals'] > matches['home_goals']))
        ])
        
        # Away wins (from perspective of current away team)
        away_wins = len(matches[
            ((matches['home_team_id'] == away_id) & 
            (matches['home_goals'] > matches['away_goals'])) |
            ((matches['away_team_id'] == away_id) & 
            (matches['away_goals'] > matches['home_goals']))
        ])
        
        draws = len(matches[matches['home_goals'] == matches['away_goals']])
        
        # Goal stats
        home_goals = matches.apply(
            lambda x: x['home_goals'] if x['home_team_id'] == home_id else x['away_goals'], axis=1).sum()
        away_goals = matches.apply(
            lambda x: x['away_goals'] if x['home_team_id'] == home_id else x['home_goals'], axis=1).sum()
        
        # Apply to DataFrame
        df.at[idx, f'{prefix}matches'] = total_matches
        df.at[idx, f'{prefix}home_wins'] = home_wins
        df.at[idx, f'{prefix}away_wins'] = away_wins
        df.at[idx, f'{prefix}draws'] = draws
        df.at[idx, f'{prefix}home_goals'] = home_goals
        df.at[idx, f'{prefix}away_goals'] = away_goals
        df.at[idx, f'{prefix}goal_diff'] = home_goals - away_goals
        df.at[idx, f'{prefix}home_win_pct'] = home_wins / total_matches if total_matches > 0 else 0
        df.at[idx, f'{prefix}away_win_pct'] = away_wins / total_matches if total_matches > 0 else 0
        df.at[idx, f'{prefix}avg_goals'] = (home_goals + away_goals) / total_matches if total_matches > 0 else 0
        
        # Recent form (last 5 matches)
        recent_matches = matches.head(5)
        if len(recent_matches) > 0:
            df.at[idx, f'{prefix}recent_home_wins_last5'] = len(recent_matches[
                ((recent_matches['home_team_id'] == home_id) & 
                (recent_matches['home_goals'] > recent_matches['away_goals'])) |
                ((recent_matches['away_team_id'] == home_id) & 
                (recent_matches['away_goals'] > recent_matches['home_goals']))
            ])
            
            df.at[idx, f'{prefix}recent_away_wins_last5'] = len(recent_matches[
                ((recent_matches['home_team_id'] == away_id) & 
                (recent_matches['home_goals'] > recent_matches['away_goals'])) |
                ((recent_matches['away_team_id'] == away_id) & 
                (recent_matches['away_goals'] > recent_matches['home_goals']))
            ])
            
            df.at[idx, f'{prefix}recent_draws_last5'] = len(recent_matches[
                recent_matches['home_goals'] == recent_matches['away_goals']
            ])
            
            df.at[idx, f'{prefix}recent_avg_goals_last5'] = (
                recent_matches['home_goals'].sum() + recent_matches['away_goals'].sum()
            ) / len(recent_matches)

    def _calculate_streak(self, matches, home_team):
        """
        Enhanced streak calculation with:
        - Exponential weighting of recent matches
        - Goal difference consideration
        - Returns float value between -1 (terrible streak) and 1 (excellent streak)
        """
        if matches.empty:
            return 0.0
        
        streak_score = 0.0
        total_weight = 0.0
        streak_direction = None
        consecutive_count = 0
        
        for i, (_, match) in enumerate(matches.sort_values('date', ascending=False).iterrows()):
            # Determine perspective
            is_home = match['home_team_id'] == home_team
            goals_for = match['home_goals'] if is_home else match['away_goals']
            goals_against = match['away_goals'] if is_home else match['home_goals']
            
            # Calculate match result and weight
            weight = 0.8 ** i  # Exponential decay (most recent match has highest weight)
            
            if goals_for > goals_against:  # Win
                current_direction = 1
                result_score = 1.0
            elif goals_for < goals_against:  # Loss
                current_direction = -1
                result_score = 0.0
            else:  # Draw
                current_direction = 0
                result_score = 0.5
            
            # Add goal difference factor (capped at ±3)
            gd_factor = min(3, max(-3, goals_for - goals_against)) / 12  # ±0.25 max effect
            
            # Track consecutive results in same direction
            if streak_direction is None:
                streak_direction = current_direction
                consecutive_count = 1
            elif current_direction == streak_direction:
                consecutive_count += 1
            else:
                break  # Streak broken
            
            # Calculate weighted contribution
            match_contribution = (result_score + gd_factor) * weight
            streak_score += match_contribution
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize and apply streak length bonus
        normalized_streak = streak_score / total_weight
        streak_bonus = min(0.2, consecutive_count * 0.05)  # Max 20% bonus for long streaks
        
        if streak_direction == 1:  # Winning streak
            final_score = min(1.0, normalized_streak + streak_bonus)
        elif streak_direction == -1:  # Losing streak
            final_score = max(-1.0, normalized_streak - streak_bonus)
        else:  # Drawing streak
            final_score = normalized_streak
        
        return round(final_score, 4)

    # Add this to your _update_h2h_data method to store competition_type:
    def _update_h2h_data(self, new_matches: pd.DataFrame):
        """Update H2H data with new matches while preserving history"""
        for _, match in new_matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            key = frozenset({home_id, away_id})
            
            if key not in self.h2h_data:
                self.h2h_data[key] = []
            
            # Store all needed information including competition type
            self.h2h_data[key].append({
                'date': match['date'],
                'home_id': home_id,
                'home_goals': match['home_goals'],
                'away_id': away_id,
                'away_goals': match['away_goals'],
                'league_id': match['league_id'],
                'competition_type': match.get('competition_type', 'league')  # Add this
            })

    def run_pipeline(self) -> pd.DataFrame:
        """Run pipeline with full data concatenation before preprocessing"""
        data_structure = self._discover_data_structure()
        all_raw_data = pd.DataFrame()
        
        # 1. First gather ALL raw data while maintaining folder structure
        self.logger.info("Gathering all raw data...")
        for country, leagues in data_structure.items():
            country_dir = Path(self.config['merged_dir']) / country
            country_dir.mkdir(parents=True, exist_ok=True)
            
            for league, seasons in leagues.items():
                league_dir = country_dir / league
                league_dir.mkdir(exist_ok=True)
                
                # Handle incremental mode filtering
                if self.config['incremental_mode'] and self.config['current_season']:
                    seasons = [s for s in seasons if s == self.config['current_season']]
                    if not seasons:
                        continue
                
                league_data = pd.DataFrame()
                for season in seasons:
                    season_data = self._process_single_season(country, league, season)
                    if season_data is not None:
                        league_data = pd.concat([league_data, season_data], ignore_index=True)
                
                if not league_data.empty:
                    # Save raw merged league data (optional)
                    league_path = league_dir / "all_seasons_merged.csv"
                    league_data.to_csv(league_path, index=False)
                    self.logger.info(f"Saved raw merged data to {league_path}")
                    
                    # Add to complete dataset
                    all_raw_data = pd.concat([all_raw_data, league_data], ignore_index=True)
        
        if all_raw_data.empty:
            self.logger.warning("\nPipeline completed but no data was processed")
            return pd.DataFrame()
        
        # 2. Preprocess the complete concatenated dataset
        self.logger.info("\nPreprocessing complete dataset...")
        processed_data = self._preprocess_and_feature_engineer(all_raw_data)
        
        # 3. Save final output
        if not processed_data.empty:
            processed_data.to_csv(self.config['final_output'], index=False)
            self.logger.info(f"Saved final processed data to {self.config['final_output']}")
            self.logger.info(f"Final Dataset contains {len(processed_data)} records and {len(processed_data.columns)} features.")
        
        return processed_data

    def _preprocess_and_feature_engineer(self, df: pd.DataFrame, is_incremental: bool = False, current_season: str = None) -> pd.DataFrame:
        """
        Enhanced preprocessing with optimized incremental processing:
        1. Standings: Season-specific, only recalculates current season in incremental mode
        2. H2H: Uses complete historical data from h2h_store
        3. Other features: Processed normally with rolling averages
        """
        df = df.copy()
        
        # 1. Always perform basic preprocessing
        df = self._clean_data(df)
        df = self._create_temporal_features(df)
        df = self._create_target_column(df)
        
        # 2. Handle standings - special incremental logic
        if is_incremental and current_season:
            self.logger.info(f"Processing standings incrementally for season {current_season}")
            
            # Split data into current season and historical data
            current_mask = df['season'] == current_season
            current_data = df[current_mask].copy()
            historical_data = df[~current_mask].copy()
            
            if not current_data.empty:
                # Calculate standings only for current season
                current_data = self._create_standings(current_data)
                
                # Combine with historical data (preserving existing standings)
                df = pd.concat([historical_data, current_data], ignore_index=True)
        else:
            # Full processing - calculate standings for all data
            df = self._create_standings(df)
        
        # 3. Update H2H store with any new matches before calculation
        if is_incremental and current_season:
            new_matches = df[df['season'] == current_season]
            if not new_matches.empty:
                self._update_h2h_data(new_matches)
                self._save_h2h_data()
        
        # 4. Calculate H2H features using complete stored history
        df = self._create_h2h_features(df, LEAGUES)
        
        # 5. Calculate other features with optimized rolling averages
        df = self._add_new_metrics(df)
        
        # Special rolling average handling for incremental updates
        if is_incremental and current_season:
            # Only update rolling averages for current season teams
            current_teams = set(df[df['season'] == current_season]['home_team_id']).union(
                            set(df[df['season'] == current_season]['away_team_id']))
            
            for window in self.config['rolling_windows']:
                for feature in self.combined_metrics:
                    if feature in df.columns:
                        rolling_col = f"{feature}_rolling_{window}"
                        
                        # Only update rolling features for current season teams
                        mask = (df['home_team_id'].isin(current_teams)) | (df['away_team_id'].isin(current_teams))
                        df.loc[~mask, rolling_col] = np.nan  # Reset for affected teams
                        
                        # Recalculate for current season teams
                        if feature.startswith('home_'):
                            df.loc[mask, rolling_col] = df[mask].groupby(
                                ['season', 'home_team_id'])[feature].transform(
                                lambda x: x.rolling(window, min_periods=1).mean())
                        elif feature.startswith('away_'):
                            df.loc[mask, rolling_col] = df[mask].groupby(
                                ['season', 'away_team_id'])[feature].transform(
                                lambda x: x.rolling(window, min_periods=1).mean())
                        else:
                            # For non-team-specific features
                            home_vals = df[mask].groupby(
                                ['season', 'home_team_id'])[feature].transform(
                                lambda x: x.rolling(window, min_periods=1).mean())
                            away_vals = df[mask].groupby(
                                ['season', 'away_team_id'])[feature].transform(
                                lambda x: x.rolling(window, min_periods=1).mean())
                            df.loc[mask, rolling_col] = (home_vals + away_vals) / 2
        else:
            # Full rolling average calculation
            df = self._calculate_rolling_averages(df)
        
        return df


    def run_incremental(self, current_season: str) -> pd.DataFrame:
        """Process new data with optimized H2H handling"""
        # 1. Process new matches
        new_data = self._process_new_data(current_season)
        
        if new_data.empty:
            return self._load_final_data()
        
        # 2. Update H2H store with new matches
        self._update_h2h_data(new_data)
        
        # 3. Load existing processed data
        existing_data = self._load_final_data()
        
        # 4. Calculate standings only for new season
        new_data = self._create_standings(new_data)
        
        # 5. Combine with existing data
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # 6. Apply H2H features using stored H2H data
        combined_data = self._create_h2h_features(combined_data)
        
        # 7. Calculate other features
        combined_data = self._add_new_metrics(combined_data)
        combined_data = self._calculate_rolling_averages(combined_data)
        
        # 8. Save updated data
        self._save_final_data(combined_data)
        self._save_h2h_data()
        
        return combined_data




# Usage example
if __name__ == "__main__":
    try:
        pipeline = FootballDataPipeline(log_dir="logs")
        full_data = pipeline.run_pipeline()
        pipeline.logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.getLogger(__name__).critical(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise