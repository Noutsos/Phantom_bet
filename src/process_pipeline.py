from datetime import datetime, timedelta
import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings
import logging
import os
import pickle
#from tqdm import tqdm

warnings.filterwarnings("ignore")
from src.utils import LEAGUES  # Assuming LEAGUES is defined in utils.py

class ProcessPipeline:
    """
    Enhanced pipeline with incremental processing support:
    1. Processes only new data when available
    2. Maintains standings and H2H history across runs
    3. Comprehensive logging system
    """
    
    def __init__(self, log_dir="logs/process", config: Optional[Dict] = None):
        # Default configuration
        self.config = {
            'raw_dir': 'data/extracted',
            'merged_dir': 'data/processed',
            'final_output': 'data/final_processed.csv',
            'verbose': True,
            'data_types': {
                'fixtures': 'fixture_events.csv',
                'team_stats': 'team_statistics.csv',
                'odds' : 'odds.csv'
            },
            'required_cols': {
                'fixtures': ['fixture_id', 'home_team_id', 'away_team_id', 'date'],
                'team_stats': ['fixture_id', 'team_id']
            },
            'rolling_windows': [5],
            'min_matches': 5,
            'merge_first': True,
            'h2h_store': 'data/processed/h2h_store.pkl',
            'standings_store': 'data/processed/standings_store.pkl'
        }
        
        if config:
            self.config.update(config)
        
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)

        self.standings_data = {}  # Initialize here
        self.h2h_data = {}  # If you have this too
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        Path(self.config['merged_dir']).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load historical data
        self.h2h_data = self._load_h2h_data()
        self.standings_data = self._load_standings_data()



    def _setup_logging(self):
        """Set up logging with file handler only"""
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
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
                    self.logger.info(f"Loaded H2H data with {len(data)} team pairs")
                    return data
            except Exception as e:
                self.logger.error(f"Failed to load H2H data: {str(e)}")
                return {}
        else:
            self.logger.info("No existing H2H data found")
            return {}

    def _save_h2h_data(self):
        """Save current H2H data to disk"""
        try:
            with open(self.config['h2h_store'], 'wb') as f:
                pickle.dump(self.h2h_data, f)
            self.logger.info(f"Saved H2H data with {len(self.h2h_data)} team pairs")
        except Exception as e:
            self.logger.error(f"Failed to save H2H data: {str(e)}")

    def _load_standings_data(self) -> Dict:
        """Load stored standings data from disk"""
        standings_path = Path(self.config['standings_store'])
        if standings_path.exists():
            try:
                with open(standings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.logger.info(f"Loaded standings data for {len(data)} teams")
                    return data
            except Exception as e:
                self.logger.error(f"Failed to load standings data: {str(e)}")
                return {}
        else:
            self.logger.info("No existing standings data found")
            return {}

    def _save_standings_data(self):
        """Save current standings data to disk"""
        try:
            with open(self.config['standings_store'], 'wb') as f:
                pickle.dump(self.standings_data, f)
            self.logger.info(f"Saved standings data for {len(self.standings_data)} teams")
        except Exception as e:
            self.logger.error(f"Failed to save standings data: {str(e)}")

    def _discover_data_structure(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover available countries, leagues and seasons"""
        structure = {}
        raw_path = Path(self.config['raw_dir'])
        
        if not raw_path.exists():
            self.logger.warning(f"Raw directory {self.config['raw_dir']} does not exist")
            return structure
        
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

    def _get_processed_fixtures(self) -> set:
        """Get set of already processed fixture IDs"""
        final_path = Path(self.config['final_output'])
        if final_path.exists():
            try:
                existing_data = pd.read_csv(final_path)
                if 'fixture_id' in existing_data.columns:
                    return set(existing_data['fixture_id'].unique())
            except Exception as e:
                self.logger.warning(f"Error reading existing fixtures: {e}")
        return set()

    def _check_new_data_exists(self) -> bool:
        """Check if there's any new data to process"""
        data_structure = self._discover_data_structure()
        processed_fixtures = self._get_processed_fixtures()
        
        if not data_structure:
            self.logger.warning("No data structure found in raw directory")
            return False
            
        for country, leagues in data_structure.items():
            for league, seasons in leagues.items():
                for season in seasons:
                    season_path = Path(self.config['raw_dir']) / country / league / season
                    fixtures_path = season_path / self.config['data_types']['fixtures']
                    
                    if fixtures_path.exists():
                        try:
                            fixtures = pd.read_csv(fixtures_path)
                            if 'fixture_id' in fixtures.columns:
                                new_fixtures = set(fixtures['fixture_id']) - processed_fixtures
                                if new_fixtures:
                                    self.logger.info(f"Found {len(new_fixtures)} new fixtures in {country}/{league}/{season}")
                                    return True
                        except Exception as e:
                            self.logger.warning(f"Error checking fixtures in {country}/{league}/{season}: {e}")
        
        self.logger.info("No new data found to process")
        return False


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

        # Remove specific problematic columns instead of truncating
        columns_to_remove = [
          'Thorwins', 'Substitutions', 'Assists', 'Medical Treatment', 'Goal Attempts', 'Counter Attacks',
          'Free Kicks', 'Cross Attacks', 'Goals'
           
        ]
        
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

        # 5. Filter FT matches only if status exists
        if 'status' in df.columns:
            df = df[df['status'].isin(['FT', 'AET', 'PEN', 'NS'])]
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
                    print(f"Error processing {poss_col}: {e}")
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
                    print(f"Error processing {pass_pct_col}: {e}")
                    df[pass_pct_col] = np.nan
                
        # Filter only FT games for the shot data check
        ft_games = df[df['status'] == 'FT'].copy()
        
        # Check for missing data in FT games only
        missing_data = ft_games[
            ['home_goals', 'away_goals',
            'halftime_home', 'halftime_away',
            'fulltime_home', 'fulltime_away']
        ].isna().any(axis=1)
        
        # Get indices of FT games with missing data
        bad_ft_indices = ft_games[missing_data].index
        
        # Drop only those FT games with missing data
        df = df.drop(bad_ft_indices)  

        # 4. Convert numeric columns with proper null handling
        numeric_cols = [
            'maintime', 'first_half', 'second_half', 
            'home_goals', 'away_goals', 
            'halftime_home', 'halftime_away',
            'fulltime_home', 'fulltime_away'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # 5. Handle float columns that should stay float
        float_cols = ['extratime']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')


        # 7. Handle missing values
        fill_rules = {
            'extratime': 0,
            'penalty_home': 0,
            'penalty_away': 0,
            'home_shots_on_goal': 0,
            'away_shots_on_goal': 0,
            'home_shots_off_goal': 0,
            'away_shots_off_goal': 0,
            'home_total_shots': 0,
            'away_total_shots': 0,
            'home_blocked_shots': 0,
            'away_blocked_shots': 0,
            'home_shots_insidebox': 0,
            'away_shots_insidebox': 0,
            'home_shots_outsidebox': 0,
            'away_shots_outsidebox': 0,
            'home_ball_possession': 0.5,  # Default to 50% if missing
            'away_ball_possession': 0.5,  # Default to 50% if missing
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

    def _create_target_column_2(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Creates target outcome column, handling both completed and NS games.
            
            For completed matches (FT status with goals):
            - home_win: home_goals > away_goals
            - away_win: home_goals < away_goals  
            - draw: home_goals == away_goals
            
            For NS games:
            - outcome set to 'NS' (Not Started)
            
            Args:
                df: DataFrame containing match data with home_goals, away_goals columns
                
            Returns:
                DataFrame with added 'outcome' column
            """
            df = df.copy()  # Avoid modifying original DataFrame
            
            # Initialize outcome column with 'NS' as default
            df['outcome'] = 'NS'
            
            # Only process rows where we have goal data (completed matches)
            if all(col in df.columns for col in ['home_goals', 'away_goals']):
                # Create mask for completed matches
                completed_mask = df['home_goals'].notna() & df['away_goals'].notna()
                
                # Only process completed matches
                if completed_mask.any():
                    completed_df = df[completed_mask].copy()
                    
                    # Create boolean conditions
                    home_wins = (completed_df['home_goals'] > completed_df['away_goals'])
                    away_wins = (completed_df['home_goals'] < completed_df['away_goals'])

                    total_goals = completed_df['home_goals'] + completed_df['away_goals']
                    total_corners = completed_df.get('home_corners', 0) + completed_df.get('away_corners', 0)
                    
                    # Apply to original dataframe
                    df.loc[completed_mask, 'outcome'] = np.select(
                        condlist=[home_wins, away_wins],
                        choicelist=['home_win', 'away_win'],
                        default='draw'
                    )
                    df.loc[completed_mask, 'total_goals'] = total_goals
                    df.loc[completed_mask, 'total_corners'] = total_corners
            
            return df 
 
 
    def _create_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates multiple target columns for both classification and regression tasks.
        
        For completed matches:
        - outcome: classification target (home_win, away_win, draw)
        - total_goals: regression target
        - total_corners: regression target  
        - total_cards: regression target (if available)
        - total_shots: regression target (if available)
        
        For NS games: targets set to NaN
        
        Args:
            df: DataFrame containing match data
            
        Returns:
            DataFrame with added target columns
        """
        df = df.copy()  # Avoid modifying original DataFrame
        logging.info("Creating target columns...")
        # Initialize target columns with NaN as default
        target_columns = ['outcome', 'total_goals', 'total_corners', 'total_yellow_cards', 'total_red_cards']
        for col in target_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Only process rows where we have goal data (completed matches)
        if all(col in df.columns for col in ['home_goals', 'away_goals']):
            # Create mask for completed matches
            completed_mask = df['home_goals'].notna() & df['away_goals'].notna()
            
            # Only process completed matches
            if completed_mask.any():
                completed_df = df[completed_mask].copy()
                
                # Create outcome column (classification)
                home_wins = (completed_df['home_goals'] > completed_df['away_goals'])
                away_wins = (completed_df['home_goals'] < completed_df['away_goals'])
                
                df.loc[completed_mask, 'outcome'] = np.select(
                    condlist=[home_wins, away_wins],
                    choicelist=['home_win', 'away_win'],
                    default='draw'
                )
                
                # Create regression targets
                df.loc[completed_mask, 'total_goals'] = (
                    completed_df['home_goals'] + completed_df['away_goals']
                )
                
                # Optional regression targets (if columns exist)
                if all(col in df.columns for col in ['home_corner_kicks', 'away_corner_kicks']):
                    df.loc[completed_mask, 'total_corners'] = (
                        completed_df['home_corner_kicks'] + completed_df['away_corner_kicks']
                    )
                
                if all(col in df.columns for col in ['home_yellow_cards', 'away_yellow_cards']):
                    df.loc[completed_mask, 'total_yellow_cards'] = (
                        completed_df['home_yellow_cards'] + completed_df['away_yellow_cards']
                    )
               

                if all(col in df.columns for col in ['home_red_cards', 'away_red_cards']):
                    df.loc[completed_mask, 'total_red_cards'] = (
                        completed_df['home_red_cards'] + completed_df['away_red_cards']
                    )
           


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
            days_since_col = f'{team_type}_days_rest'
            
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

        
    def _create_standings_2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fixed standings calculation with proper ranking
        """
        df = df.copy()
        
        # Ensure perfect chronological order
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['league_id', 'season', 'date', 'fixture_id']).reset_index(drop=True)
        
        # Initialize standings columns
        self._standings_metrics = [
            'home_goals_for', 'away_goals_for',
            'home_goals_against', 'away_goals_against',
            'home_goals_diff', 'away_goals_diff'
        ]
        self._basic_standings_features = ['home_played', 'away_played',
            'home_wins', 'away_wins', 'home_draws', 'away_draws', 'home_losses', 'away_losses',
             'home_rank', 'away_rank']

        combined_standings = self._standings_metrics + self._basic_standings_features
        
        for col in combined_standings:
            df[col] = 0
        
        # Dictionary to track cumulative standings
        current_standings = {}
        
        total_matches = len(df)
        self.logger.info(f"Calculating proper standings for {total_matches} matches...")
        start_time = time.time()
        
        # Process each match in chronological order
        for idx, row in df.iterrows():
            fixture_id = row['fixture_id']
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            league_id = row['league_id']
            season = row['season']
            
            home_key = (home_id, league_id, season)
            away_key = (away_id, league_id, season)
            
            # Initialize teams if not exists
            if home_key not in current_standings:
                current_standings[home_key] = {
                    'points': 0, 'goals_for': 0, 'goals_against': 0,
                    'wins': 0, 'draws': 0, 'losses': 0, 'matches_played': 0,
                    'goal_diff': 0, 'team_id': home_id
                }
            
            if away_key not in current_standings:
                current_standings[away_key] = {
                    'points': 0, 'goals_for': 0, 'goals_against': 0,
                    'wins': 0, 'draws': 0, 'losses': 0, 'matches_played': 0,
                    'goal_diff': 0, 'team_id': away_id
                }
            
            # Store PRE-MATCH standings
            home_pre = current_standings[home_key].copy()
            away_pre = current_standings[away_key].copy()
            
            # Set PRE-MATCH standings in dataframe
            df.at[idx, 'home_points'] = home_pre['points']
            df.at[idx, 'home_goals_for'] = home_pre['goals_for']
            df.at[idx, 'home_goals_against'] = home_pre['goals_against']
            df.at[idx, 'home_played'] = home_pre['matches_played']
            df.at[idx, 'home_wins'] = home_pre['wins']
            df.at[idx, 'home_draws'] = home_pre['draws']
            df.at[idx, 'home_losses'] = home_pre['losses']
            df.at[idx, 'home_goals_diff'] = home_pre['goal_diff']
            
            df.at[idx, 'away_points'] = away_pre['points']
            df.at[idx, 'away_goals_for'] = away_pre['goals_for']
            df.at[idx, 'away_goals_against'] = away_pre['goals_against']
            df.at[idx, 'away_played'] = away_pre['matches_played']
            df.at[idx, 'away_wins'] = away_pre['wins']
            df.at[idx, 'away_draws'] = away_pre['draws']
            df.at[idx, 'away_losses'] = away_pre['losses']
            df.at[idx, 'away_goals_diff'] = away_pre['goal_diff']
            
            # FIXED: Calculate rankings for current league/season
            # Get all teams in this league/season with their current standings
            league_teams = []
            for key, stats in current_standings.items():
                if key[1] == league_id and key[2] == season:
                    team_stats = stats.copy()
                    team_stats['team_id'] = key[0]  # Add team_id to the stats
                    league_teams.append(team_stats)
            
            # Sort by points, goal difference, goals for
            league_teams.sort(key=lambda x: (-x['points'], -x['goal_diff'], -x['goals_for']))
            
            # Assign ranks (handle ties properly)
            ranks = {}
            current_rank = 1
            prev_stats = None
            
            for i, team in enumerate(league_teams):
                current_stats = (team['points'], team['goal_diff'], team['goals_for'])
                
                if prev_stats is not None and current_stats == prev_stats:
                    # Same rank for tied teams
                    ranks[team['team_id']] = current_rank
                else:
                    current_rank = i + 1
                    ranks[team['team_id']] = current_rank
                
                prev_stats = current_stats
            
            # Apply ranks to current match
            df.at[idx, 'home_rank'] = ranks.get(home_id, 999)
            df.at[idx, 'away_rank'] = ranks.get(away_id, 999)
            
            # Update POST-MATCH standings for completed matches
            if row['status'] in ['FT', 'AET', 'PEN']:
                try:
                    home_goals = int(row['home_goals'])
                    away_goals = int(row['away_goals'])
                    
                    # Update goals and matches played
                    current_standings[home_key]['goals_for'] += home_goals
                    current_standings[home_key]['goals_against'] += away_goals
                    current_standings[home_key]['goal_diff'] = current_standings[home_key]['goals_for'] - current_standings[home_key]['goals_against']
                    current_standings[home_key]['matches_played'] += 1
                    
                    current_standings[away_key]['goals_for'] += away_goals
                    current_standings[away_key]['goals_against'] += home_goals
                    current_standings[away_key]['goal_diff'] = current_standings[away_key]['goals_for'] - current_standings[away_key]['goals_against']
                    current_standings[away_key]['matches_played'] += 1
                    
                    # Update points and results
                    if home_goals > away_goals:
                        current_standings[home_key]['points'] += 3
                        current_standings[home_key]['wins'] += 1
                        current_standings[away_key]['losses'] += 1
                    elif home_goals == away_goals:
                        current_standings[home_key]['points'] += 1
                        current_standings[home_key]['draws'] += 1
                        current_standings[away_key]['points'] += 1
                        current_standings[away_key]['draws'] += 1
                    else:
                        current_standings[away_key]['points'] += 3
                        current_standings[away_key]['wins'] += 1
                        current_standings[home_key]['losses'] += 1
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error updating standings for fixture {fixture_id}: {e}")
            


            # Progress logging
            if (idx + 1) % max(1, len(total_matches) // 10) == 0 or (idx + 1) == len(total_matches):
                progress = ((idx + 1) / len(total_matches)) * 100
                elapsed = time.time() - start_time
                self.logger.info(f"H2H Progress: {progress:.0f}% ({idx + 1}/{len(total_matches)}) - {elapsed:.1f}s")
        
        # Update global standings
        self.standings_data.update(current_standings)

        
        return df

    def _create_standings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fixed standings calculation with proper ranking
        """
        df = df.copy()
        
        # Ensure perfect chronological order
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['league_id', 'season', 'date', 'fixture_id']).reset_index(drop=True)
        
        # Initialize standings columns
        self._standings_metrics = [
            'home_goals_for', 'away_goals_for',
            'home_goals_against', 'away_goals_against',
            'home_goals_diff', 'away_goals_diff'
        ]
        self._basic_standings_features = ['home_played', 'away_played',
            'home_wins', 'away_wins', 'home_draws', 'away_draws', 'home_losses', 'away_losses',
            'home_rank', 'away_rank']

        combined_standings = self._standings_metrics + self._basic_standings_features
        
        for col in combined_standings:
            df[col] = 0
        
        # Dictionary to track cumulative standings
        current_standings = {}
        
        total_matches = len(df)
        self.logger.info(f"Calculating proper standings for {total_matches} matches...")
        start_time = time.time()
        
        # Progress tracking variables
        progress_interval = 10  # percentage
        last_reported_progress = -progress_interval
        
        # Process each match in chronological order
        for idx, row in df.iterrows():
            fixture_id = row['fixture_id']
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            league_id = row['league_id']
            season = row['season']
            
            home_key = (home_id, league_id, season)
            away_key = (away_id, league_id, season)
            
            # Initialize teams if not exists
            if home_key not in current_standings:
                current_standings[home_key] = {
                    'points': 0, 'goals_for': 0, 'goals_against': 0,
                    'wins': 0, 'draws': 0, 'losses': 0, 'matches_played': 0,
                    'goal_diff': 0, 'team_id': home_id
                }
            
            if away_key not in current_standings:
                current_standings[away_key] = {
                    'points': 0, 'goals_for': 0, 'goals_against': 0,
                    'wins': 0, 'draws': 0, 'losses': 0, 'matches_played': 0,
                    'goal_diff': 0, 'team_id': away_id
                }
            
            # Store PRE-MATCH standings
            home_pre = current_standings[home_key].copy()
            away_pre = current_standings[away_key].copy()
            
            # Set PRE-MATCH standings in dataframe
            df.at[idx, 'home_points'] = home_pre['points']
            df.at[idx, 'home_goals_for'] = home_pre['goals_for']
            df.at[idx, 'home_goals_against'] = home_pre['goals_against']
            df.at[idx, 'home_played'] = home_pre['matches_played']
            df.at[idx, 'home_wins'] = home_pre['wins']
            df.at[idx, 'home_draws'] = home_pre['draws']
            df.at[idx, 'home_losses'] = home_pre['losses']
            df.at[idx, 'home_goals_diff'] = home_pre['goal_diff']
            
            df.at[idx, 'away_points'] = away_pre['points']
            df.at[idx, 'away_goals_for'] = away_pre['goals_for']
            df.at[idx, 'away_goals_against'] = away_pre['goals_against']
            df.at[idx, 'away_played'] = away_pre['matches_played']
            df.at[idx, 'away_wins'] = away_pre['wins']
            df.at[idx, 'away_draws'] = away_pre['draws']
            df.at[idx, 'away_losses'] = away_pre['losses']
            df.at[idx, 'away_goals_diff'] = away_pre['goal_diff']
            
            # FIXED: Calculate rankings for current league/season
            # Get all teams in this league/season with their current standings
            league_teams = []
            for key, stats in current_standings.items():
                if key[1] == league_id and key[2] == season:
                    team_stats = stats.copy()
                    team_stats['team_id'] = key[0]  # Add team_id to the stats
                    league_teams.append(team_stats)
            
            # Sort by points, goal difference, goals for
            league_teams.sort(key=lambda x: (-x['points'], -x['goal_diff'], -x['goals_for']))
            
            # Assign ranks (handle ties properly)
            ranks = {}
            current_rank = 1
            prev_stats = None
            
            for i, team in enumerate(league_teams):
                current_stats = (team['points'], team['goal_diff'], team['goals_for'])
                
                if prev_stats is not None and current_stats == prev_stats:
                    # Same rank for tied teams
                    ranks[team['team_id']] = current_rank
                else:
                    current_rank = i + 1
                    ranks[team['team_id']] = current_rank
                
                prev_stats = current_stats
            
            # Apply ranks to current match
            df.at[idx, 'home_rank'] = ranks.get(home_id, 999)
            df.at[idx, 'away_rank'] = ranks.get(away_id, 999)
            
            # Update POST-MATCH standings for completed matches
            if row['status'] in ['FT', 'AET', 'PEN']:
                try:
                    home_goals = int(row['home_goals'])
                    away_goals = int(row['away_goals'])
                    
                    # Update goals and matches played
                    current_standings[home_key]['goals_for'] += home_goals
                    current_standings[home_key]['goals_against'] += away_goals
                    current_standings[home_key]['goal_diff'] = current_standings[home_key]['goals_for'] - current_standings[home_key]['goals_against']
                    current_standings[home_key]['matches_played'] += 1
                    
                    current_standings[away_key]['goals_for'] += away_goals
                    current_standings[away_key]['goals_against'] += home_goals
                    current_standings[away_key]['goal_diff'] = current_standings[away_key]['goals_for'] - current_standings[away_key]['goals_against']
                    current_standings[away_key]['matches_played'] += 1
                    
                    # Update points and results
                    if home_goals > away_goals:
                        current_standings[home_key]['points'] += 3
                        current_standings[home_key]['wins'] += 1
                        current_standings[away_key]['losses'] += 1
                    elif home_goals == away_goals:
                        current_standings[home_key]['points'] += 1
                        current_standings[home_key]['draws'] += 1
                        current_standings[away_key]['points'] += 1
                        current_standings[away_key]['draws'] += 1
                    else:
                        current_standings[away_key]['points'] += 3
                        current_standings[away_key]['wins'] += 1
                        current_standings[home_key]['losses'] += 1
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error updating standings for fixture {fixture_id}: {e}")
            
            # Progress logging every 10%
            current_progress = int(((idx + 1) / total_matches) * 100)
            if current_progress >= last_reported_progress + progress_interval or (idx + 1) == total_matches:
                elapsed = time.time() - start_time
                self.logger.info(f"Standings Progress: {current_progress}% ({idx + 1}/{total_matches}) - {elapsed:.1f}s")
                last_reported_progress = current_progress
        
        # Update global standings
        self.standings_data.update(current_standings)

        return df
 
    def _calculate_rolling_standings(self, df: pd.DataFrame, windows: list = [5]) -> pd.DataFrame:
        """
        Calculate rolling standings metrics for multiple window sizes
        """
        df = df.copy()
        
        # Ensure chronological order
        df = df.sort_values(['home_team_id', 'date'])
        
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            points_col = f'{prefix}_points'
            
            for window in windows:
                # Rolling points per game
                df[f'{prefix}_ppg_{window}'] = df.groupby(team_col)[points_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                ).round(2)
                
                # Rolling goals for
                df[f'{prefix}_gf_{window}'] = df.groupby(team_col)[f'{prefix}_goals'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                ).round(1)
                
                # Rolling goals against
                df[f'{prefix}_ga_{window}'] = df.groupby(team_col)[f'{"away" if prefix == "home" else "home"}_goals'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                ).round(1)
                
                # Rolling goal difference
                df[f'{prefix}_gd_{window}'] = df[f'{prefix}_gf_{window}'] - df[f'{prefix}_ga_{window}']
        
                # SIMPLE DROP LOGIC - ONLY THIS PART CHANGED
        if self.config.get('drop_non_roll_standings', False):
            cols_to_drop = [col for col in self._standings_metrics if col in df.columns]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            self.logger.info(f"Dropped {len(cols_to_drop)} non rolled standings metrics")
        
        return df
 

    def _calculate_form_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate form strength based on data BEFORE each current game
        Filter by league_id and season to ensure calculations are within the same competition and season
        """
        df = df.copy()
        
        # Validate that we have the required columns
        #self._validate_form_calculation_columns(df)
        
        # Ensure we have points data
        if 'home_points' not in df.columns:
            df['home_points'] = (df['home_goals'] > df['away_goals']).astype(int) * 3 + \
                            (df['home_goals'] == df['away_goals']).astype(int)
            df['away_points'] = (df['away_goals'] > df['home_goals']).astype(int) * 3 + \
                            (df['away_goals'] == df['home_goals']).astype(int)
        
        # Sort by league, season, team and date for proper calculations
        df = df.sort_values(['league_id', 'season', 'home_team_id', 'date'])
        
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            points_col = f'{prefix}_points'
            
            # Calculate form strength using EXPANDING mean within each league and season
            df[f'{prefix}_form_strength'] = df.groupby(['league_id', 'season', team_col])[points_col].transform(
                lambda x: (x.expanding().mean().shift(1) / 3 * 100).clip(0, 100).fillna(50)
            ).round(1)
            
            # Calculate momentum using a more robust approach
            df[f'{prefix}_momentum'] = df.groupby(['league_id', 'season', team_col]).apply(
                lambda group: self._calculate_momentum_fixed(group, points_col)
            ).reset_index(level=[0, 1, 2], drop=True)
            
            # Calculate consistency (std dev of points) within each league and season
            df[f'{prefix}_consistency'] = df.groupby(['league_id', 'season', team_col])[points_col].transform(
                lambda x: x.expanding().std().shift(1).fillna(0)
            ).round(2)
        
        # Validate the form calculations
        #self._validate_form_calculations(df)
        
        return df

    def _calculate_momentum_fixed(self, group, points_col):
        """
        Calculate momentum based on matches BEFORE current game within same league and season
        Fixed version to handle edge cases and prevent extreme values
        """
        momentum_values = []
        points_series = group[points_col]
        
        for i in range(len(group)):
            if i < 6:  # Need at least 6 previous matches for proper momentum
                momentum_values.append(0)
                continue
            
            # Get all points before the current match
            points_before = points_series.iloc[:i]
            
            if len(points_before) < 6:
                momentum_values.append(0)
                continue
            
            # Calculate last 3 and previous 3 matches
            last_3_points = points_before.iloc[-3:].sum()
            prev_3_points = points_before.iloc[-6:-3].sum()
            
            momentum = last_3_points - prev_3_points
            
            # Clip extreme values to reasonable range (-9 to +9)
            momentum = max(-9, min(9, momentum))
            momentum_values.append(momentum)
        
        return pd.Series(momentum_values, index=group.index)

    def _validate_form_calculations(self, df: pd.DataFrame):
        """Validate that form calculations are reasonable"""
        
        # Check for NaN values in calculated columns
        form_columns = ['home_form_strength', 'away_form_strength', 
                    'home_momentum', 'away_momentum',
                    'home_consistency', 'away_consistency']
        
        nan_counts = {}
        for col in form_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_counts[col] = nan_count
        
        if nan_counts:
            print(f"Warning: NaN values found in form calculations: {nan_counts}")
        
        # Validate form strength range (should be 0-100)
        for prefix in ['home', 'away']:
            col = f'{prefix}_form_strength'
            if col in df.columns:
                out_of_range = df[(df[col] < 0) | (df[col] > 100)]
                if not out_of_range.empty:
                    print(f"Warning: {len(out_of_range)} rows have {col} outside range 0-100")
        
        # Validate momentum range and diagnose issues
        for prefix in ['home', 'away']:
            col = f'{prefix}_momentum'
            if col in df.columns:
                # Check for extreme values
                extreme_momentum = df[df[col].abs() > 9]
                if not extreme_momentum.empty:
                    print(f"Warning: {len(extreme_momentum)} rows have extreme momentum values")
                    print(f"Momentum range: {df[col].min()} to {df[col].max()}")
                    
                    # Debug: Show some examples of extreme momentum
                    sample_extreme = extreme_momentum[[f'{prefix}_team_id', 'league_id', 'season', col]].head(5)
                    print(f"Sample extreme momentum cases:\n{sample_extreme}")
                
                # Check distribution
                momentum_stats = df[col].describe()
                print(f"{col} statistics:\n{momentum_stats}")

    # Additional debug function to help diagnose momentum issues
    def _debug_momentum_calculation(self, df: pd.DataFrame, team_id: int, league_id: int, season: str):
        """
        Debug function to see momentum calculation for a specific team in a specific league/season
        """
        team_matches = df[(df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)]
        team_matches = team_matches[(team_matches['league_id'] == league_id) & (team_matches['season'] == season)]
        team_matches = team_matches.sort_values('date')
        
        print(f"Debug momentum for team {team_id} in league {league_id}, season {season}")
        print(f"Total matches: {len(team_matches)}")
        
        for i, (idx, row) in enumerate(team_matches.iterrows()):
            if i >= 6:  # Only show where momentum is calculated
                is_home = row['home_team_id'] == team_id
                prefix = 'home' if is_home else 'away'
                momentum = row[f'{prefix}_momentum']
                
                if abs(momentum) > 9:
                    print(f"Match {i+1}: Extreme momentum {momentum}")
                    # You could add more detailed debugging here

    def _validate_form_calculation_columns(self, df: pd.DataFrame):
        """Validate that required columns exist for form calculations"""
        required_columns = ['league_id', 'season', 'home_team_id', 'away_team_id', 'date', 
                        'home_goals', 'away_goals']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for form calculation: {missing_columns}")
        
        # Check if we have enough data per league-season combination
        league_season_counts = df.groupby(['league_id', 'season']).size()
        if league_season_counts.min() < 10:  # Arbitrary minimum threshold
            print(f"Warning: Some league-season combinations have very few matches: {league_season_counts[league_season_counts < 10]}")



 

    def _validate_standings(self, df: pd.DataFrame) -> bool:
        """Validate that standings make sense by checking cumulative progression"""
        valid = True
        errors = []
        
        # Check each team's progression
        for (team_id, league_id, season), group in df.groupby(['home_team_id', 'league_id', 'season']):
            team_matches = group.sort_values('date')
            
            current_points = 0
            current_goals_for = 0
            current_goals_against = 0
            current_matches = 0
            
            for idx, match in team_matches.iterrows():
                pre_points = match['home_points']
                pre_gf = match['home_goals_for']
                pre_ga = match['home_goals_against']
                
                # Check if pre-match stats match our cumulative calculation
                if abs(pre_points - current_points) > 3:  # Allow for small rounding errors
                    errors.append(f"Points mismatch: Team {team_id} in {league_id}/{season} "
                                f"at fixture {match['fixture_id']}: "
                                f"Expected {current_points}, got {pre_points}")
                    valid = False
                
                if abs(pre_gf - current_goals_for) > 3:
                    errors.append(f"Goals for mismatch: Team {team_id} in {league_id}/{season} "
                                f"at fixture {match['fixture_id']}: "
                                f"Expected {current_goals_for}, got {pre_gf}")
                    valid = False
                
                if abs(pre_ga - current_goals_against) > 3:
                    errors.append(f"Goals against mismatch: Team {team_id} in {league_id}/{season} "
                                f"at fixture {match['fixture_id']}: "
                                f"Expected {current_goals_against}, got {pre_ga}")
                    valid = False
                
                # Update current stats based on match result
                if match['status'] in ['FT', 'AET', 'PEN']:
                    home_goals = int(match['home_goals'])
                    away_goals = int(match['away_goals'])
                    
                    current_goals_for += home_goals
                    current_goals_against += away_goals
                    
                    if home_goals > away_goals:
                        current_points += 3
                    elif home_goals == away_goals:
                        current_points += 1
                    # else: away wins, home gets 0 points
                
                current_matches += 1
        
        if errors:
            self.logger.error(f"Standings validation found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                self.logger.error(error)
        
        return valid


    def _create_h2h_features_2(self, df: pd.DataFrame, LEAGUES: dict, monitor_teams: list = None) -> pd.DataFrame:
        """Enhanced H2H calculator with proper goal calculation and competition types"""
        
        # Add team monitoring
        if monitor_teams:
            team_matches = {}
            for team in monitor_teams:
                team_matches[team] = {
                    'processed': 0,
                    'found_h2h': 0,
                    'no_h2h': 0
                }
        
        df = df.copy().sort_values('date')
        
        # Initialize H2H features
        h2h_features = [
            'h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
            'h2h_home_win_pct', 'h2h_away_win_pct', 'h2h_avg_goals',
            'h2h_recent_home_wins_last5', 'h2h_recent_away_wins_last5',
            'h2h_recent_draws_last5', 'h2h_recent_avg_goals_last5',
            'h2h_streak', 'h2h_league_matches', 'h2h_cup_matches',
            'h2h_cup_home_wins', 'h2h_cup_away_wins', 'h2h_same_country',
            'h2h_win_streak', 'h2h_loss_streak', 'h2h_europe_matches'
        ]
        
        for feature in h2h_features:
            df[feature] = 0
        
        completed_matches = df[df['status'].isin(['FT', 'AET', 'PEN'])].copy()
        upcoming_matches = df[df['status'] == 'NS'].copy()
        
        def calculate_match_stats(target_df, source_df, desc: str):
            """Calculate H2H stats using only matches before current date"""
            if len(target_df) == 0:
                self.logger.info(f"{desc}: No matches to process")
                return target_df
            
            target_df = target_df.reset_index(drop=True)
            result_df = target_df.copy()
            
            self.logger.info(f"{desc}: Processing {len(result_df)} matches...")
            start_time = time.time()
            
            # Convert source_df dates to datetime for proper comparison
            source_df = source_df.copy()
            if 'date' in source_df.columns:
                source_df['date'] = pd.to_datetime(source_df['date'], errors='coerce')
            
            for idx in range(len(result_df)):
                row = result_df.iloc[idx]
                home_team, away_team, match_date, country = row[['home_team_id', 'away_team_id', 'date', 'country']]
                
                # Use stored H2H data instead of recalculating from source_df
                h2h_key = frozenset({int(home_team), int(away_team)})
                past_matches_data = self.h2h_data.get(h2h_key, [])
                
                # Convert to DataFrame for easier processing
                past_matches = pd.DataFrame(past_matches_data)
                
                if not past_matches.empty:
                    # Convert dates to datetime and filter matches that occurred before current match date
                    past_matches['date'] = pd.to_datetime(past_matches['date'], errors='coerce')
                    past_matches = past_matches[past_matches['date'] < match_date]
                
                # Monitor specific teams
                if monitor_teams:
                    for team in monitor_teams:
                        if team in [home_team, away_team]:
                            team_matches[team]['processed'] += 1
                
                # Find past matches BEFORE current match date
                past_matches = source_df[
                    (((source_df['home_team_id'] == home_team) & (source_df['away_team_id'] == away_team)) |
                    ((source_df['home_team_id'] == away_team) & (source_df['away_team_id'] == home_team))) &
                    (source_df['date'] < match_date)  # ONLY matches before current match
                ].copy()
                
                # Filter for same country matches only for league and cup stats
                same_country_matches = past_matches[past_matches['country'] == country].copy()
                result_df.iloc[idx, result_df.columns.get_loc('h2h_same_country')] = 1 if not same_country_matches.empty else 0
                
                if not past_matches.empty:
                    # Calculate overall stats
                    total_matches = len(past_matches)
                    
                    # Calculate competition-specific matches
                    cup_matches = past_matches[past_matches['is_cup_competition'] == 1].copy()
                    europe_matches = past_matches[past_matches['is_europe_competition'] == 1].copy()
                    league_matches = past_matches[
                        (past_matches['is_cup_competition'] != 1) & 
                        (past_matches['is_europe_competition'] != 1) &
                        (past_matches['country'] == country)
                    ].copy()
                    
                    # Store competition counts
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_league_matches')] = len(league_matches)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_matches')] = len(cup_matches)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_europe_matches')] = len(europe_matches)
                    
                    # Wins from home team's perspective
                    home_wins = 0
                    away_wins = 0
                    home_goals_scored = 0
                    away_goals_scored = 0
                    
                    # Cup-specific wins
                    cup_home_wins = 0
                    cup_away_wins = 0
                    
                    for _, past_match in past_matches.iterrows():
                        # Determine if home team was home or away in past match
                        if past_match['home_team_id'] == home_team:
                            # Home team was home in past match
                            home_goals = past_match['home_goals']
                            away_goals = past_match['away_goals']
                            if home_goals > away_goals:
                                home_wins += 1
                                # Check if this is a cup match
                                if past_match['is_cup_competition'] == 1:
                                    cup_home_wins += 1
                            elif away_goals > home_goals:
                                away_wins += 1
                        else:
                            # Home team was away in past match
                            home_goals = past_match['away_goals']  # Home team's goals
                            away_goals = past_match['home_goals']  # Away team's goals
                            if home_goals > away_goals:
                                home_wins += 1
                                # Check if this is a cup match
                                if past_match['is_cup_competition'] == 1:
                                    cup_home_wins += 1
                            elif away_goals > home_goals:
                                away_wins += 1
                                # Check if this is a cup match
                                if past_match['is_cup_competition'] == 1:
                                    cup_away_wins += 1
                        
                        home_goals_scored += home_goals
                        away_goals_scored += away_goals
                    
                    draws = total_matches - home_wins - away_wins
                    goal_diff = home_goals_scored - away_goals_scored
                    
                    # Store all calculated values
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_matches')] = total_matches
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_home_wins')] = home_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_away_wins')] = away_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_draws')] = draws
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_home_goals')] = home_goals_scored
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_away_goals')] = away_goals_scored
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_goal_diff')] = goal_diff
                    
                    # Store cup-specific wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_home_wins')] = cup_home_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_away_wins')] = cup_away_wins
                    
                    # Calculate percentages and averages
                    if total_matches > 0:
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_home_win_pct')] = home_wins / total_matches
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_away_win_pct')] = away_wins / total_matches
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_avg_goals')] = (home_goals_scored + away_goals_scored) / total_matches
                    
                    # Calculate streak
                    streak = self._calculate_streak(past_matches, home_team, 
                                                debug=(monitor_teams and home_team in monitor_teams))
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_streak')] = streak
                    
                    # Also calculate win/loss streaks
                    wins, losses = self._calculate_win_loss_streaks(past_matches, home_team)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_win_streak')] = wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_loss_streak')] = losses
                    
                    # Monitor H2H found
                    if monitor_teams:
                        for team in monitor_teams:
                            if team in [home_team, away_team]:
                                team_matches[team]['found_h2h'] += 1
                    
                    # Debug logging for first few matches
                    if idx < 3:
                        self.logger.debug(f"Match {idx}: {home_team} vs {away_team}")
                        self.logger.debug(f"  Found {total_matches} past matches")
                        self.logger.debug(f"  League: {len(league_matches)}, Cup: {len(cup_matches)}, Europe: {len(europe_matches)}")
                        self.logger.debug(f"  Home wins: {home_wins}, Away wins: {away_wins}, Draws: {draws}")
                        self.logger.debug(f"  Goals: {home_goals_scored}-{away_goals_scored}")
                
                else:
                    # No past matches found
                    if monitor_teams:
                        for team in monitor_teams:
                            if team in [home_team, away_team]:
                                team_matches[team]['no_h2h'] += 1
                
                # Progress logging
                if (idx + 1) % max(1, len(result_df) // 10) == 0 or (idx + 1) == len(result_df):
                    progress = ((idx + 1) / len(result_df)) * 100
                    elapsed = time.time() - start_time
                    self.logger.info(f"H2H Progress: {progress:.0f}% ({idx + 1}/{len(result_df)}) - {elapsed:.1f}s")
            
            return result_df
        
        # Process matches
        completed_matches = calculate_match_stats(completed_matches, completed_matches, "Completed")
        upcoming_matches = calculate_match_stats(upcoming_matches, completed_matches, "Upcoming")
        
        # Combine results
        result_df = pd.concat([completed_matches, upcoming_matches]).sort_values('date')
        

        
        # Validate results
        #self._validate_h2h_results(result_df)
        #self._validate_h2h_batch(result_df, sample_size=20)
        #self._validate_streak_calculation(result_df,  492, 489, "2024-03-15")  # Re-validate after batch checks
    
        
        return result_df 
 
    def _create_h2h_features(self, df: pd.DataFrame, LEAGUES: dict, monitor_teams: list = None) -> pd.DataFrame:
        """Enhanced H2H calculator with proper goal calculation and competition types"""
        
        # Add team monitoring
        if monitor_teams:
            team_matches = {}
            for team in monitor_teams:
                team_matches[team] = {
                    'processed': 0,
                    'found_h2h': 0,
                    'no_h2h': 0
                }
        
        df = df.copy().sort_values('date')
        
        # Initialize H2H features
        h2h_features = [
            'h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
            'h2h_home_win_pct', 'h2h_away_win_pct', 'h2h_avg_goals',
            'h2h_recent_home_wins_last5', 'h2h_recent_away_wins_last5',
            'h2h_recent_draws_last5', 'h2h_recent_avg_goals_last5',
            'h2h_streak', 'h2h_league_matches', 'h2h_cup_matches',
            'h2h_cup_home_wins', 'h2h_cup_away_wins', 'h2h_same_country',
            'h2h_win_streak', 'h2h_loss_streak', 'h2h_europe_matches'
        ]
        
        for feature in h2h_features:
            df[feature] = 0
        
        completed_matches = df[df['status'].isin(['FT', 'AET', 'PEN'])].copy()
        upcoming_matches = df[df['status'] == 'NS'].copy()
        
        def calculate_match_stats(target_df, source_df, desc: str):
            """Calculate H2H stats using only matches before current date"""
            if len(target_df) == 0:
                self.logger.info(f"{desc}: No matches to process")
                return target_df
            
            target_df = target_df.reset_index(drop=True)
            result_df = target_df.copy()
            
            self.logger.info(f"{desc}: Processing {len(result_df)} matches...")
            start_time = time.time()
            
            # Progress tracking variables
            progress_interval = 10  # percentage
            last_reported_progress = -progress_interval
            
            # Convert source_df dates to datetime for proper comparison
            source_df = source_df.copy()
            if 'date' in source_df.columns:
                source_df['date'] = pd.to_datetime(source_df['date'], errors='coerce')
            
            for idx in range(len(result_df)):
                row = result_df.iloc[idx]
                home_team, away_team, match_date, country = row[['home_team_id', 'away_team_id', 'date', 'country']]
                
                # Use stored H2H data instead of recalculating from source_df
                h2h_key = frozenset({int(home_team), int(away_team)})
                past_matches_data = self.h2h_data.get(h2h_key, [])
                
                # Convert to DataFrame for easier processing
                past_matches = pd.DataFrame(past_matches_data)
                
                if not past_matches.empty:
                    # Convert dates to datetime and filter matches that occurred before current match date
                    past_matches['date'] = pd.to_datetime(past_matches['date'], errors='coerce')
                    past_matches = past_matches[past_matches['date'] < match_date]
                
                # Monitor specific teams
                if monitor_teams:
                    for team in monitor_teams:
                        if team in [home_team, away_team]:
                            team_matches[team]['processed'] += 1
                
                # Find past matches BEFORE current match date
                past_matches = source_df[
                    (((source_df['home_team_id'] == home_team) & (source_df['away_team_id'] == away_team)) |
                    ((source_df['home_team_id'] == away_team) & (source_df['away_team_id'] == home_team))) &
                    (source_df['date'] < match_date)  # ONLY matches before current match
                ].copy()
                
                # Filter for same country matches only for league and cup stats
                same_country_matches = past_matches[past_matches['country'] == country].copy()
                result_df.iloc[idx, result_df.columns.get_loc('h2h_same_country')] = 1 if not same_country_matches.empty else 0
                
                if not past_matches.empty:
                    # Calculate overall stats
                    total_matches = len(past_matches)
                    
                    # Calculate competition-specific matches
                    cup_matches = past_matches[past_matches['is_cup_competition'] == 1].copy()
                    europe_matches = past_matches[past_matches['is_europe_competition'] == 1].copy()
                    league_matches = past_matches[
                        (past_matches['is_cup_competition'] != 1) & 
                        (past_matches['is_europe_competition'] != 1) &
                        (past_matches['country'] == country)
                    ].copy()
                    
                    # Store competition counts
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_league_matches')] = len(league_matches)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_matches')] = len(cup_matches)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_europe_matches')] = len(europe_matches)
                    
                    # Wins from home team's perspective
                    home_wins = 0
                    away_wins = 0
                    home_goals_scored = 0
                    away_goals_scored = 0
                    
                    # Cup-specific wins
                    cup_home_wins = 0
                    cup_away_wins = 0
                    
                    for _, past_match in past_matches.iterrows():
                        # Determine if home team was home or away in past match
                        if past_match['home_team_id'] == home_team:
                            # Home team was home in past match
                            home_goals = past_match['home_goals']
                            away_goals = past_match['away_goals']
                            if home_goals > away_goals:
                                home_wins += 1
                                # Check if this is a cup match
                                if past_match['is_cup_competition'] == 1:
                                    cup_home_wins += 1
                            elif away_goals > home_goals:
                                away_wins += 1
                        else:
                            # Home team was away in past match
                            home_goals = past_match['away_goals']  # Home team's goals
                            away_goals = past_match['home_goals']  # Away team's goals
                            if home_goals > away_goals:
                                home_wins += 1
                                # Check if this is a cup match
                                if past_match['is_cup_competition'] == 1:
                                    cup_home_wins += 1
                            elif away_goals > home_goals:
                                away_wins += 1
                                # Check if this is a cup match
                                if past_match['is_cup_competition'] == 1:
                                    cup_away_wins += 1
                        
                        home_goals_scored += home_goals
                        away_goals_scored += away_goals
                    
                    draws = total_matches - home_wins - away_wins
                    goal_diff = home_goals_scored - away_goals_scored
                    
                    # Store all calculated values
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_matches')] = total_matches
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_home_wins')] = home_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_away_wins')] = away_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_draws')] = draws
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_home_goals')] = home_goals_scored
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_away_goals')] = away_goals_scored
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_goal_diff')] = goal_diff
                    
                    # Store cup-specific wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_home_wins')] = cup_home_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_away_wins')] = cup_away_wins
                    
                    # Calculate percentages and averages
                    if total_matches > 0:
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_home_win_pct')] = home_wins / total_matches
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_away_win_pct')] = away_wins / total_matches
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_avg_goals')] = (home_goals_scored + away_goals_scored) / total_matches
                    
                    # Calculate streak
                    streak = self._calculate_streak(past_matches, home_team, 
                                                debug=(monitor_teams and home_team in monitor_teams))
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_streak')] = streak
                    
                    # Also calculate win/loss streaks
                    wins, losses = self._calculate_win_loss_streaks(past_matches, home_team)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_win_streak')] = wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_loss_streak')] = losses
                    
                    # Monitor H2H found
                    if monitor_teams:
                        for team in monitor_teams:
                            if team in [home_team, away_team]:
                                team_matches[team]['found_h2h'] += 1
                    
                    # Debug logging for first few matches
                    if idx < 3:
                        self.logger.debug(f"Match {idx}: {home_team} vs {away_team}")
                        self.logger.debug(f"  Found {total_matches} past matches")
                        self.logger.debug(f"  League: {len(league_matches)}, Cup: {len(cup_matches)}, Europe: {len(europe_matches)}")
                        self.logger.debug(f"  Home wins: {home_wins}, Away wins: {away_wins}, Draws: {draws}")
                        self.logger.debug(f"  Goals: {home_goals_scored}-{away_goals_scored}")
                
                else:
                    # No past matches found
                    if monitor_teams:
                        for team in monitor_teams:
                            if team in [home_team, away_team]:
                                team_matches[team]['no_h2h'] += 1
                
                # Progress logging every 10%
                current_progress = int(((idx + 1) / len(result_df)) * 100)
                if current_progress >= last_reported_progress + progress_interval or (idx + 1) == len(result_df):
                    elapsed = time.time() - start_time
                    self.logger.info(f"H2H {desc} Progress: {current_progress}% ({idx + 1}/{len(result_df)}) - {elapsed:.1f}s")
                    last_reported_progress = current_progress
            
            return result_df
        
        # Process matches
        completed_matches = calculate_match_stats(completed_matches, completed_matches, "Completed")
        upcoming_matches = calculate_match_stats(upcoming_matches, completed_matches, "Upcoming")
        
        # Combine results
        result_df = pd.concat([completed_matches, upcoming_matches]).sort_values('date')
        
        # Validate results
        #self._validate_h2h_results(result_df)
        #self._validate_h2h_batch(result_df, sample_size=20)
        #self._validate_streak_calculation(result_df,  492, 489, "2024-03-15")  # Re-validate after batch checks

        return result_df  

    def _create_h2h_features_3(self, df: pd.DataFrame, LEAGUES: dict, monitor_teams: list = None) -> pd.DataFrame:
        """Enhanced H2H calculator with proper goal calculation and competition types"""
        
        # Add team monitoring
        if monitor_teams:
            team_matches = {}
            for team in monitor_teams:
                team_matches[team] = {
                    'processed': 0,
                    'found_h2h': 0,
                    'no_h2h': 0
                }
        
        df = df.copy().sort_values('date')
        
        # Initialize H2H features
        h2h_features = [
            'h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
            'h2h_home_win_pct', 'h2h_away_win_pct', 'h2h_avg_goals',
            'h2h_recent_home_wins_last5', 'h2h_recent_away_wins_last5',
            'h2h_recent_draws_last5', 'h2h_recent_avg_goals_last5',
            'h2h_streak', 'h2h_league_matches', 'h2h_cup_matches',
            'h2h_cup_home_wins', 'h2h_cup_away_wins',
            'h2h_win_streak', 'h2h_loss_streak', 'h2h_europe_matches'
        ]
        
        for feature in h2h_features:
            df[feature] = 0
        
        completed_matches = df[df['status'].isin(['FT', 'AET', 'PEN'])].copy()
        upcoming_matches = df[df['status'] == 'NS'].copy()
        
        def calculate_match_stats(target_df, source_df, desc: str):
            """Calculate H2H stats using only matches before current date"""
            if len(target_df) == 0:
                self.logger.info(f"{desc}: No matches to process")
                return target_df
            
            target_df = target_df.reset_index(drop=True)
            result_df = target_df.copy()
            
            self.logger.info(f"{desc}: Processing {len(result_df)} matches...")
            start_time = time.time()
            
            for idx in range(len(result_df)):
                row = result_df.iloc[idx]
                home_team, away_team, match_date, country = row[['home_team_id', 'away_team_id', 'date', 'country']]
                
                # Monitor specific teams
                if monitor_teams:
                    for team in monitor_teams:
                        if team in [home_team, away_team]:
                            team_matches[team]['processed'] += 1
                
                # Find past matches BEFORE current match date
                past_matches = source_df[
                    (((source_df['home_team_id'] == home_team) & (source_df['away_team_id'] == away_team)) |
                    ((source_df['home_team_id'] == away_team) & (source_df['away_team_id'] == home_team))) &
                    (source_df['date'] < match_date)  # ONLY matches before current match
                ].copy()
                
                # Filter for same country matches only for league and cup stats
                same_country_matches = past_matches[past_matches['country'] == country].copy()
                result_df.iloc[idx, result_df.columns.get_loc('h2h_same_country')] = 1 if not same_country_matches.empty else 0
                
                if not past_matches.empty:
                    # Calculate overall stats
                    total_matches = len(past_matches)
                    
                    # Calculate competition-specific matches
                    cup_matches = past_matches[past_matches['cup_competition'] == 1].copy()
                    europe_matches = past_matches[past_matches['europe_competition'] == 1].copy()
                    league_matches = past_matches[
                        (past_matches['cup_competition'] != 1) & 
                        (past_matches['europe_competition'] != 1) &
                        (past_matches['country'] == country)
                    ].copy()
                    
                    # Store competition counts
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_league_matches')] = len(league_matches)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_matches')] = len(cup_matches)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_europe_matches')] = len(europe_matches)
                    
                    # Wins from home team's perspective
                    home_wins = 0
                    away_wins = 0
                    home_goals_scored = 0
                    away_goals_scored = 0
                    
                    # Cup-specific wins
                    cup_home_wins = 0
                    cup_away_wins = 0
                    
                    for _, past_match in past_matches.iterrows():
                        # Determine if home team was home or away in past match
                        if past_match['home_team_id'] == home_team:
                            # Home team was home in past match
                            home_goals = past_match['home_goals']
                            away_goals = past_match['away_goals']
                            if home_goals > away_goals:
                                home_wins += 1
                                # Check if this is a cup match
                                if past_match['cup_competition'] == 1:
                                    cup_home_wins += 1
                            elif away_goals > home_goals:
                                away_wins += 1
                        else:
                            # Home team was away in past match
                            home_goals = past_match['away_goals']  # Home team's goals
                            away_goals = past_match['home_goals']  # Away team's goals
                            if home_goals > away_goals:
                                home_wins += 1
                                # Check if this is a cup match
                                if past_match['cup_competition'] == 1:
                                    cup_home_wins += 1
                            elif away_goals > home_goals:
                                away_wins += 1
                                # Check if this is a cup match
                                if past_match['cup_competition'] == 1:
                                    cup_away_wins += 1
                        
                        home_goals_scored += home_goals
                        away_goals_scored += away_goals
                    
                    draws = total_matches - home_wins - away_wins
                    goal_diff = home_goals_scored - away_goals_scored
                    
                    # Store all calculated values
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_matches')] = total_matches
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_home_wins')] = home_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_away_wins')] = away_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_draws')] = draws
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_home_goals')] = home_goals_scored
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_away_goals')] = away_goals_scored
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_goal_diff')] = goal_diff
                    
                    # Store cup-specific wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_home_wins')] = cup_home_wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_cup_away_wins')] = cup_away_wins
                    
                    # Calculate percentages and averages
                    if total_matches > 0:
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_home_win_pct')] = home_wins / total_matches
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_away_win_pct')] = away_wins / total_matches
                        result_df.iloc[idx, result_df.columns.get_loc('h2h_avg_goals')] = (home_goals_scored + away_goals_scored) / total_matches
                    
                    # Calculate streak
                    streak = self._calculate_streak(past_matches, home_team, 
                                                debug=(monitor_teams and home_team in monitor_teams))
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_streak')] = streak
                    
                    # Also calculate win/loss streaks
                    wins, losses = self._calculate_win_loss_streaks(past_matches, home_team)
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_win_streak')] = wins
                    result_df.iloc[idx, result_df.columns.get_loc('h2h_loss_streak')] = losses
                    
                    # Monitor H2H found
                    if monitor_teams:
                        for team in monitor_teams:
                            if team in [home_team, away_team]:
                                team_matches[team]['found_h2h'] += 1
                    
                    # Debug logging for first few matches
                    if idx < 3:
                        self.logger.debug(f"Match {idx}: {home_team} vs {away_team}")
                        self.logger.debug(f"  Found {total_matches} past matches")
                        self.logger.debug(f"  League: {len(league_matches)}, Cup: {len(cup_matches)}, Europe: {len(europe_matches)}")
                        self.logger.debug(f"  Home wins: {home_wins}, Away wins: {away_wins}, Draws: {draws}")
                        self.logger.debug(f"  Goals: {home_goals_scored}-{away_goals_scored}")
                
                else:
                    # No past matches found
                    if monitor_teams:
                        for team in monitor_teams:
                            if team in [home_team, away_team]:
                                team_matches[team]['no_h2h'] += 1
                
                # Progress logging
                if (idx + 1) % max(1, len(result_df) // 10) == 0 or (idx + 1) == len(result_df):
                    progress = ((idx + 1) / len(result_df)) * 100
                    elapsed = time.time() - start_time
                    self.logger.info(f"Progress: {progress:.0f}% ({idx + 1}/{len(result_df)}) - {elapsed:.1f}s")
            
                # Add streak validation
            self._validate_streaks_batch(result_df, sample_size=10)
            
            return result_df
        
        # Process matches
        completed_matches = calculate_match_stats(completed_matches, completed_matches, "Completed")
        upcoming_matches = calculate_match_stats(upcoming_matches, completed_matches, "Upcoming")
        
        # Combine results
        result_df = pd.concat([completed_matches, upcoming_matches]).sort_values('date')
        
        # Log team monitoring results
        if monitor_teams:
            self.logger.info("=== TEAM MONITORING RESULTS ===")
            for team, stats in team_matches.items():
                self.logger.info(f"Team {team}: {stats['processed']} matches, "
                            f"{stats['found_h2H']} with H2H, "
                            f"{stats['no_h2h']} without H2H")
        
        # Validate results
        self._validate_h2h_results(result_df)
        self._validate_h2h_batch(result_df, sample_size=20)
        self._validate_streak_calculation(result_df,  492, 489, "2024-03-15")  # Re-validate after batch checks
    
        
        return result_df

    def _validate_h2h_results(self, df):
        """Fix validation logic - the current validation is wrong!"""
        self.logger.info("=== H2H VALIDATION ===")
        
        # CORRECTED: Check for matches with H2H data but BOTH home AND away goals are zero
        # This is different from your current logic which checks OR condition
        no_goals_issue = df[(df['h2h_matches'] > 0) & 
                        (df['h2h_home_goals'] == 0) &  # AND both are zero
                        (df['h2h_away_goals'] == 0)]   # AND both are zero
        
        if len(no_goals_issue) > 0:
            self.logger.warning(f"Found {len(no_goals_issue)} matches with H2H but ZERO goals recorded")
            for idx, row in no_goals_issue.head(3).iterrows():
                self.logger.warning(f"  Match {idx}: {row['h2h_matches']} matches, "
                                f"Goals: {row['h2h_home_goals']}-{row['h2h_away_goals']}")
        else:
            self.logger.info("No matches found with H2H data and zero goals - GOOD!")
        
        # Additional validation: Check for logical consistency
        logical_issues = []
        for idx, row in df.iterrows():
            if row['h2h_matches'] > 0:
                # Check if wins + draws = total matches
                if row['h2h_home_wins'] + row['h2h_away_wins'] + row['h2h_draws'] != row['h2h_matches']:
                    logical_issues.append(f"Match {idx}: Wins+draws ({row['h2h_home_wins']+row['h2h_away_wins']+row['h2h_draws']}) ≠ total ({row['h2h_matches']})")
                
                # Check if goals make sense relative to wins
                if row['h2h_home_goals'] == 0 and row['h2h_home_wins'] > 0:
                    logical_issues.append(f"Match {idx}: {row['h2h_home_wins']} home wins but 0 home goals")
                if row['h2h_away_goals'] == 0 and row['h2h_away_wins'] > 0:
                    logical_issues.append(f"Match {idx}: {row['h2h_away_wins']} away wins but 0 away goals")
        
        if logical_issues:
            self.logger.warning(f"Found {len(logical_issues)} logical inconsistencies")
            for issue in logical_issues[:3]:
                self.logger.warning(f"  {issue}")
        
        # Check feature summary
        self.logger.info("H2H features summary:")
        features_to_check = ['h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 
                            'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff']
        
        for feature in features_to_check:
            non_zero = (df[feature] != 0).sum()
            total_with_h2h = (df['h2h_matches'] > 0).sum()
            self.logger.info(f"  {feature}: {non_zero}/{total_with_h2h} non-zero values")

    def _validate_h2h_batch(self, df, sample_size=10):
        """Validate multiple random matches"""
        self.logger.info("=== BATCH VALIDATION ===")
        
        # Get matches with H2H data
        h2h_matches = df[df['h2h_matches'] > 0]
        
        if len(h2h_matches) == 0:
            self.logger.warning("No matches with H2H data to validate")
            return
        
        # Sample random matches
        sample_matches = h2h_matches.sample(min(sample_size, len(h2h_matches)))
        
        correct_count = 0
        total_checked = 0
        
        for idx, match_row in sample_matches.iterrows():
            home_team = match_row['home_team_id']
            away_team = match_row['away_team_id']
            match_date = match_row['date']
            
            # Get historical matches manually
            historical_matches = df[
                (((df['home_team_id'] == home_team) & (df['away_team_id'] == away_team)) |
                ((df['home_team_id'] == away_team) & (df['away_team_id'] == home_team))) &
                (df['date'] < match_date) &
                (df['status'].isin(['FT', 'AET', 'PEN']))
            ]
            
            # Manual calculation
            manual_home_wins = 0
            manual_away_wins = 0
            manual_home_goals = 0
            manual_away_goals = 0
            
            for _, hist_match in historical_matches.iterrows():
                if hist_match['home_team_id'] == home_team:
                    home_goals = hist_match['home_goals']
                    away_goals = hist_match['away_goals']
                else:
                    home_goals = hist_match['away_goals']
                    away_goals = hist_match['home_goals']
                
                if home_goals > away_goals:
                    manual_home_wins += 1
                elif away_goals > home_goals:
                    manual_away_wins += 1
                    
                manual_home_goals += home_goals
                manual_away_goals += away_goals
            
            manual_draws = len(historical_matches) - manual_home_wins - manual_away_wins
            
            # Compare
            is_correct = (
                len(historical_matches) == match_row['h2h_matches'] and
                manual_home_wins == match_row['h2h_home_wins'] and
                manual_away_wins == match_row['h2h_away_wins'] and
                manual_draws == match_row['h2h_draws'] and
                manual_home_goals == match_row['h2h_home_goals'] and
                manual_away_goals == match_row['h2h_away_goals']
            )
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
                
            total_checked += 1
            
            self.logger.info(f"{status} {home_team} vs {away_team}: "
                        f"Manual {manual_home_wins}-{manual_away_wins}-{manual_draws} "
                        f"vs Calculated {match_row['h2h_home_wins']}-{match_row['h2h_away_wins']}-{match_row['h2h_draws']}")
        
        accuracy = correct_count / total_checked * 100
        self.logger.info(f"=== BATCH VALIDATION COMPLETE ===")
        self.logger.info(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_checked} correct)")
        
        return accuracy

    def debug_team_h2h(self, team1_id: int, team2_id: int, df: pd.DataFrame):
        """Debug H2H between two specific teams"""
        self.logger.info(f"=== DEBUG H2H: {team1_id} vs {team2_id} ===")
        
        # Get all matches between these teams
        h2h_matches = df[
            (((df['home_team_id'] == team1_id) & (df['away_team_id'] == team2_id)) |
            ((df['home_team_id'] == team2_id) & (df['away_team_id'] == team1_id))) &
            (df['status'].isin(['FT', 'AET', 'PEN']))
        ].sort_values('date')
        
        self.logger.info(f"Found {len(h2h_matches)} completed matches:")
        
        for idx, match in h2h_matches.iterrows():
            home_team = match['home_team_id']
            away_team = match['away_team_id']
            result = f"{match['home_goals']}-{match['away_goals']}"
            
            if home_team == team1_id:
                perspective = "Home"
                win = match['home_goals'] > match['away_goals']
            else:
                perspective = "Away" 
                win = match['away_goals'] > match['home_goals']
            
            self.logger.info(f"  {match['date']}: {home_team} {result} {away_team} ({perspective} for {team1_id}, Win: {win})")
        
        # Calculate manual H2H stats
        if not h2h_matches.empty:
            home_wins = len(h2h_matches[
                (h2h_matches['home_team_id'] == team1_id) & 
                (h2h_matches['home_goals'] > h2h_matches['away_goals'])
            ])
            
            away_wins = len(h2h_matches[
                (h2h_matches['away_team_id'] == team1_id) & 
                (h2h_matches['away_goals'] > h2h_matches['home_goals'])
            ])
            
            total = len(h2h_matches)
            draws = total - home_wins - away_wins
            
            self.logger.info(f"H2H Summary for {team1_id}:")
            self.logger.info(f"  Total: {total}")
            self.logger.info(f"  Wins: {home_wins + away_wins}")
            self.logger.info(f"  Losses: {total - (home_wins + away_wins + draws)}")
            self.logger.info(f"  Draws: {draws}")
            self.logger.info(f"  Win %: {(home_wins + away_wins) / total * 100:.1f}%")

    # Usage:
    # self.debug_team_h2h(123, 456, your_dataframe)

    def _calculate_streak(self, matches, home_team, debug=False):
        """
        Enhanced streak calculation with validation and debugging
        Returns float value between -1 (terrible streak) and 1 (excellent streak)
        """
        if matches.empty:
            return 0.0
        
        if debug:
            self.logger.info(f"=== STREAK CALCULATION DEBUG: Team {home_team} ===")
            self.logger.info(f"Processing {len(matches)} matches")
        
        streak_score = 0.0
        total_weight = 0.0
        streak_direction = None
        consecutive_count = 0
        streak_matches = []  # Store match details for debugging
        
        # Sort by date descending (most recent first)
        sorted_matches = matches.sort_values('date', ascending=False)
        
        for i, (_, match) in enumerate(sorted_matches.iterrows()):
            # Determine perspective
            is_home = match['home_team_id'] == home_team
            goals_for = match['home_goals'] if is_home else match['away_goals']
            goals_against = match['away_goals'] if is_home else match['home_goals']
            
            # Skip matches with NaN goals
            if pd.isna(goals_for) or pd.isna(goals_against):
                if debug:
                    self.logger.warning(f"  Match {i}: Skipped - NaN goals")
                continue
                
            weight = 0.8 ** i  # Exponential decay
            
            # Determine result
            try:
                if goals_for > goals_against:  # Win
                    current_direction = 1
                    result_score = 1.0
                    result_type = "WIN"
                elif goals_for < goals_against:  # Loss
                    current_direction = -1
                    result_score = 0.0
                    result_type = "LOSS"
                else:  # Draw
                    current_direction = 0
                    result_score = 0.5
                    result_type = "DRAW"
            except TypeError:
                if debug:
                    self.logger.warning(f"  Match {i}: Skipped - Type error in goals")
                continue
                
            # Calculate goal difference factor
            try:
                goal_diff = goals_for - goals_against
                gd_factor = min(3, max(-3, goal_diff)) / 12
            except TypeError:
                gd_factor = 0
            
            # Track consecutive results
            if streak_direction is None:
                streak_direction = current_direction
                consecutive_count = 1
                streak_broken = False
            elif current_direction == streak_direction:
                consecutive_count += 1
                streak_broken = False
            else:
                streak_broken = True
                if debug:
                    self.logger.info(f"  Match {i}: Streak broken! Expected {streak_direction}, got {current_direction}")
                break  # Streak broken
                
            # Calculate weighted contribution
            match_contribution = (result_score + gd_factor) * weight
            streak_score += match_contribution
            total_weight += weight
            
            # Store match details for debugging
            match_info = {
                'date': match['date'],
                'opponent': match['away_team_id'] if is_home else match['home_team_id'],
                'score': f"{goals_for}-{goals_against}",
                'result': result_type,
                'direction': current_direction,
                'weight': weight,
                'contribution': match_contribution,
                'consecutive': consecutive_count
            }
            streak_matches.append(match_info)
            
            if debug:
                self.logger.info(f"  Match {i}: {match['date']} vs {match_info['opponent']} "
                            f"{match_info['score']} ({result_type}) - "
                            f"Dir: {current_direction}, Weight: {weight:.3f}, "
                            f"Contrib: {match_contribution:.3f}, Consec: {consecutive_count}")
        
        if total_weight == 0:
            if debug:
                self.logger.info("No valid matches for streak calculation")
            return 0.0
            
        # Normalize and apply streak length bonus
        normalized_streak = streak_score / total_weight
        streak_bonus = min(0.2, consecutive_count * 0.05)
        
        if streak_direction == 1:  # Winning streak
            final_score = min(1.0, normalized_streak + streak_bonus)
            streak_type = "WINNING"
        elif streak_direction == -1:  # Losing streak
            final_score = max(-1.0, normalized_streak - streak_bonus)
            streak_type = "LOSING"
        else:  # Drawing streak
            final_score = normalized_streak
            streak_type = "DRAWING"
        
        final_score = round(final_score, 4)
        
        if debug:
            self.logger.info(f"=== STREAK RESULTS ===")
            self.logger.info(f"Total weight: {total_weight:.3f}")
            self.logger.info(f"Raw streak score: {streak_score:.3f}")
            self.logger.info(f"Normalized: {normalized_streak:.3f}")
            self.logger.info(f"Consecutive matches: {consecutive_count}")
            self.logger.info(f"Streak bonus: {streak_bonus:.3f}")
            self.logger.info(f"Final score: {final_score} ({streak_type} streak)")
            self.logger.info(f"Matches in streak: {len(streak_matches)}")
        
        return final_score


    def _calculate_win_loss_streaks(self, matches, home_team):
        """Calculate simple win/loss streaks"""
        if matches.empty:
            return 0, 0
        
        sorted_matches = matches.sort_values('date', ascending=False)
        win_streak = 0
        loss_streak = 0
        
        for _, match in sorted_matches.iterrows():
            is_home = match['home_team_id'] == home_team
            goals_for = match['home_goals'] if is_home else match['away_goals']
            goals_against = match['away_goals'] if is_home else match['home_goals']
            
            if pd.isna(goals_for) or pd.isna(goals_against):
                continue
                
            if goals_for > goals_against:
                win_streak += 1
                loss_streak = 0
            elif goals_for < goals_against:
                loss_streak += 1
                win_streak = 0
            else:
                break  # Draw breaks both streaks
        
        return win_streak, loss_streak

    def _validate_streak_calculation(self, df, home_team_id, away_team_id, match_date):
        """Validate streak calculation for a specific match"""
        self.logger.info(f"=== STREAK VALIDATION: {home_team_id} vs {away_team_id} on {match_date} ===")
        
        # Find the specific match
        match = df[(df['home_team_id'] == home_team_id) & 
                (df['away_team_id'] == away_team_id) & 
                (df['date'] == match_date)]
        
        if match.empty:
            self.logger.error("Match not found!")
            return
        
        match_row = match.iloc[0]
        calculated_streak = match_row['h2h_streak']
        
        # Get all historical matches BEFORE this date
        historical_matches = df[
            (((df['home_team_id'] == home_team_id) & (df['away_team_id'] == away_team_id)) |
            ((df['home_team_id'] == away_team_id) & (df['away_team_id'] == home_team_id))) &
            (df['date'] < match_date) &
            (df['status'].isin(['FT', 'AET', 'PEN']))
        ]
        
        # Manually calculate streak with debugging
        manual_streak = self._calculate_streak(historical_matches, home_team_id, debug=True)
        
        self.logger.info(f"=== STREAK COMPARISON ===")
        self.logger.info(f"Calculated streak: {calculated_streak}")
        self.logger.info(f"Manual streak: {manual_streak}")
        
        # Check if they match (allow for small floating point differences)
        if abs(calculated_streak - manual_streak) < 0.01:
            self.logger.info("✅ STREAK CALCULATION CORRECT!")
        else:
            self.logger.error(f"❌ STREAK CALCULATION ERROR! Difference: {abs(calculated_streak - manual_streak):.4f}")
        
        return abs(calculated_streak - manual_streak) < 0.01
    
    def _validate_streaks_batch(self, df, sample_size=5):
        """Validate streak calculations for multiple matches"""
        self.logger.info("=== BATCH STREAK VALIDATION ===")
        
        # Get matches with H2H data
        h2h_matches = df[df['h2h_matches'] > 0]
        
        if len(h2h_matches) == 0:
            self.logger.warning("No matches with H2H data to validate")
            return
        
        # Sample random matches
        sample_matches = h2h_matches.sample(min(sample_size, len(h2h_matches)))
        
        correct_count = 0
        total_checked = 0
        
        for idx, match_row in sample_matches.iterrows():
            home_team = match_row['home_team_id']
            away_team = match_row['away_team_id']
            match_date = match_row['date']
            
            self.logger.info(f"Validating streak for {home_team} vs {away_team} on {match_date}")
            
            # Get historical matches
            historical_matches = df[
                (((df['home_team_id'] == home_team) & (df['away_team_id'] == away_team)) |
                ((df['home_team_id'] == away_team) & (df['away_team_id'] == home_team))) &
                (df['date'] < match_date) &
                (df['status'].isin(['FT', 'AET', 'PEN']))
            ]
            
            # Manual calculation
            manual_streak = self._calculate_streak(historical_matches, home_team)
            
            # Compare
            is_correct = abs(match_row['h2h_streak'] - manual_streak) < 0.01
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
                
            total_checked += 1
            
            self.logger.info(f"{status} Calculated: {match_row['h2h_streak']}, Manual: {manual_streak}")
        
        accuracy = correct_count / total_checked * 100
        self.logger.info(f"=== STREAK VALIDATION COMPLETE ===")
        self.logger.info(f"Streak accuracy: {accuracy:.1f}% ({correct_count}/{total_checked} correct)")
        
        return accuracy

    def _get_competition_type_from_name_str(self, league_name: str) -> dict:
        """
        Determine competition type based on league name string
        """
        if not isinstance(league_name, str) or not league_name.strip():
            return {
              
                'is_cup_competition': 0,
                'is_europe_competition': 0,
                'is_league_competition': 1,
                'competition_tier': 1
            }
        
        league_name_lower = league_name.lower()
        
        # Cup competition detection
        cup_keywords = [
            'cup', 'trophy', 'shield', 'pokal', 'coupe', 'copa', 'coppa', 'supercup',
            'fa cup', 'dfb-pokal', 'coupe de france', 'copa del rey', 'coppa italia',
            'knockout', 'playoff', 'final', 'super cup', 'community shield', 'taca'
        ]
        is_cup = any(keyword in league_name_lower for keyword in cup_keywords)
        
        # European competition detection
        europe_keywords = [
            'champions league', 'europa league', 'europa conference', 'conference league',
            'uefa', 'european', 'euro_elite', 'champions cup', 'euro cup',
            'europa', 'conference', 'champions', 'euro'
        ]
        is_europe = any(keyword in league_name_lower for keyword in europe_keywords)
        
        # Domestic league detection (not cup and not europe)
        is_league = not (is_cup or is_europe)
        
        # Determine competition tier
        tier = 1  # Default to top tier
        if any(word in league_name_lower for word in ['2', 'ii', 'second', '2nd', 'championship', '2.', 'b']):
            tier = 2
        elif any(word in league_name_lower for word in ['3', 'iii', 'third', '3rd', '3.', 'c']):
            tier = 3
        elif any(word in league_name_lower for word in ['4', 'iv', 'fourth', '4th', '4.', 'd']):
            tier = 4
        
        return {
         
            'is_cup_competition': int(is_cup),
            'is_europe_competition': int(is_europe),
            'is_league_competition': int(is_league),
            'competition_tier': tier
        }

    def _add_competition_info_from_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add competition type information based on league names in DataFrame
        """
        df = df.copy()
        
        if 'league_name' not in df.columns:
            self.logger.warning("No 'league_name' column found in DataFrame")
            return df
        
        # Apply competition type mapping based on league names
        competition_data = df['league_name'].apply(self._get_competition_type_from_name_str)
        competition_df = pd.DataFrame(competition_data.tolist(), index=df.index)
        
        # Merge with original dataframe
        df = pd.concat([df, competition_df], axis=1)
        
        # Log competition distribution
        self._log_competition_stats(df)
        
        return df

    def _log_competition_stats(self, df: pd.DataFrame):
        """Log competition statistics"""
        cup_matches = df['is_cup_competition'].sum()
        europe_matches = df['is_europe_competition'].sum()
        league_matches = df['is_league_competition'].sum()
        total_matches = len(df)
        
        self.logger.info(f"Competition Distribution:")
        self.logger.info(f"  Cup matches: {cup_matches} ({cup_matches/total_matches*100:.1f}%)")
        self.logger.info(f"  European matches: {europe_matches} ({europe_matches/total_matches*100:.1f}%)")
        self.logger.info(f"  League matches: {league_matches} ({league_matches/total_matches*100:.1f}%)")
        
        # Log top competitions
        if 'competition_name' in df.columns:
            comp_counts = df['competition_name'].value_counts().head(10)
            self.logger.info("Top 10 competitions by match count:")
            for comp, count in comp_counts.items():
                self.logger.info(f"  {comp}: {count}")

    def _add_competition_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that leverage competition information"""
        df = df.copy()
        
        # Competition importance score
        df['competition_importance'] = (
            df['is_cup_competition'] * 1.5 + 
            df['is_europe_competition'] * 2.0 + 
            df['is_league_competition'] * 1.0
        )
        
        # Tier-based features
        #df['is_top_tier'] = (df['competition_tier'] == 1).astype(int)
        #df['is_lower_tier'] = (df['competition_tier'] > 1).astype(int)
        
        # Competition context for team experience
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            
            # European experience
            df[f'{prefix}_europe_experience'] = df.groupby(team_col)['is_europe_competition'].transform(
                lambda x: x.expanding().sum()
            )
            
            # Cup experience
            df[f'{prefix}_cup_experience'] = df.groupby(team_col)['is_cup_competition'].transform(
                lambda x: x.expanding().sum()
            )
    
        
        return df

    def _add_competition_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features describing the competition context and importance
        """
        df = df.copy()
        
        # Check if competition columns exist
        comp_cols = ['is_cup_competition', 'is_europe_competition']
        if not all(col in df.columns for col in comp_cols):
            self.logger.warning("Competition columns missing - skipping competition context")
            return df
        
        # Competition importance score (weighted)
        df['competition_importance'] = (
            df['is_cup_competition'] * 1.5 + 
            df['is_europe_competition'] * 2.0 + 
            (~df['is_cup_competition'] & ~df['is_europe_competition']).astype(int) * 1.0
        )
        
        # Knockout stage indicator
        if 'round' in df.columns:
            knockout_keywords = ['final', 'semi', 'quarter', 'round of', 'last', 'knockout', 'playoff']
            df['is_knockout_stage'] = (
                df['round'].str.lower().str.contains('|'.join(knockout_keywords), case=False, na=False) &
                (df['is_cup_competition'].any() == 1 | df['is_europe_competition'].any() == 1)
            ).astype(int)
            
            # Increase importance for knockout matches
            df.loc[df['is_knockout_stage'] == 1, 'competition_importance'] *= 1.5
        else:
            df['is_knockout_stage'] = 0
        
        # Competition fatigue factors
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            
            # Cup match frequency (last 10 matches)
            df[f'{prefix}_cup_fatigue'] = df.groupby(team_col)['is_cup_competition'].transform(
                lambda x: x.rolling(10, min_periods=1).sum()
            )
            
            # European match frequency (last 5 matches)
            df[f'{prefix}_europe_fatigue'] = df.groupby(team_col)['is_europe_competition'].transform(
                lambda x: x.rolling(5, min_periods=1).sum()
            )
            
            # Total competition load
            df[f'{prefix}_competition_load'] = (
                df[f'{prefix}_cup_fatigue'] * 0.7 + 
                df[f'{prefix}_europe_fatigue'] * 1.0
            )
        
        # Squad rotation likelihood
        if 'home_days_rest' in df.columns and 'away_days_rest' in df.columns:
            df['rotation_risk_home'] = (
                (df['home_cup_fatigue'] > 2) & 
                (df['home_days_rest'] < 4) &
                (df['is_europe_competition'] == 1)
            ).astype(int)
            
            df['rotation_risk_away'] = (
                (df['away_cup_fatigue'] > 2) & 
                (df['away_days_rest'] < 4) &
                (df['is_europe_competition'] == 1)
            ).astype(int)
        
        # Competition experience
        for prefix in ['home', 'away']:
            team_col = f'{prefix}_team_id'
            
            
            # Recent European experience (last 2 seasons)
            if 'season' in df.columns:
                # This would require season parsing - simplified version
                df[f'{prefix}_recent_europe_exp'] = df.groupby(team_col)['is_europe_competition'].transform(
                    lambda x: x.rolling(20, min_periods=1).sum()  # Approx last 20 matches
                )
        
        return df

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
            
            # First fill NA values with 0 or appropriate default value
            cols_to_fill = ['halftime_home', 'halftime_away', 'home_goals', 'away_goals', 'fulltime_home', 'fulltime_away']
            df[cols_to_fill] = df[cols_to_fill].fillna(0)
            
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
                self.logger.info(f"Restored missing column: {col}")

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

            self.logger.info(f"Calculating rolling averages for {len(self.combined_metrics)} metrics with windows {self.config['rolling_windows']}")
            
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

            self.logger.info(f"Calculated rolling averages completed")
            
            return df

    def _process_single_season_2(self, country: str, league: str, season: str, processed_fixtures: set) -> Optional[pd.DataFrame]:
        """Process a single season, filtering already processed fixtures"""
        season_path = Path(self.config['raw_dir']) / country / league / season
        processed_path = Path(self.config['merged_dir']) / country / league / season
        processed_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Processing {country}/{league}/{season}")
        
        try:
            fixtures_path = season_path / self.config['data_types']['fixtures']
            team_stats_path = season_path / self.config['data_types']['team_stats']
            odds_path = season_path / self.config['data_types']['odds']
            
            if not fixtures_path.exists():
                self.logger.warning(f"Missing fixtures file: {fixtures_path}")
                return None
            if not team_stats_path.exists():
                self.logger.warning(f"Missing team stats file: {team_stats_path}")
                return None
            if not odds_path.exists():
                self.logger.warning(f"Missing odds file: {team_stats_path}")
                return None
                
            fixtures = pd.read_csv(fixtures_path)
            team_stats = pd.read_csv(team_stats_path)
            odds = pd.read_csv(odds_path)

            # Filter out already processed fixtures
            if 'fixture_id' in fixtures.columns:
                new_fixtures = fixtures[~fixtures['fixture_id'].isin(processed_fixtures)]
                if len(new_fixtures) == 0:
                    self.logger.info(f"No new fixtures in {country}/{league}/{season}")
                    return None
                fixtures = new_fixtures

            # Add season information if not present
            if 'season' not in fixtures.columns:
                fixtures['season'] = season

            # Validate required columns
            required_fixture_cols = self.config['required_cols']['fixtures']
            missing_fixture_cols = [col for col in required_fixture_cols 
                                if col not in fixtures.columns]
            if missing_fixture_cols:
                raise ValueError(f"Missing required columns in fixtures: {missing_fixture_cols}")

            # Prepare team references
            team_ref = fixtures[['fixture_id', 'home_team_id', 'away_team_id']].copy()
            
            # Handle optional columns
            optional_team_stats_columns = ['expected_goals', 'goals_prevented', 'Assists', 'Counter  Attacks', 'Cross Attacks', 'Free Kicks', 
                                           'Goals','Goal Attempts','Substitutions','Throwins','Medical Treatment']
            columns_to_drop = [col for col in optional_team_stats_columns 
                            if col in team_stats.columns]
            
            # Safely filter team stats
            if columns_to_drop:
                team_stats = team_stats.drop(columns=columns_to_drop, errors='ignore')
            
            # Handle penalty columns in fixtures
            penalty_columns = ['penalty_home', 'penalty_away']
            if any(col in fixtures.columns for col in penalty_columns):
                fixtures = fixtures.drop(columns=penalty_columns, errors='ignore')
            
            # Merge team stats with references
            team_data = team_stats.merge(team_ref, on='fixture_id', how='left')
            
            # Split into home/away data
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
            
            # Final merge with validation
            if len(home_data) == 0 or len(away_data) == 0:
                raise ValueError("No home or away data found after splitting")
                
            # THIS IS THE CRITICAL FIX - return the merged data, not just fixtures
            merged_data = (
                fixtures
                .merge(home_data, on='fixture_id', how='left')
                .merge(away_data, on='fixture_id', how='left')
            )
            
            # Add odds data if available
            merged_data = self._add_odds_to_data(merged_data, odds)
            
            
            # Validate merge succeeded
            if len(merged_data) == 0:
                raise ValueError("Final merge resulted in empty DataFrame")
            
            self.logger.info(f"Successfully merged {len(merged_data)} records with team statistics")
            # SAVE THE MERGED DATA (BEFORE FEATURE ENGINEERING)
            merged_output_file = processed_path / f"merged_{season}.csv"
            merged_data.to_csv(merged_output_file, index=False)
            self.logger.info(f"Saved merged data to {merged_output_file}")
            
            self.logger.info(f"Successfully merged {len(merged_data)} records")
                
            return merged_data  # Return the actual merged data with team stats
            
        except Exception as e:
            self.logger.error(f"Error processing {country}/{league}/{season}: {str(e)}")
            if self.config.get('verbose', False):
                import traceback
                self.logger.debug(traceback.format_exc())
            return None
            
    def _process_single_season(self, country: str, league: str, season: str, processed_fixtures: set) -> Optional[pd.DataFrame]:
        """Process a single season, filtering already processed fixtures"""
        season_path = Path(self.config['raw_dir']) / country / league / season
        processed_path = Path(self.config['merged_dir']) / country / league / season
        processed_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Processing {country}/{league}/{season}")
        
        try:
            fixtures_path = season_path / self.config['data_types']['fixtures']
            team_stats_path = season_path / self.config['data_types']['team_stats']
            odds_path = season_path / self.config['data_types']['odds']
            
            self.logger.debug(f"Looking for files: {fixtures_path}, {team_stats_path}, {odds_path}")
            
            if not fixtures_path.exists():
                self.logger.warning(f"Missing fixtures file: {fixtures_path}")
                return None
            if not team_stats_path.exists():
                self.logger.warning(f"Missing team stats file: {team_stats_path}")
                return None
                
            fixtures = pd.read_csv(fixtures_path)
            team_stats = pd.read_csv(team_stats_path)
            self.logger.debug(f"Fixtures shape: {fixtures.shape}, Team stats shape: {team_stats.shape}")

            # Check if odds file exists
            if odds_path.exists():
                odds = pd.read_csv(odds_path)
                self.logger.info(f"Odds data found for {country}/{league}/{season}")
                self.logger.debug(f"Odds shape: {odds.shape}")
            else:
                odds = None
                self.logger.warning(f"Missing odds file: {odds_path}")

            # Filter out already processed fixtures
            if 'fixture_id' in fixtures.columns:
                new_fixtures = fixtures[~fixtures['fixture_id'].isin(processed_fixtures)]
                if len(new_fixtures) == 0:
                    self.logger.info(f"No new fixtures in {country}/{league}/{season}")
                    return None
                fixtures = new_fixtures
                self.logger.debug(f"After filtering processed fixtures: {fixtures.shape}")

            # Check if we have any fixtures left to process
            if len(fixtures) == 0:
                self.logger.info(f"No fixtures to process after filtering for {country}/{league}/{season}")
                return None

            # Add season information if not present
            if 'season' not in fixtures.columns:
                fixtures['season'] = season

            # Validate required columns
            required_fixture_cols = self.config['required_cols']['fixtures']
            missing_fixture_cols = [col for col in required_fixture_cols 
                                if col not in fixtures.columns]
            if missing_fixture_cols:
                error_msg = f"Missing required columns in fixtures: {missing_fixture_cols}. Available columns: {list(fixtures.columns)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Prepare team references
            team_ref = fixtures[['fixture_id', 'home_team_id', 'away_team_id']].copy()
            
            # Handle optional columns
            optional_team_stats_columns = ['expected_goals', 'goals_prevented', 'Assists', 'Counter  Attacks', 'Cross Attacks', 'Free Kicks', 
                                        'Goals','Goal Attempts','Substitutions','Throwins','Medical Treatment']
            columns_to_drop = [col for col in optional_team_stats_columns 
                            if col in team_stats.columns]
            
            # Safely filter team stats
            if columns_to_drop:
                team_stats = team_stats.drop(columns=columns_to_drop, errors='ignore')
            
            # Handle penalty columns in fixtures
            penalty_columns = ['penalty_home', 'penalty_away']
            if any(col in fixtures.columns for col in penalty_columns):
                fixtures = fixtures.drop(columns=penalty_columns, errors='ignore')
            
            # Merge team stats with references
            team_data = team_stats.merge(team_ref, on='fixture_id', how='left')
            self.logger.debug(f"Team data after merge: {team_data.shape}")
            
            # Split into home/away data
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
                
                self.logger.debug(f"Home data shape: {home_data.shape}, Away data shape: {away_data.shape}")
                
            except KeyError as e:
                error_msg = f"Missing team reference columns: {str(e)}. Team data columns: {list(team_data.columns)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Final merge with validation
            if len(home_data) == 0 or len(away_data) == 0:
                error_msg = "No home or away data found after splitting"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Merge fixtures with home and away data
            merged_data = (
                fixtures
                .merge(home_data, on='fixture_id', how='left')
                .merge(away_data, on='fixture_id', how='left')
            )
            
            self.logger.debug(f"Merged data shape before odds: {merged_data.shape}")
            
            # Add odds data if available
            merged_data = self._add_odds_to_data(merged_data, odds)
            
            self.logger.debug(f"Final merged data shape: {merged_data.shape}")
            self.logger.debug(f"Merged data columns: {list(merged_data.columns)}")
            
            # Validate merge succeeded
            if len(merged_data) == 0:
                error_msg = "Final merge resulted in empty DataFrame"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check if date column exists
            if 'date' not in merged_data.columns:
                error_msg = f"Missing 'date' column in merged data. Available columns: {list(merged_data.columns)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"Successfully merged {len(merged_data)} records with team statistics")
            
            # SAVE THE MERGED DATA (BEFORE FEATURE ENGINEERING)
            merged_output_file = processed_path / f"merged_{season}.csv"
            merged_data.to_csv(merged_output_file, index=False)
            self.logger.info(f"Saved merged data to {merged_output_file}")
            
            self.logger.info(f"Successfully merged {len(merged_data)} records")
                
            return merged_data  # Return the actual merged data with team stats and odds
            
        except Exception as e:
            self.logger.error(f"Error processing {country}/{league}/{season}: {str(e)}")
            if self.config.get('verbose', False):
                import traceback
                self.logger.debug(traceback.format_exc())
            return None

    def _merge_all_seasons(self, country: str, league: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        """Merge all seasons for a specific country/league"""
        league_data = pd.DataFrame()
        
        for season in seasons:
            season_data = self._process_single_season(country, league, season)
            if season_data is not None:
                league_data = pd.concat([league_data, season_data], ignore_index=True)
        
        if league_data.empty:
            self.logger.warning(f"No valid data found for {country}/{league}")
            return None
            
        return league_data
    
    def _preprocess_and_feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        self.logger.info("Starting preprocessing and feature engineering")
        self.logger.info(f"Initial data shape: {df.shape}")




        self.logger.info("Cleaning data")
        df = self._clean_data(df)
        df = self._create_temporal_features(df)
        df = self._create_target_column(df)
        #self.logger.info("Adding competition info from league names")
        #df = self._add_competition_info_from_name(df)
        #self.logger.info("Adding competition context")
        #df= self._add_competition_context(df)
        #self.logger.info("Adding competition metrics")
        #df = self._add_competition_specific_features(df)
        
        #df = self._create_standings(df)  
        #df = self._calculate_rolling_standings(df)
        #df = self._calculate_form_strength(df)
        #self._debug_momentum_calculation(df, team_id=504, league_id=135,season=2024)
        # Debug specific teams
        debug_teams = [492, 489, 505]  # Replace with actual team IDs

        # Option 1: Use debug mode in main function
        #df = self._create_h2h_features(df, LEAGUES)
        #df = self._add_new_metrics(df)
        #df = self._calculate_rolling_averages(df)
        df = df.fillna(0)
        self.logger.info(f"Final shape: {df.shape}")
        return df

   

    def _update_h2h_data(self, new_matches: pd.DataFrame):
        """Update H2H data with new matches - enhanced version"""
        new_matches = new_matches[new_matches['status'].isin(['FT', 'AET', 'PEN'])]  # Only completed matches
        
        for _, match in new_matches.iterrows():
            try:
                home_id = match['home_team_id']
                away_id = match['away_team_id']
                
                # Skip if team IDs are missing
                if pd.isna(home_id) or pd.isna(away_id):
                    continue
                    
                key = frozenset({int(home_id), int(away_id)})
                
                if key not in self.h2h_data:
                    self.h2h_data[key] = []
                
                # Add match to H2H data
                self.h2h_data[key].append({
                    'date': match['date'],
                    'home_id': int(home_id),
                    'home_goals': int(match['home_goals']),
                    'away_id': int(away_id),
                    'away_goals': int(match['away_goals']),
                    'league_id': match.get('league_id'),
                    'season': match.get('season'),
                    'is_cup_competition': match.get('is_cup_competition', 0),
                    'is_europe_competition': match.get('is_europe_competition', 0)
                })
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error adding match to H2H data: {e}")

    def _update_standings_data(self, new_matches: pd.DataFrame):
        """Update standings data with new matches - enhanced version"""
        completed_matches = new_matches[new_matches['status'].isin(['FT', 'AET', 'PEN'])]
        
        for _, match in completed_matches.iterrows():
            try:
                home_id = int(match['home_team_id'])
                away_id = int(match['away_team_id'])
                league_id = match['league_id']
                season = match.get('season', '2025')
                home_goals = int(match['home_goals'])
                away_goals = int(match['away_goals'])
                
                # Home team standings
                home_key = (home_id, league_id, season)
                if home_key not in self.standings_data:
                    self.standings_data[home_key] = {
                        'points': 0, 'goals_for': 0, 'goals_against': 0,
                        'wins': 0, 'draws': 0, 'losses': 0, 'matches_played': 0,
                        'goal_diff': 0
                    }
                
                # Away team standings
                away_key = (away_id, league_id, season)
                if away_key not in self.standings_data:
                    self.standings_data[away_key] = {
                        'points': 0, 'goals_for': 0, 'goals_against': 0,
                        'wins': 0, 'draws': 0, 'losses': 0, 'matches_played': 0,
                        'goal_diff': 0
                    }
                
                # Update home team
                self.standings_data[home_key]['goals_for'] += home_goals
                self.standings_data[home_key]['goals_against'] += away_goals
                self.standings_data[home_key]['matches_played'] += 1
                self.standings_data[home_key]['goal_diff'] = (
                    self.standings_data[home_key]['goals_for'] - 
                    self.standings_data[home_key]['goals_against']
                )
                
                # Update away team
                self.standings_data[away_key]['goals_for'] += away_goals
                self.standings_data[away_key]['goals_against'] += home_goals
                self.standings_data[away_key]['matches_played'] += 1
                self.standings_data[away_key]['goal_diff'] = (
                    self.standings_data[away_key]['goals_for'] - 
                    self.standings_data[away_key]['goals_against']
                )
                
                # Update points based on result
                if home_goals > away_goals:
                    self.standings_data[home_key]['points'] += 3
                    self.standings_data[home_key]['wins'] += 1
                    self.standings_data[away_key]['losses'] += 1
                elif home_goals == away_goals:
                    self.standings_data[home_key]['points'] += 1
                    self.standings_data[home_key]['draws'] += 1
                    self.standings_data[away_key]['points'] += 1
                    self.standings_data[away_key]['draws'] += 1
                else:
                    self.standings_data[away_key]['points'] += 3
                    self.standings_data[away_key]['wins'] += 1
                    self.standings_data[home_key]['losses'] += 1
                    
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error updating standings for match: {e}")


    def run_incremental_pipeline_2(self, force_processing: bool = False) -> pd.DataFrame:
        """Run pipeline with incremental processing, optionally force processing"""
        
        # Reset standings data at the start of processing if forcing
        if force_processing:
            self.standings_data = {}

        # Check if we should force processing regardless of new data
        if not force_processing and not self._check_new_data_exists():
            self.logger.info("No new data to process")
            return pd.DataFrame()
        
        processed_fixtures = self._get_processed_fixtures()
        data_structure = self._discover_data_structure()
        new_merged_data = pd.DataFrame()
        new_processed_data = pd.DataFrame()
        
        # Process data from each season
        for country, leagues in data_structure.items():
            for league, seasons in leagues.items():
                for season in seasons:
                    merged_data = self._process_single_season(country, league, season, processed_fixtures if not force_processing else set())
                    if merged_data is not None and not merged_data.empty:
                        new_merged_data = pd.concat([new_merged_data, merged_data], ignore_index=True)
                        
        if new_merged_data.empty and not force_processing:
            self.logger.info("No new data processed after filtering")
            return pd.DataFrame()
        
        # Preprocess new merged data (this will update standings_data internally)
        new_processed_data = self._preprocess_and_feature_engineer(new_merged_data)
        
        # Update historical data
        self._update_h2h_data(new_merged_data)
        self._update_standings_data(new_merged_data)
        
        # Combine with existing processed data
        final_path = Path(self.config['final_output'])
        if final_path.exists():
            try:
                existing_data = pd.read_csv(final_path)
                
                # Ensure date columns are consistent datetime types
                if 'date' in existing_data.columns:
                    existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
                if 'date' in new_processed_data.columns:
                    new_processed_data['date'] = pd.to_datetime(new_processed_data['date'], errors='coerce')
                
                # When forcing, we might want to replace rather than append
                if force_processing:
                    combined_data = new_processed_data
                else:
                    combined_data = pd.concat([existing_data, new_processed_data], ignore_index=True)
                
                combined_data = combined_data.drop_duplicates(subset=['fixture_id'], keep='last')
                
                # Sort by date (now both are datetime)
                if 'date' in combined_data.columns:
                    combined_data = combined_data.sort_values('date')
                
            except Exception as e:
                self.logger.error(f"Error combining with existing data: {e}")
                combined_data = new_processed_data
        else:
            combined_data = new_processed_data
        
        # Save final combined processed data
        combined_data.to_csv(self.config['final_output'], index=False)
        self._save_h2h_data()
        self._save_standings_data()
        
        self.logger.info(f"Incremental processing complete. Total records: {len(combined_data)}")
        self.logger.info(f"New records processed: {len(new_processed_data)}")
        
        return combined_data

    def run_incremental_pipeline(self, force_processing: bool = False) -> pd.DataFrame:
        """Run pipeline with incremental processing, optionally force processing"""
        
        # Reset standings data at the start of processing if forcing
        if force_processing:
            self.standings_data = {}
            self.logger.info("Force processing: resetting standings data")

        # Check if we should force processing regardless of new data
        if not force_processing:
            has_new_data = self._check_new_data_exists()
            self.logger.info(f"New data check: {has_new_data}")
            if not has_new_data:
                self.logger.info("No new data to process")
                return pd.DataFrame()
        
        processed_fixtures = self._get_processed_fixtures()
        self.logger.info(f"Already processed fixtures: {len(processed_fixtures)}")
        
        data_structure = self._discover_data_structure()
        new_merged_data = pd.DataFrame()
        new_processed_data = pd.DataFrame()
        
        # Process data from each season
        for country, leagues in data_structure.items():
            for league, seasons in leagues.items():
                for season in seasons:
                    self.logger.info(f"Processing {country} - {league} - {season}")
                    merged_data = self._process_single_season(
                        country, league, season, 
                        processed_fixtures if not force_processing else set()
                    )
                    if merged_data is not None and not merged_data.empty:
                        self.logger.info(f"  Found {len(merged_data)} new matches")
                        new_merged_data = pd.concat([new_merged_data, merged_data], ignore_index=True)
                    else:
                        self.logger.info(f"  No new matches found")
        
        self.logger.info(f"Total new matches to process: {len(new_merged_data)}")
        
        if new_merged_data.empty and not force_processing:
            self.logger.info("No new data processed after filtering")
            return pd.DataFrame()
        
        # Preprocess new merged data (this will update standings_data internally)
        new_processed_data = self._preprocess_and_feature_engineer(new_merged_data)
        self.logger.info(f"New processed data shape: {new_processed_data.shape}")
        
        # Update historical data
        self._update_h2h_data(new_merged_data)
        self._update_standings_data(new_merged_data)
        
        # Combine with existing processed data
        final_path = Path(self.config['final_output'])
        if final_path.exists() and not force_processing:
            try:
                existing_data = pd.read_csv(final_path)
                self.logger.info(f"Existing data shape: {existing_data.shape}")
                
                # Ensure date columns are consistent datetime types
                if 'date' in existing_data.columns:
                    existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
                if 'date' in new_processed_data.columns:
                    new_processed_data['date'] = pd.to_datetime(new_processed_data['date'], errors='coerce')
                
                # When forcing, we might want to replace rather than append
                if force_processing:
                    combined_data = new_processed_data
                else:
                    combined_data = pd.concat([existing_data, new_processed_data], ignore_index=True)
                
                # Remove duplicates - this is CRITICAL
                before_dedup = len(combined_data)
                combined_data = combined_data.drop_duplicates(subset=['fixture_id'], keep='last')
                after_dedup = len(combined_data)
                self.logger.info(f"Removed {before_dedup - after_dedup} duplicate fixtures")
                
                # Sort by date (now both are datetime)
                if 'date' in combined_data.columns:
                    combined_data = combined_data.sort_values('date')
                
            except Exception as e:
                self.logger.error(f"Error combining with existing data: {e}")
                combined_data = new_processed_data
        else:
            combined_data = new_processed_data
        
        # Save final combined processed data
        combined_data.to_csv(self.config['final_output'], index=False)
        self._save_h2h_data()
        self._save_standings_data()
        
        self.logger.info(f"Incremental processing complete. Total records: {len(combined_data)}")
        self.logger.info(f"New records processed: {len(new_processed_data)}")
        
        return combined_data

    def run_pipeline(self, force_processing: bool = False) -> pd.DataFrame:
        """Main pipeline method - runs incremental processing by default, with force option"""
        return self.run_incremental_pipeline(force_processing=force_processing)

    def simple_progress_bar(iterable, desc="Progress", total=None):
        """Enhanced progress bar with optional total parameter"""
        total = total or len(iterable)
        start_time = time.time()
        
        for i, item in enumerate(iterable):
            percent = (i + 1) / total * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            
            bar = ('#' * int(percent//2)).ljust(50)
            print(f"\r{desc}: [{bar}] {percent:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
            yield item
        
        print(f"\r{desc}: [{'#'*50}] 100.0% | Completed in {time.time()-start_time:.1f}s")
 

    def _add_odds_to_data(self, merged_data: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """
        Add odds data to the merged dataframe
        
        Args:
            merged_data: DataFrame containing fixture and team stats data
            odds: DataFrame containing odds data or None if no odds data available
            
        Returns:
            DataFrame with odds columns added
        """
        # Add empty odds columns first
        merged_data = self._add_empty_odds_columns(merged_data)
        
        # If no odds data available, return with empty columns
        if odds is None:
            return merged_data
        
        try:
            # Filter for "Match Winner" bets only
            match_winner_odds = odds[odds['bet_name'] == 'Match Winner'].copy()
            
            if match_winner_odds.empty:
                self.logger.warning("No 'Match Winner' odds found in the data")
                return merged_data
            
            # Process each fixture's odds
            for fixture_id in merged_data['fixture_id'].unique():
                fixture_odds = match_winner_odds[match_winner_odds['fixture_id'] == fixture_id]
                
                if not fixture_odds.empty:
                    # Update the odds values in the merged data
                    for _, row in fixture_odds.iterrows():
                        bet_value = row['bet_value']
                        bet_odd = row['bet_odd']
                        
                        if bet_value == 'Home':
                            merged_data.loc[merged_data['fixture_id'] == fixture_id, 'odds_home_win'] = bet_odd
                        elif bet_value == 'Draw':
                            merged_data.loc[merged_data['fixture_id'] == fixture_id, 'odds_draw'] = bet_odd
                        elif bet_value == 'Away':
                            merged_data.loc[merged_data['fixture_id'] == fixture_id, 'odds_away_win'] = bet_odd
            
            self.logger.info(f"Added odds data for {len(match_winner_odds['fixture_id'].unique())} fixtures")
            
        except Exception as e:
            self.logger.warning(f"Error processing odds data: {str(e)}")
            # If odds processing fails, keep the empty columns
        
        return merged_data

    def _add_empty_odds_columns(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add empty odds columns with NaN values to the dataframe
        
        Args:
            merged_data: DataFrame to add odds columns to
            
        Returns:
            DataFrame with empty odds columns added
        """
        odds_columns = ['odds_home_win', 'odds_draw', 'odds_away_win']
        for col in odds_columns:
            if col not in merged_data.columns:
                merged_data[col] = float('nan')
        
        return merged_data