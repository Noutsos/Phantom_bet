from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings
from tqdm import tqdm  # For progress tracking (optional)
warnings.filterwarnings("ignore")

class FootballDataPipeline:
    """
    Final corrected pipeline with:
    - Proper categorical feature handling
    - Accurate rolling averages for home/away teams
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Default configuration
        self.config = {
            'raw_dir': 'data/extracted',
            'merged_dir': 'data/merged',
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
            'rolling_windows': [3, 5, 10],
            'min_matches': 5
        }
        
        if config:
            self.config.update(config)
            
        Path(self.config['merged_dir']).mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        if self.config['verbose']:
            print(f"[FootballDataPipeline] {message}")

    def _discover_data_structure(self) -> Dict[str, List[str]]:
        """Discover available leagues and seasons"""
        structure = {}
        raw_path = Path(self.config['raw_dir'])
        
        for league_dir in raw_path.iterdir():
            if league_dir.is_dir():
                seasons = []
                for season_dir in league_dir.iterdir():
                    if season_dir.is_dir():
                        seasons.append(season_dir.name)
                if seasons:
                    structure[league_dir.name] = sorted(seasons)
        
        return structure
    
    def _create_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if all(col in df.columns for col in ['home_goals', 'away_goals']):
            df['outcome'] = np.select(
                condlist=[
                    df['home_goals'] > df['away_goals'],
                    df['home_goals'] < df['away_goals']
                ],
                choicelist=['home_win', 'away_win'],
                default='draw'
            )
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
        
        for idx, row in df.iterrows():
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
            'home_form': 'object',
            'away_form': 'object'
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
        for idx, row in df.iterrows():
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
            df.at[idx, 'home_form'] = home_form_str
            
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
            df.at[idx, 'away_form'] = away_form_str
            
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

    def _create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixed H2H calculator with corrected streak calculation"""
        df = df.copy()
        df = df.sort_values('date')
        
        h2h_features = [
            'h2h_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
            'h2h_home_win_pct', 'h2h_away_win_pct',
            'h2h_recent_home_wins_last5',
            'h2h_recent_away_wins_last5',
            'h2h_recent_draws_last5',
            'h2h_recent_avg_goals_last5',
            'h2h_streak', 'h2h_avg_goals'
        ]
        
        # Initialize all H2H columns
        for col in h2h_features:
            if col == 'h2h_streak':
                df[col] = "N"  # Initialize as empty string
            else:
                df[col] = 0.0  # Initialize as float 0.0

        # Pre-calculate league averages
        league_avg_goals = {}
        for league in df['league_name'].unique():
            league_matches = df[df['league_name'] == league]
            league_avg_goals[league] = (league_matches['home_goals'].mean() + 
                                    league_matches['away_goals'].mean()) / 2

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating H2H"):
            home_team = row['home_team_id']
            away_team = row['away_team_id']
            current_date = row['date']
            league = row['league_name']
            
            # Get ALL historical matches between these teams (any league/season)
            past_matches = df[
                (((df['home_team_id'] == home_team) & 
                (df['away_team_id'] == away_team)) |
                ((df['home_team_id'] == away_team) & 
                (df['away_team_id'] == home_team)))
                &
                (df['date'] < current_date)
            ].copy()
            
            if len(past_matches) == 0:
                # Apply league averages for new matchups
                df.at[idx, 'h2h_avg_goals'] = league_avg_goals.get(league, 2.5)
                df.at[idx, 'h2h_home_win_pct'] = 0.45  # Typical home win rate
                df.at[idx, 'h2h_away_win_pct'] = 0.30  # Typical away win rate
                continue
                
            # Sort matches by date (newest first) for streak calculation
            past_matches = past_matches.sort_values('date', ascending=False)
            
            # Calculate basic H2H stats
            total_matches = len(past_matches)
            df.at[idx, 'h2h_matches'] = total_matches
            
            # Home wins (from perspective of current home team)
            home_wins = len(past_matches[
                ((past_matches['home_team_id'] == home_team) & (past_matches['home_goals'] > past_matches['away_goals'])) |
                ((past_matches['away_team_id'] == home_team) & (past_matches['away_goals'] > past_matches['home_goals']))
            ])
            df.at[idx, 'h2h_home_wins'] = home_wins
            
            # Away wins (from perspective of current away team)
            away_wins = len(past_matches[
                ((past_matches['home_team_id'] == away_team) & (past_matches['home_goals'] > past_matches['away_goals'])) |
                ((past_matches['away_team_id'] == away_team) & (past_matches['away_goals'] > past_matches['home_goals']))
            ])
            df.at[idx, 'h2h_away_wins'] = away_wins
            
            # Draws
            draws = len(past_matches[past_matches['home_goals'] == past_matches['away_goals']])
            df.at[idx, 'h2h_draws'] = draws
            
            # Goal stats (total goals, not averages)
            home_goals = past_matches.apply(
                lambda x: x['home_goals'] if x['home_team_id'] == home_team else x['away_goals'], axis=1).sum()
            away_goals = past_matches.apply(
                lambda x: x['away_goals'] if x['home_team_id'] == home_team else x['home_goals'], axis=1).sum()
            
            df.at[idx, 'h2h_home_goals'] = home_goals
            df.at[idx, 'h2h_away_goals'] = away_goals
            df.at[idx, 'h2h_goal_diff'] = home_goals - away_goals
            
            # Win percentages
            df.at[idx, 'h2h_home_win_pct'] = home_wins / total_matches if total_matches > 0 else 0
            df.at[idx, 'h2h_away_win_pct'] = away_wins / total_matches if total_matches > 0 else 0
            
            # Calculate average goals for all matches
            df.at[idx, 'h2h_avg_goals'] = (home_goals + away_goals) / total_matches if total_matches > 0 else league_avg_goals.get(league, 2.5)
            
            # Recent form (last 5 matches)
            recent_matches = past_matches.head(5)
            if len(recent_matches) > 0:
                # Recent home wins
                df.at[idx, 'h2h_recent_home_wins_last5'] = len(recent_matches[
                    ((recent_matches['home_team_id'] == home_team) & (recent_matches['home_goals'] > recent_matches['away_goals'])) |
                    ((recent_matches['away_team_id'] == home_team) & (recent_matches['away_goals'] > recent_matches['home_goals']))
                ])
                
                # Recent away wins
                df.at[idx, 'h2h_recent_away_wins_last5'] = len(recent_matches[
                    ((recent_matches['home_team_id'] == away_team) & (recent_matches['home_goals'] > recent_matches['away_goals'])) |
                    ((recent_matches['away_team_id'] == away_team) & (recent_matches['away_goals'] > recent_matches['home_goals']))
                ])
                
                # Recent draws
                df.at[idx, 'h2h_recent_draws_last5'] = len(recent_matches[
                    recent_matches['home_goals'] == recent_matches['away_goals']
                ])
                
                # Recent average goals
                df.at[idx, 'h2h_recent_avg_goals_last5'] = (
                    recent_matches['home_goals'].sum() + recent_matches['away_goals'].sum()
                ) / len(recent_matches)
                
            # IMPROVED STREAK CALCULATION
            streak_type = None
            streak_length = 0
            
            for _, match in past_matches.iterrows():
                # Determine result from perspective of current home team
                if match['home_team_id'] == home_team:
                    if match['home_goals'] > match['away_goals']:
                        result = 'H'
                    elif match['home_goals'] < match['away_goals']:
                        result = 'A'
                    else:
                        result = 'D'
                else:
                    if match['away_goals'] > match['home_goals']:
                        result = 'H'
                    elif match['away_goals'] < match['home_goals']:
                        result = 'A'
                    else:
                        result = 'D'
                
                # Initialize streak if first match
                if streak_type is None:
                    streak_type = result
                    streak_length = 1
                # Continue streak if same result
                elif result == streak_type:
                    streak_length += 1
                # Break streak if different result
                else:
                    break
            
            # Format streak string
            if streak_length > 0:
                if streak_length > 1:
                    df.at[idx, 'h2h_streak'] = f"{streak_type}{streak_length}"
                else:
                    df.at[idx, 'h2h_streak'] = streak_type
        
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
        
        # Clean column names
        df.columns = (
            df.columns.str.lower()
            .str.replace(r'[^a-z0-9_]', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
        )
        
        # Columns to drop
        cols_to_drop = ['home_team_name', 'away_team_name', 'home_winner', 'away_winner',
                    'home_team_logo', 'away_team_logo']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        # --- Numeric Conversion ---
        for col in df.columns:
            if col not in ['fixture_id', 'team_id', 'league', 'season', 'date']:
                try:
                    if df[col].dtype == 'object' and df[col].astype(str).str.contains('%').any():
                        df[col] = (
                            df[col].astype(str)
                            .str.replace('%', '', regex=False)
                            .str.replace(',', '.', regex=False)
                            .apply(pd.to_numeric, errors='coerce')
                            / 100
                        )
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError, AttributeError):
                    continue

        # --- Feature Engineering ---
        self.original_metrics = [
            'shots_on_goal', 'shots_off_goal', 'total_shots', 'blocked_shots', 
            'shots_insidebox', 'shots_outsidebox', 'fouls', 'corner_kicks', 
            'offsides', 'ball_possession', 'yellow_cards', 'red_cards', 
            'goalkeeper_saves', 'total_passes', 'passes_accurate', 'passes',
            'expected_goals', 'goals_prevented'
        ]
        
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

        # Finalize new metrics sets
        self.original_metrics = {
            f"{prefix}_{metric}" 
            for metric in self.original_metrics 
            for prefix in ['home', 'away']
        }

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
            self._log(f"Dropped {len(cols_to_drop)} original metrics")

        # Restore string columns
        for col, data in string_backups.items():
            if col in df.columns:
                df[col] = data
            else:
                df[col] = data
                self._log(f"Restored missing column: {col}")

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
            self._log(f"Dropped {len(cols_to_drop)} non rolled metrics")


        
        return df

    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove non-rolled team metrics features while preserving:
        - Rolled/aggregated versions of metrics
        - Match outcome features
        - Contextual features (date, venue, etc.)
        - Derived metrics (xg_difference, possession_difference, etc.)
        
        Args:
            df: DataFrame containing both raw and rolled features
            
        Returns:
            DataFrame with non-rolled team metrics removed
        """
        # 1. Identify all team-specific metric columns
        team_metric_cols = {
            f"{prefix}_{metric}"
            for metric in self.combined_metrics
            for prefix in ['home', 'away']
            if f"{prefix}_{metric}" in df.columns
        }
        
        # 2. Identify which metrics exist in both rolled and unrolled forms
        non_rolled_metrics = set()
        for metric in self.combined_metrics:
            # Check if raw metric exists without corresponding rolled version
            has_home = f'home_{metric}' in df.columns
            has_away = f'away_{metric}' in df.columns
            has_rolled = any(col.endswith(f'_{metric}_roll') for col in df.columns)
            
            if (has_home or has_away) and not has_rolled:
                non_rolled_metrics.add(metric)
        
        # 3. Prepare columns to remove (only raw versions when rolled exist)
        cols_to_remove = [
            f"{prefix}_{metric}"
            for metric in non_rolled_metrics
            for prefix in ['home', 'away']
            if f"{prefix}_{metric}" in df.columns
        ]
        
        # 4. Always keep these special columns regardless of rolling status
        protected_cols = {
            'fixture_id', 'date', 'league_id', 'season',
            'home_team', 'away_team', 'venue_id',
            'home_goals', 'away_goals', 'outcome',
            # Keep all differential metrics
            *[col for col in df.columns if col in self.diff_metrics],
            # Keep all momentum metrics
          
        }
        
        # 5. Final removal (protecting special columns)
        cols_to_remove = [col for col in cols_to_remove if col not in protected_cols]
        
        self._log(f"Removing {len(cols_to_remove)} unrolled features: {sorted(cols_to_remove)}")
        
        return df.drop(columns=cols_to_remove, errors='ignore').copy()

    
    def _preprocess_and_feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing and feature engineering method that combines:
        1. Standings creation
        2. Additional feature engineering
        3. Rolling averages
        """
        #df = self._create_target_column(df)
        
        # First create the standings
        #df = self._create_standings(df)

        # 2. Add head-to-head features
        #df = self._create_h2h_features(df)
        
        #print(df.columns)
        # Then add other features
        df = self._add_new_metrics(df)
        
        # Finally calculate rolling averages
        df = self._calculate_rolling_averages(df)

        # Cleanup to keep only relevant features
        #df = self._cleanup_features(df)
        
        return df
    
    def _process_single_season(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Process and merge data for a single league season"""
        season_path = Path(self.config['raw_dir']) / league / season
        self._log(f"Processing {league} - {season}")
        
        try:
            # Load data files
            fixtures_path = season_path / self.config['data_types']['fixtures']
            team_stats_path = season_path / self.config['data_types']['team_stats']
            
            if not fixtures_path.exists() or not team_stats_path.exists():
                self._log(f"Missing files for {league}/{season}")
                return None
                
            fixtures = pd.read_csv(fixtures_path)
            team_stats = pd.read_csv(team_stats_path)
            
            # Verify required columns
            required_fixture_cols = self.config['required_cols']['fixtures']
            missing_fixture_cols = [col for col in required_fixture_cols if col not in fixtures.columns]
            if missing_fixture_cols:
                raise ValueError(f"Missing columns in fixtures: {missing_fixture_cols}")
            

            
            # Prepare team references
            team_ref = fixtures[['fixture_id', 'home_team_id', 'away_team_id']].copy()
            
            # Define optional columns that might be missing
            optional_columns = ['expected_goals', 'goals_prevented']
            
            # Get list of columns that actually exist in the data
            available_optional = [col for col in optional_columns if col in team_stats.columns]
            
            # Select only the columns we want to keep
            columns_to_keep = [
                col for col in team_stats.columns 
                if col not in optional_columns 
            ]
            
            # Filter the team_stats dataframe
            team_stats = team_stats[columns_to_keep]
            
            
            # Merge team stats with team references
            team_data = team_stats.merge(team_ref, on='fixture_id', how='left')
            
            # Split into home and away data
            home_data = (
                team_data[team_data['team_id'] == team_data['home_team_id']]
                .drop(columns=['home_team_id', 'away_team_id', 'team_id'])
                .rename(columns=lambda x: f'home_{x}' if x != 'fixture_id' else x)
            )
            
            away_data = (
                team_data[team_data['team_id'] == team_data['away_team_id']]
                .drop(columns=['home_team_id', 'away_team_id', 'team_id'])
                .rename(columns=lambda x: f'away_{x}' if x != 'fixture_id' else x)
            )
            
            # Merge with fixtures
            merged = (
                fixtures
                .merge(home_data, on='fixture_id', how='left')
                .merge(away_data, on='fixture_id', how='left')
            )
            
            return merged
            
        except Exception as e:
            self._log(f"Error processing {league}/{season}: {str(e)}")
            return None

    def run_pipeline(self) -> pd.DataFrame:
        """Run pipeline for all leagues and seasons"""
        data_structure = self._discover_data_structure()
        all_data = pd.DataFrame()
        
        for league, seasons in data_structure.items():
            # Create league folder in merged directory
            league_dir = Path(self.config['merged_dir']) / league
            league_dir.mkdir(exist_ok=True)
            
            league_data = pd.DataFrame()
            
            for season in seasons:
                season_data = self._process_single_season(league, season)
                
                if season_data is not None:
                    # Add to league dataset
                    league_data = pd.concat([league_data, season_data], ignore_index=True)
            
            if not league_data.empty:
                # Apply preprocessing and feature engineering
                league_data = self._preprocess_and_feature_engineer(league_data)
                
                # Save the processed league file
                league_filename = "all_seasons_merged.csv"
                league_path = league_dir / league_filename
                league_data.to_csv(league_path, index=False)
                self._log(f"Saved processed data to {league_path}")
                
                # Add to complete dataset
                all_data = pd.concat([all_data, league_data], ignore_index=True)


        
        if not all_data.empty:
            # Save final combined file
            all_data.to_csv(self.config['final_output'], index=False)
            self._log(f"\nPipeline complete. Final dataset saved to {self.config['final_output']}")
            self._log(f"Final Dataset contains {len(all_data)} records and {len(all_data.columns)} features.")
            self._log(f"Columns: {', '.join(all_data.columns)}")
        else:
            self._log("\nPipeline completed but no data was processed")
        
        return all_data

if __name__ == "__main__":
    config = {
        'raw_dir': 'data/extracted',
        'final_output': 'data/final_processed.csv',
        'verbose': True,
        'rolling_windows': [3],  # Calculate 3, 5, and 10-match rolling averages
        'min_matches': 3 , # Require at least 5 matches for rolling averages
        'drop_non_roll_features': True, # Drop non rolled features after rolling averages
        'drop_original_metrics': True, # Drop original features after rolling averages
    }
    
    pipeline = FootballDataPipeline(config)
    final_data = pipeline.run_pipeline()