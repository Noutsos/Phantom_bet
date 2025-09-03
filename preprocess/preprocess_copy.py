import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
from sklearn.preprocessing import OneHotEncoder


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
            "injuries": self._process_injuries,
            "lineups": self._process_lineups,
            "player_stats": self._process_player_statistics,
            "standings": self._generate_team_standings,
            "team_stats": self._process_team_statistics,
        }
        
        # File name mappings
        self.file_patterns = {
            "fixtures": "fixture_events.csv",
            "injuries": "injuries.csv",
            "lineups": "lineups.csv",
            "player_stats": "player_statistics.csv",
            "standings": "fixture_events.csv",
            "team_stats": "team_statistics.csv",
        }
        
        # Processed file name suffixes
        self.processed_suffix = "_processed.csv"
        
        # Cache for fixtures data needed by other processors
        self._fixtures_cache = {}

    def _process_fixture_events(self, df):
        """
        Processes fixture data with:
        - Time-lagged features
        - First-match handling
        - Opponent-aware stats
        - Leakage prevention
        - Multi-class outcome column
        
        Outcome encoding:
        - 0: Draw
        - 1: Home win
        - 2: Away win
        """
        # Use the pipeline's configured window size
        window = self.config.get('window', 5)
        league_avg_goals = self.config.get('league_avg_goals', 1.5)
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['home_team_id', 'date'])
        
        # --- Create Outcome Column (MUST BE FIRST STEP TO PREVENT LEAKAGE) ---
        df['outcome'] = np.select(
            condlist=[
                df['home_goals'] > df['away_goals'],  # Home win
                df['home_goals'] < df['away_goals']  # Away win
            ],
            choicelist=[1, 2],  # 1=home win, 2=away win
            default=0  # 0=draw
        )
        
        # --- Static Features ---
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        # --- Team-Level Features ---
        for team_type in ['home', 'away']:
            team_id = f'{team_type}_team_id'
            opp_type = 'away' if team_type == 'home' else 'home'
            
            # Goals scored (fill first match with league average)
            df[f'{team_type}_goals'] = (df.groupby(team_id)[f'{team_type}_goals']
                                        .shift(1)
                                        .rolling(window, min_periods=1)
                                        .mean()
                                        .fillna(league_avg_goals))
            
            # Goals conceded
            df[f'{team_type}_goals_conceded'] = (df.groupby(team_id)[f'{opp_type}_goals']
                                                .shift(1)
                                                .rolling(window, min_periods=1)
                                                .mean()
                                                .fillna(league_avg_goals))
            
            # Win/loss stats
            df[f'{team_type}_winner'] = (df[f'{team_type}_goals'] > df[f'{opp_type}_goals']).astype(int)
            df[f'{team_type}_win_streak'] = (df.groupby(team_id)[f'{team_type}_winner']
                                            .shift(1)
                                            .rolling(window, min_periods=1)
                                            .sum()
                                            .fillna(0))
        
        # --- Head-to-Head Features ---
        df = df.sort_values(['home_team_id', 'away_team_id', 'date'])
        df['h2h_home_wins'] = (df.groupby(['home_team_id', 'away_team_id'])['home_winner']
                            .shift(1)
                            .rolling(5, min_periods=1)
                            .sum()
                            .fillna(0))
        
        # --- First-Match Flags ---
        df['is_home_first_match'] = ~df.duplicated(['home_team_id'], keep='first')
        df['is_away_first_match'] = ~df.duplicated(['away_team_id'], keep='first')
        
        # --- Cleanup ---
        to_drop = ['home_winner', 'away_winner']
        return df.drop(columns=to_drop, errors='ignore')
    
    def _process_lineups(self, lineup_df):
        """
        Processes lineup data into team-level features while avoiding data leakage.
        Returns engineered features at (fixture_id × team_id) level.
        
        Parameters:
            lineup_df (pd.DataFrame): Raw lineup data with multiple rows per fixture-team
            
        Returns:
            pd.DataFrame: Engineered features ready for modeling
        """
        df = lineup_df.copy()
        
        # Validate required columns
        required_cols = ['fixture_id', 'team_id', 'player_id', 'player_pos', 'is_substitute']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # --- Feature 1: Basic Team Stats ---
        team_features = df.groupby(['fixture_id', 'team_id']).agg({
            'player_id': 'count',
            'is_substitute': 'sum',
            'formation': 'first',
            'coach_id': 'first'
        }).rename(columns={
            'player_id': 'squad_size',
            'is_substitute': 'substitutes_count',
            'formation': 'formation',
            'coach_id': 'coach_id'
        }).reset_index()
        
        # --- Feature 2: Position Distribution ---
        position_counts = (
            df[df['is_substitute'] == False]  # Only starting players
            .groupby(['fixture_id', 'team_id', 'player_pos'])
            .size()
            .unstack(fill_value=0)
            .add_prefix('pos_')
            .reset_index()
        )
        
        # --- Feature 3: Player Experience (would need external data) ---
        # This would require merging with player stats data
        # For now just include placeholder
        #team_features['avg_player_experience'] = 0  # Placeholder
        
        # --- Feature 4: Formation Features ---
        if 'formation' in df.columns:
            team_features['formation'] = team_features['formation'].fillna('Unknown')
            # One-hot encode common formations
            common_formations = ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1']
            for formation in common_formations:
                team_features[f'formation_{formation}'] = (
                    team_features['formation'].str.contains(formation).astype(int)
                )
        else:
            team_features['formation'] = 'Unknown'
        
        # --- Feature 5: Substitute Quality ---
        # Calculate average position strength of substitutes
        if 'player_pos' in df.columns:
            pos_strength = {
                'G': 1, 'D': 2, 'M': 3, 'F': 4  # Simple position importance scale
            }
            df['pos_strength'] = df['player_pos'].map(pos_strength).fillna(0)
            
            sub_strength = (
                df[df['is_substitute'] == True]
                .groupby(['fixture_id', 'team_id'])['pos_strength']
                .mean()
                .reset_index()
                .rename(columns={'pos_strength': 'substitute_quality'})
            )
            
            team_features = team_features.merge(
                sub_strength, 
                on=['fixture_id', 'team_id'], 
                how='left'
            ).fillna({'substitute_quality': 0})
        else:
            team_features['substitute_quality'] = 0
        
        # --- Combine All Features ---
        # Merge position counts with team features
        final_features = team_features.merge(
            position_counts,
            on=['fixture_id', 'team_id'],
            how='left'
        ).fillna(0)
        
        # Add time-safe features (would need fixture dates)
        # final_features = add_time_features(final_features, fixtures_df)
        
        return final_features
    
    def _process_injuries(self, injuries_df):
        """
        Processes injury data with configurable rolling windows.
        Returns engineered features at (fixture_id × team_id) level.
        
        Args:
            injuries_df: DataFrame containing injury data
            window_short: Days for short-term rolling window (default: 7)
            window_long: Days for long-term rolling window (default: 30)
        
        Returns:
            DataFrame with injury features
        """
        # Use the pipeline's configured window size
        window_short = self.config.get('window_short', 7)
        window_long = self.config.get('window_long', 30)
        df = injuries_df.copy()
        
        # Validate required columns exist
        required_cols = ['fixture_id', 'date', 'team_id', 'player_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['team_id', 'date'])
        
        # --- Feature 1: Current Match Injury Counts ---
        current_features = df.groupby(['fixture_id', 'team_id', 'date']).agg({
            'player_id': 'count',
            'type': lambda x: x.str.contains('Muscle|Fracture|ACL|Rupture', case=False, na=False).sum()
        }).rename(columns={
            'player_id': 'current_injuries',
            'type': 'current_serious_injuries'
        }).reset_index()
        
        # --- Feature 2: Rolling Injury Burden ---
        # Create daily injury counts
        daily_counts = (
            df.groupby(['team_id', pd.Grouper(key='date', freq='D')])
            .size()
            .rename('daily_injuries')
            .reset_index()
        )
        
        # Calculate rolling averages with configurable windows
        def add_rolling_features(group):
            group = group.set_index('date')
            group[f'injuries_{window_short}d'] = group['daily_injuries'].rolling(f'{window_short}D', min_periods=1).mean()
            group[f'injuries_{window_long}d'] = group['daily_injuries'].rolling(f'{window_long}D', min_periods=1).mean()
            return group.reset_index()
        
        rolling_features = (
            daily_counts
            .groupby('team_id', group_keys=False)
            .apply(add_rolling_features)
            .reset_index(drop=True)
        )
        
        # --- Feature 3: Key Injury Types ---
        if 'type' in df.columns:
            # Create injury type flags
            injury_types = df[df['type'].notna()].copy()
            injury_types['is_muscle'] = injury_types['type'].str.contains('Muscle', case=False, na=False)
            injury_types['is_serious'] = injury_types['type'].str.contains('Fracture|ACL|Rupture', case=False, na=False)
            
            # Calculate daily type counts
            type_counts = (
                injury_types.groupby(['team_id', pd.Grouper(key='date', freq='D')])
                [['is_muscle', 'is_serious']]
                .sum()
                .reset_index()
            )
            
            # Add rolling type counts with configurable window
            def add_type_rolling(group):
                group = group.set_index('date')
                group[f'muscle_{window_long}d'] = group['is_muscle'].rolling(f'{window_long}D', min_periods=1).sum()
                group[f'serious_{window_long}d'] = group['is_serious'].rolling(f'{window_long}D', min_periods=1).sum()
                return group.reset_index()
            
            type_features = (
                type_counts
                .groupby('team_id', group_keys=False)
                .apply(add_type_rolling)
                .reset_index(drop=True)
            )
        else:
            # Create empty type features if no type data
            unique_dates = daily_counts['date'].unique()
            type_features = pd.DataFrame({
                'team_id': np.repeat(daily_counts['team_id'].unique(), len(unique_dates)),
                'date': np.tile(unique_dates, len(daily_counts['team_id'].unique())),
                f'muscle_{window_long}d': 0,
                f'serious_{window_long}d': 0
            })
        
        # --- Combine All Features ---
        # First merge current with rolling
        features = current_features.merge(
            rolling_features,
            on=['team_id', 'date'],
            how='left'
        )
        
        # Then merge with type features
        features = features.merge(
            type_features[['team_id', 'date', f'muscle_{window_long}d', f'serious_{window_long}d']],
            on=['team_id', 'date'],
            how='left'
        ).fillna(0)
        
        # Calculate composite injury burden score
        features['injury_burden'] = (
            0.4 * features['current_injuries'] +
            0.3 * features[f'injuries_{window_short}d'] +
            0.2 * features[f'muscle_{window_long}d'] +
            0.1 * features[f'serious_{window_long}d']
        )
        
        # Select final features
        final_features = [
            'fixture_id', 'team_id', 'date',
            'current_injuries', 'current_serious_injuries',
            f'injuries_{window_short}d', f'injuries_{window_long}d',
            f'muscle_{window_long}d', f'serious_{window_long}d',
            'injury_burden'
        ]
        
        return features[final_features].drop_duplicates()
    
    def _process_team_statistics(self, stats_df, fixtures_df):
        """
        Processes team stats with bulletproof type handling and configurable rolling averages.
        Returns data with consistent lowercase column names.
        """
        # Use the pipeline's configured window size
        window = self.config.get('window', 5)
        # Validate inputs
        required_stats = ['fixture_id', 'team_id']
        missing = [col for col in required_stats if col not in stats_df.columns]
        if missing:
            raise ValueError(f"Missing columns in stats_df: {missing}")
        
        required_fixtures = ['fixture_id', 'date']
        missing = [col for col in required_fixtures if col not in fixtures_df.columns]
        if missing:
            raise ValueError(f"Missing columns in fixtures_df: {missing}")

        # Create working copies
        df = stats_df.copy()
        fixtures_df = fixtures_df.copy()
        
        # Standardize column names (lowercase with underscores)
        def clean_column_names(df):
            df.columns = (
                df.columns.str.lower()
                .str.replace(r'[^a-z0-9_]', '_', regex=True)  # Replace all special chars
                .str.replace(r'_+', '_', regex=True)  # Remove duplicate underscores
                .str.strip('_')
            )
            return df
        
        df = clean_column_names(df)
        fixtures_df = clean_column_names(fixtures_df)
        
        # --- TYPE CONVERSION SAFEGUARDS ---
        # Convert IDs to consistent string type
        df['fixture_id'] = df['fixture_id'].astype(str).str.strip()
        df['team_id'] = df['team_id'].astype(str).str.strip()
        fixtures_df['fixture_id'] = fixtures_df['fixture_id'].astype(str).str.strip()
        
        # Convert numeric columns (excluding IDs)
        numeric_cols = []
        for col in df.columns:
            if col not in required_stats:
                # Handle percentage columns
                if df[col].dtype == 'object' and df[col].astype(str).str.contains('%').any():
                    df[col] = (
                        df[col].astype(str)
                        .str.replace('%', '')
                        .str.replace(',', '.')
                        .astype(float) / 100
                    )
                    numeric_cols.append(col)
                # Convert other numeric columns
                else:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except (ValueError, TypeError):
                        continue

        # Convert and validate date column
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'], errors='coerce', utc=True)
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

        # --- FINAL TYPE CHECKS BEFORE SORTING ---
        df['team_id'] = df['team_id'].astype(str)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        if not pd.api.types.is_string_dtype(df['team_id']):
            problematic = df[~df['team_id'].apply(lambda x: isinstance(x, str))]['team_id'].unique()
            raise TypeError(f"team_id contains non-string values: {problematic}")
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            problematic = df[~pd.to_datetime(df['date'], errors='coerce').notna()]['date'].unique()
            raise TypeError(f"date contains invalid values: {problematic}")

        # Sort by team_id (string) and date (datetime)
        df = df.sort_values(['team_id', 'date'])

        # --- Calculate Rolling Averages ---
        rolling_features = []
        for team in df['team_id'].unique():
            team_data = df[df['team_id'] == team].copy()
            
            # Calculate rolling stats (excluding current match)
            for col in numeric_cols:
                team_data[col] = (
                    team_data[col].shift(1)  # Exclude current match
                    .rolling(window, min_periods=1)
                    .mean()
                )
            rolling_features.append(team_data)
        
        # Combine all teams
        df = pd.concat(rolling_features).sort_values(['team_id', 'date'])

        # --- Create Derived Features ---
        # Shot accuracy
        if all(col in df.columns for col in ['shots_on_goal', 'total_shots']):
            df['shot_accuracy'] = (
                df['shots_on_goal'] / df['total_shots'].replace(0, 1)
            ).clip(0, 1)

        # Defensive intensity
        defensive_cols = [col for col in ['fouls', 'blocked_shots', 'goalkeeper_saves'] 
                        if col in df.columns]
        if defensive_cols:
            df['defensive_actions'] = df[defensive_cols].sum(axis=1)

        # Team style
        if 'ball_possession' in df.columns:
            df['playing_style'] = pd.cut(
                df['ball_possession'],
                bins=[0, 40, 60, 100],
                labels=['counter', 'balanced', 'possession']
            ).astype(object)

        # Select only relevant columns
        keep_cols = ['fixture_id', 'team_id'] + numeric_cols
        if 'shot_accuracy' in df.columns:
            keep_cols.append('shot_accuracy')
        if 'defensive_actions' in df.columns:
            keep_cols.append('defensive_actions')
        if 'playing_style' in df.columns:
            keep_cols.append('playing_style')

        return df[keep_cols].fillna(0)
    
    def _process_player_statistics(self, player_stats_df, fixtures_df):
        """
        Processes player stats with configurable rolling averages window.
        Returns rolling features without suffixes in column names.
        
        Parameters:
            player_stats_df: DataFrame with player match stats
            fixtures_df: DataFrame with fixture dates
            window: Number of matches for rolling averages (default: 5)
        
        Returns:
            DataFrame with rolling features at (fixture_id × player_id) level
        """
        # Use the pipeline's configured window size
        window = self.config.get('window', 5)
        # Validate inputs
        required_player_cols = ['fixture_id', 'team_id', 'player_id', 'games_rating']
        missing = [col for col in required_player_cols if col not in player_stats_df.columns]
        if missing:
            raise ValueError(f"Missing columns in player_stats_df: {missing}")
        
        if not all(col in fixtures_df.columns for col in ['fixture_id', 'date']):
            raise ValueError("fixtures_df must contain 'fixture_id' and 'date'")

        # Create working copy and handle data types
        df = player_stats_df.copy()
        fixtures_df = fixtures_df.copy()
        
        # Convert IDs to consistent string type
        df['fixture_id'] = df['fixture_id'].astype(str)
        df['player_id'] = df['player_id'].astype(str)
        fixtures_df['fixture_id'] = fixtures_df['fixture_id'].astype(str)
        
        # Convert percentages and numeric columns
        for col in df.columns:
            if '%' in str(col):
                df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100

        # Merge with fixtures and sort chronologically
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'])
        df = df.merge(fixtures_df[['fixture_id', 'date']], on='fixture_id')
        df = df.sort_values(['player_id', 'date'])

        # Select numeric performance columns
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                    if col not in ['fixture_id', 'team_id', 'player_id', 'games_number', 'games_minutes']]

        # --- Calculate Rolling Averages ---
        rolling_features = []
        for player in df['player_id'].unique():
            player_data = df[df['player_id'] == player].copy()
            
            # Calculate rolling stats (excluding current match)
            for col in numeric_cols:
                player_data[col] = (  # Overwrite original column with rolling average
                    player_data[col].shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            
            # Add minutes-weighted features for key metrics
            if 'games_minutes' in player_data.columns:
                for metric in ['games_rating', 'passes_accuracy', 'shots_on']:
                    if metric in numeric_cols:
                        player_data[f'weighted_{metric}'] = (
                            (player_data[metric] * player_data['games_minutes']).shift(1)
                            .rolling(window, min_periods=1)
                            .sum() / 
                            player_data['games_minutes'].shift(1)
                            .rolling(window, min_periods=1)
                            .sum()
                        ).fillna(0)
            
            rolling_features.append(player_data)

        # Combine all players
        df = pd.concat(rolling_features).sort_values(['player_id', 'date'])

        # --- Create Derived Features ---
        # Performance composites
        if all(col in df.columns for col in ['shots_on', 'shots_total']):
            df['shot_accuracy'] = (
                df['shots_on'] / df['shots_total'].replace(0, 1)
            ).clip(0, 1)
        
        if all(col in df.columns for col in ['tackles_total', 'interceptions']):
            df['defensive_impact'] = df['tackles_total'] + df['interceptions']
        
        # Positional indicators
        if 'games_position' in df.columns:
            position_dummies = pd.get_dummies(df['games_position'], prefix='pos')
            df = pd.concat([df, position_dummies], axis=1)

        # --- Select Final Features ---
        keep_cols = ['fixture_id', 'team_id', 'player_id'] + numeric_cols
        keep_cols += [col for col in df.columns if col.startswith('weighted_')]
        keep_cols += [col for col in df.columns if col in ['shot_accuracy', 'defensive_impact']]
        keep_cols += [col for col in df.columns if col.startswith('pos_')]

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
            'injuries': os.path.join(processed_dir, f"injuries{self.processed_suffix}"),
            'team_stats': os.path.join(processed_dir, f"team_stats{self.processed_suffix}"),
            'player_stats': os.path.join(processed_dir, f"player_stats{self.processed_suffix}"),
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
        merged['league'] = league
        merged['season'] = season
        
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
    
    def _merge_datasets_optimized(self, fixtures, lineups, injuries, team_stats, player_stats, standings):
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
            .merge(injuries, on=['fixture_id', 'team_id'], how='left')
            .merge(player_stats.groupby(['fixture_id', 'team_id']).mean(), 
                on=['fixture_id', 'team_id'], how='left')
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
        def clean_column_name(name):
            """Remove _x/_y suffixes while preserving other underscores"""
            if name.endswith('_x') or name.endswith('_y'):
                return name[:-2]  # Remove last 2 characters
            return name
        
        merged.columns = [clean_column_name(col) for col in merged.columns]
        #merged['home_team_id'] = merged['home_team_id_x']
        #merged['away_team_id'] = merged['away_team_id_x']
        cols_to_drop = [
            'home_standings_team_id', 'away_standings_team_id', 'league',
            'home_home_team_id', 'away_away_team_id', 'away_team_id_x', 'home_team_id_x',
            'date_away', 'home_team_id_y', 'away_team_id_y', 'home_date', 'away_date'
        ]
        merged.drop(columns=[col for col in cols_to_drop if col in merged.columns], inplace=True)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        if self.config['verbose']:
            print("✅ Merge completed successfully!")
            print("Final shape:", merged.shape)
            print("Sample columns:", merged.columns.tolist())
        
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
            print(f"📊 Final dataset contains {len(final_data)} records")
        
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
    
    print(f"\nPipeline completed! Final dataset contains {len(final_dataset)} records")