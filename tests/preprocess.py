
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings
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
    
    def _preprocess_and_feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process pipeline that:
        1. Preserves specified metadata columns untouched
        2. Creates rolling averages for all team statistics
        3. Optionally removes original features used to create derived features
        """
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # --- Outcome Column ---
        if all(col in df.columns for col in ['home_goals', 'away_goals']):
            df['outcome'] = np.select(
                condlist=[
                    df['home_goals'] > df['away_goals'],
                    df['home_goals'] < df['away_goals']
                ],
                choicelist=[1, 2],
                default=0
            )
        
        
        # --- Temporal Features ---
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
            df['season_week'] = df['date'].dt.isocalendar().week
            
            seasons = {
                1: 'winter', 2: 'winter', 3: 'spring',
                4: 'spring', 5: 'spring', 6: 'summer',
                7: 'summer', 8: 'summer', 9: 'autumn',
                10: 'autumn', 11: 'autumn', 12: 'winter'
            }
            df['season_part'] = df['date'].dt.month.map(seasons)
        
        df['is_derby'] = df.apply(lambda x: x['venue_city'] == x.get('away_city', ''), axis=1)

        # --- Simplified Days Since Last Game Calculation ---
        def calculate_days_since_last(df, team_col, new_col_name):
            # Sort by season, team and date
            df = df.sort_values(['season', team_col, 'date'])
            
            # Calculate days since last game within season
            df[new_col_name] = (
                df.groupby(['season', team_col])['date']
                .diff()
                .dt.days
            )
            
            # Fill first game of each season with fixed value (31 days)
            df[new_col_name] = df[new_col_name].fillna(31)
            
            return df
        
        if all(col in df.columns for col in ['home_team_id', 'date', 'season']):
            df = calculate_days_since_last(df, 'home_team_id', 'days_since_last_home')
        
        if all(col in df.columns for col in ['away_team_id', 'date', 'season']):
            df = calculate_days_since_last(df, 'away_team_id', 'days_since_last_away')
        
        
        # --- Protected Metadata Columns ---
        protected_metadata = [
            'fixture_id', 'date', 'league_id', 'league_name', 'league_flag',
            'league_logo', 'season', 'round', 'venue_name', 'venue_city',
            'venue_id', 'referee', 'status', 'home_team', 'home_team_id',
            'home_team_flag', 'away_team', 'away_team_id',
            'away_team_flag', 'home_goals', 'away_goals',
            'halftime_home', 'halftime_away', 
            'fulltime_home', 'fulltime_away', 'outcome', 'month', 'is_weekend',
            'season_part', 'season_week', 'days_since_last_home', 'days_since_last_away', 'is_derby'
        ]
        
        # Clean column names to match protected list
        df.columns = (
            df.columns.str.lower()
            .str.replace(r'[^a-z0-9_]', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.strip('_')
        )
        
        # Only keep protected columns that actually exist
        protected_metadata = [col for col in protected_metadata if col in df.columns]
        
        # === Step 2: Feature Engineering ===
        # Track original features used for derived features
        original_features = set()
        derived_features = []
        
        # Match-level features
        if all(col in df.columns for col in ['home_goals', 'away_goals']):
            df['goal_difference'] = df['home_goals'] - df['away_goals']
            df['total_goals'] = df['home_goals'] + df['away_goals']
            derived_features.extend(['goal_difference', 'total_goals'])
            #original_features.update(['home_goals', 'away_goals'])
        
        # Example: Shot Accuracy
        if all(col in df.columns for col in ['home_shots_on_goal', 'home_total_shots']):
            df['home_shot_accuracy'] = df['home_shots_on_goal'] / df['home_total_shots'].replace(0, 1)
            df['away_shot_accuracy'] = df['away_shots_on_goal'] / df['away_total_shots'].replace(0, 1)
            derived_features.extend(['home_shot_accuracy', 'away_shot_accuracy'])
            original_features.update(['home_shots_on_goal', 'home_total_shots', 
                                        'away_shots_on_goal', 'away_total_shots'])
        
        # Add other derived features similarly...
        if all(col in df.columns for col in ['home_goals', 'home_total_shots']):
            df['home_shot_efficiency'] = df['home_goals'] / df['home_total_shots'].replace(0, 1)
            df['away_shot_efficiency'] = df['away_goals'] / df['away_total_shots'].replace(0, 1)
            derived_features.extend(['home_shot_efficiency', 'away_shot_efficiency'])
            original_features.update(['home_goals', 'away_goals'])

        if all(col in df.columns for col in ['home_shots_insidebox', 'home_total_shots']):
            df['home_box_threat'] = df['home_shots_insidebox'] / df['home_total_shots'].replace(0, 1)
            df['away_box_threat'] = df['away_shots_insidebox'] / df['away_total_shots'].replace(0, 1)
            derived_features.extend(['home_box_threat', 'away_box_threat'])
            original_features.update(['home_shots_insidebox', 'away_shots_insidebox'])

        if all(col in df.columns for col in ['home_shots_outsidebox', 'home_total_shots']):
            df['home_longrange_threat'] = df['home_shots_outsidebox'] / df['home_total_shots'].replace(0, 1)
            df['away_longrange_threat'] = df['away_shots_outsidebox'] / df['away_total_shots'].replace(0, 1)
            derived_features.extend(['home_longrange_threat', 'away_longrange_threat'])
            original_features.update(['home_shots_outsidebox', 'away_shots_outsidebox'])
            
        # Enhanced Defensive metrics
        if all(col in df.columns for col in ['home_goals_conceded', 'home_shots_on_target_against']):
            df['home_defensive_efficiency'] = 1 - (df['home_goals_conceded'] / df['home_shots_on_target_against'].replace(0, 1))
            df['away_defensive_efficiency'] = 1 - (df['away_goals_conceded'] / df['away_shots_on_target_against'].replace(0, 1))
            derived_features.extend(['home_defensive_efficiency', 'away_defensive_efficiency'])


        if all(col in df.columns for col in ['home_fouls', 'home_blocked_shots']):
            df['home_defensive_pressure'] = df['home_fouls'] + df['home_blocked_shots']
            df['away_defensive_pressure'] = df['away_fouls'] + df['away_blocked_shots']
            derived_features.extend(['home_defensive_pressure', 'away_defensive_pressure'])
            original_features.update(['home_fouls', 'away_fouls', 'home_blocked_shots', 'away_blocked_shots'])

        if all(col in df.columns for col in ['home_blocked_shots', 'home_shots_on_goal', 'home_shots_off_goal']):
            df['home_clearance_efficiency'] = (df['home_blocked_shots'] / (df['home_shots_on_goal'] + df['home_shots_off_goal']).replace(0, 1))
            df['away_clearance_efficiency'] = (df['away_blocked_shots'] / (df['away_shots_on_goal'] + df['away_shots_off_goal']).replace(0, 1))
            derived_features.extend(['home_clearance_efficiency', 'away_clearance_efficiency'])
            original_features.update(['home_shots_off_goal', 'away_shots_off_goal'])

        # Enhanced Goalkeeper metrics
        if all(col in df.columns for col in ['home_goalkeeper_saves', 'home_shots_on_target_against']):
            df['home_save_percentage'] = df['home_goalkeeper_saves'] / df['away_shots_insidebox'].replace(0, 1)
            df['away_save_percentage'] = df['away_goalkeeper_saves'] / df['home_shots_insidebox'].replace(0, 1)
            derived_features.extend(['home_save_percentage', 'away_save_percentage'])
            original_features.update(['home_goalkeeper_saves', 'away_goalkeeper_saves'])

        # Enhanced Possession metrics
        if all(col in df.columns for col in ['home_passes_accurate', 'home_total_passes']):
            df['home_pass_accuracy'] = df['home_passes_accurate'] / df['home_total_passes'].replace(0, 1)
            df['away_pass_accuracy'] = df['away_passes_accurate'] / df['away_total_passes'].replace(0, 1)
            derived_features.extend(['home_pass_accuracy', 'away_pass_accuracy'])
            original_features.update(['home_passes_accurate', 'away_passes_accurate', 'home_total_passes', 'away_total_passes'])

        if all(col in df.columns for col in ['home_total_passes', 'home_fouls']):
            df['home_press_resistance'] = df['home_total_passes'] / (df['home_fouls'] + 1).replace(0, 1)
            df['away_press_resistance'] = df['away_total_passes'] / (df['away_fouls'] + 1).replace(0, 1)
            derived_features.extend(['home_press_resistance', 'away_press_resistance'])
            original_features.update(['home_fouls', 'away_fouls'])
        
        # === Step 3: Calculate Rolling Averages ===
        # --- Rolling Averages ---
        if 'date' in df.columns:
            # Calculate team-specific rolling averages using your existing function
            team_stats_cols = [
                col for col in df.select_dtypes(include=['number']).columns
                if col not in protected_metadata 
                and col not in ['goal_difference', 'total_goals']
            ]
            df = self._calculate_rolling_averages(df, team_stats_cols + derived_features)
            
            # Calculate general rolling averages for match-level features
            df = df.sort_values('date')
            for window in self.config['rolling_windows']:
                # General rolling goal difference
                df[f'goal_difference_rolling_{window}'] = (
                    df['goal_difference']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                
                # General rolling total goals
                df[f'total_goals_rolling_{window}'] = (
                    df['total_goals']
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
        


        
        # === Step 5: Column Selection ===
        # Get all rolling columns
        rolling_cols = [col for col in df.columns if '_rolling_' in col]
        
        if self.config.get('drop_original_features', False):
            # Option 1: Keep only derived rolling features + metadata
            # Identify which rolling columns come from derived features
            derived_rolling_cols = [
                col for col in rolling_cols 
                if any(base in derived_features 
                    for base in [col.split('_rolling_')[0]])
            ]
            
            # Also keep playing style columns
            playing_style_cols = [col for col in df.columns if 'playing_style' in col]
            
            final_cols = protected_metadata + derived_rolling_cols + playing_style_cols
        else:
            # Option 2: Keep all rolling features + metadata
            final_cols = protected_metadata + rolling_cols + [
                col for col in df.columns if 'playing_style' in col
            ]
        
        # Ensure we only keep columns that exist
        final_cols = [col for col in final_cols if col in df.columns]
        
        return df[final_cols]

    def _calculate_rolling_averages(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Calculate rolling averages without dropping any columns
        """
        df = df.sort_values(['season', 'date'])
        first_season = df['season'].min()
        
        for window in self.config['rolling_windows']:
            for feature in features:
                if feature.startswith('home_'):
                    rolling_col = f"{feature}_rolling_{window}"
                    df[rolling_col] = (
                        df.groupby(['season', 'home_team_id'])[feature]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    )
                
                elif feature.startswith('away_'):
                    rolling_col = f"{feature}_rolling_{window}"
                    df[rolling_col] = (
                        df.groupby(['season', 'away_team_id'])[feature]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    )
        
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

                all_data['days_since_last_home'] = all_data.groupby('home_team_id')['date'].diff().dt.days
                all_data['days_since_last_away'] = all_data.groupby('away_team_id')['date'].diff().dt.days
                # Fill NaN values (for first game of each team) with 0 or some default value
                all_data['days_since_last_home'] = all_data['days_since_last_home'].fillna(0)
                all_data['days_since_last_away'] = all_data['days_since_last_away'].fillna(0)
                num_cols = all_data.select_dtypes(include=np.number).columns
                all_data[num_cols] = all_data[num_cols].fillna(0)
        
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
        'drop_original_features': True # Drop original features after rolling averages
    }
    
    pipeline = FootballDataPipeline(config)
    final_data = pipeline.run_pipeline()