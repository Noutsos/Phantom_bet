import os
import pandas as pd
from typing import Callable, Dict, Optional, List, Tuple

# Preprocessing functions (import from your modules)
#from preprocess.preprocess_fixtures import preprocess_fixtures, feature_engineer_fixtures
#from preprocess.preprocess_injuries import preprocess_injuries, feature_engineer_injuries
#from preprocess.preprocess_lineups import preprocess_lineups, feature_engineer_lineups
#from preprocess.preprocess_player_stats import preprocess_player_stats, feature_engineer_player_stats
#from preprocess.preprocess_standings import preprocess_team_standings, feature_engineer_team_standings
#from preprocess.preprocess_team_stats import preprocess_team_stats, feature_engineer_team_stats
from preprocess.process_and_engineer import process_fixture_events, process_injuries, process_lineups, process_player_statistics, process_team_standings, process_team_statistics

class FootballDataPipeline:
    """
    End-to-end pipeline for processing and merging football data.
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
            'leagues': None,  # None means process all available
            'seasons': None,  # None means process all available
            'data_types': None,  # None means process all types
            'verbose': True
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Function mappings
        self.preprocess_functions = {
            "fixtures": process_fixture_events,
            "injuries": process_injuries,
            "lineups": process_lineups,
            "player_stats": process_player_statistics,
            "standings": process_team_standings,
            "team_stats": process_team_statistics,
        }
        
        
        # File name mappings
        self.file_names = {
            "fixtures": "fixture_events.csv",
            "injuries": "injuries.csv",
            "lineups": "lineups.csv",
            "player_stats": "player_statistics.csv",
            "standings": "team_standings.csv",
            "team_stats": "team_statistics.csv",
        }
        
        # Processed file name suffixes
        self.processed_suffix = "_enhanced.csv"
    
    def process_data(
        self,
        input_path: str,
        output_path: str,
        preprocess_function: Callable[[pd.DataFrame], pd.DataFrame],
        data_type: str,
    ) -> pd.DataFrame:
        """
        Modular data processing pipeline for a single file.
        """
        try:
            # Normalize paths
            input_path = os.path.normpath(input_path)
            output_path = os.path.normpath(output_path)
            
            # Load data
            data = pd.read_csv(input_path)
            
            # Preprocess + feature engineering
            processed_data = preprocess_function(preprocess_function(data))
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save results
            processed_data.to_csv(output_path, index=False)
            
            if self.config['verbose']:
                print(f"✅ [{data_type.upper()}] Processed data saved to {output_path}")
            
            return processed_data
        
        except Exception as e:
            if self.config['verbose']:
                print(f"❌ [{data_type.upper()}] Error processing {os.path.basename(input_path)}: {str(e)}")
            raise
    
    def process_all_data(self) -> None:
        """
        Process all raw data files into enhanced versions.
        """
        if self.config['verbose']:
            print("⚙️ Starting data processing pipeline...")
        
        # Use default data types if not specified
        data_types = self.config['data_types'] or list(self.preprocess_functions.keys())
        
        # Find leagues/seasons (if not specified)
        leagues = self.config['leagues'] or [
            d for d in os.listdir(self.config['extracted_dir'])
            if os.path.isdir(os.path.join(self.config['extracted_dir'], d))
        ]
        
        for league in leagues:
            league_path = os.path.join(self.config['extracted_dir'], league)
            seasons = self.config['seasons'] or [
                d for d in os.listdir(league_path)
                if os.path.isdir(os.path.join(league_path, d))
            ]
            
            for season in seasons:
                season_path = os.path.join(league_path, season)
                
                for data_type in data_types:
                    input_file = os.path.join(season_path, self.file_names[data_type])
                    output_file = os.path.join(
                        self.config['processed_dir'],
                        league,
                        season,
                        f"{data_type}{self.processed_suffix}"
                    )
                    
                    if not os.path.exists(input_file):
                        if self.config['verbose']:
                            print(f"⚠️ [{data_type.upper()}] File not found: {input_file}")
                        continue
                    
                    # Process data
                    self.process_data(
                        input_path=input_file,
                        output_path=output_file,
                        preprocess_function=self.preprocess_functions[data_type],
                        data_type=data_type,
                    )
        
        if self.config['verbose']:
            print("✅ Data processing completed successfully!")
    
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
        
        # Merge data
        merged = self._merge_datasets(**data)
        
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
    
    def _merge_datasets(
        self,
        fixtures: pd.DataFrame,
        lineups: pd.DataFrame,
        injuries: pd.DataFrame,
        team_stats: pd.DataFrame,
        player_stats: pd.DataFrame,
        standings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Internal method to merge all datasets.
        """
        # Data Validation
        def validate_columns(df, df_name, required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {df_name}: {missing}")
        
        # Validate each dataset
        validate_columns(fixtures, 'fixtures', ['fixture_id', 'date', 'home_team_id', 'away_team_id'])
        validate_columns(standings, 'standings', ['team_id', 'points'])
        
        # Convert dates to datetime
        fixtures['date'] = pd.to_datetime(fixtures['date'])
        
        # Start with fixtures as base
        merged = fixtures.copy()
        
        # ======================
        # 1. MERGE STANDINGS DATA
        # ======================
        # Home team standings
        merged = pd.merge(
            merged,
            standings.add_prefix('home_standings_'),
            left_on='home_team_id',
            right_on='home_standings_team_id',
            how='left'
        )
        
        # Away team standings
        merged = pd.merge(
            merged,
            standings.add_prefix('away_standings_'),
            left_on='away_team_id',
            right_on='away_standings_team_id',
            how='left'
        )
        
        # ======================
        # 2. MERGE LINEUPS
        # ======================
        if not lineups.empty:
            # Home lineups - filter starters only
            home_starters = lineups[
                (lineups['team_id'].isin(merged['home_team_id']))
            ]
            
            # Count home starters
            home_lineup_counts = home_starters.groupby(['fixture_id', 'team_id']).size().reset_index(name='home_starters_count')
            merged = pd.merge(
                merged,
                home_lineup_counts,
                left_on=['fixture_id', 'home_team_id'],
                right_on=['fixture_id', 'team_id'],
                how='left'
            ).drop('team_id', axis=1)
            
            # Away lineups - filter starters only
            away_starters = lineups[
                (lineups['team_id'].isin(merged['away_team_id'])) 
            ]
            
            # Count away starters
            away_lineup_counts = away_starters.groupby(['fixture_id', 'team_id']).size().reset_index(name='away_starters_count')
            merged = pd.merge(
                merged,
                away_lineup_counts,
                left_on=['fixture_id', 'away_team_id'],
                right_on=['fixture_id', 'team_id'],
                how='left'
            ).drop('team_id', axis=1)
        
        # ======================
        # 3. MERGE TEAM STATS
        # ======================
        if not team_stats.empty:
            # Home team stats
            merged = pd.merge(
                merged,
                team_stats.add_prefix('home_teamstats_'),
                left_on=['fixture_id', 'home_team_id'],
                right_on=['home_teamstats_fixture_id', 'home_teamstats_team_id'],
                how='left'
            )
            
            # Away team stats
            merged = pd.merge(
                merged,
                team_stats.add_prefix('away_teamstats_'),
                left_on=['fixture_id', 'away_team_id'],
                right_on=['away_teamstats_fixture_id', 'away_teamstats_team_id'],
                how='left'
            )
        
        # ======================
        # 4. MERGE INJURIES
        # ======================
        if not injuries.empty:
            try:
                # Count injuries by fixture and team
                injuries_count = injuries.groupby(['fixture_id', 'team_id']).size().reset_index(name='injury_count')
                
                # Home injuries merge with proper fallback
                merged = pd.merge(
                    merged,
                    injuries_count.add_prefix('home_injury_'),
                    left_on=['fixture_id', 'home_team_id'],
                    right_on=['home_injury_fixture_id', 'home_injury_team_id'],
                    how='left'
                )
                
                # Drop merge keys if they exist
                merge_keys = ['home_injury_fixture_id', 'home_injury_team_id']
                merged = merged.drop(columns=[col for col in merge_keys if col in merged.columns])
                
                # Away injuries merge with proper fallback
                merged = pd.merge(
                    merged,
                    injuries_count.add_prefix('away_injury_'),
                    left_on=['fixture_id', 'away_team_id'],
                    right_on=['away_injury_fixture_id', 'away_injury_team_id'],
                    how='left'
                )
                
                # Drop merge keys if they exist
                merge_keys = ['away_injury_fixture_id', 'away_injury_team_id']
                merged = merged.drop(columns=[col for col in merge_keys if col in merged.columns])
                
                # Ensure injury count columns exist and fill NA
                if 'home_injury_count' not in merged.columns:
                    merged['home_injury_count'] = 0
                else:
                    merged['home_injury_count'] = merged['home_injury_count'].fillna(0)
                    
                if 'away_injury_count' not in merged.columns:
                    merged['away_injury_count'] = 0
                else:
                    merged['away_injury_count'] = merged['away_injury_count'].fillna(0)
                    
            except Exception as e:
                print(f"Error merging injuries data: {str(e)}")
                # Add default columns if merge fails
                merged['home_injury_count'] = 0
                merged['away_injury_count'] = 0
        else:
            # Add default columns if no injuries data
            merged['home_injury_count'] = 0
            merged['away_injury_count'] = 0
        
        # ======================
        # 5. MERGE PLAYER STATS
        # ======================
        if not player_stats.empty:
            # Aggregate player stats to team level
            player_stats_agg = player_stats.groupby(['fixture_id', 'team_id']).agg({
                'games_rating': 'mean',
                'goals_total': 'sum',
                'passes_accuracy': 'mean',
                'tackles_total': 'sum',
                'shots_total': 'sum',
                'shots_on': 'sum'
            }).reset_index()
            
            # Home player stats
            merged = pd.merge(
                merged,
                player_stats_agg.add_prefix('home_player_'),
                left_on=['fixture_id', 'home_team_id'],
                right_on=['home_player_fixture_id', 'home_player_team_id'],
                how='left'
            ).drop(['home_player_fixture_id', 'home_player_team_id'], axis=1)
            
            # Away player stats
            merged = pd.merge(
                merged,
                player_stats_agg.add_prefix('away_player_'),
                left_on=['fixture_id', 'away_team_id'],
                right_on=['away_player_fixture_id', 'away_player_team_id'],
                how='left'
            ).drop(['away_player_fixture_id', 'away_player_team_id'], axis=1)
        
        # Clean up duplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        # Fill NA values for injury counts
        if not injuries.empty:
            merged['home_injury_count'] = merged['home_injury_count'].fillna(0)
            merged['away_injury_count'] = merged['away_injury_count'].fillna(0)
        
        return merged
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        Merge all processed data across all leagues and seasons.
        
        Returns:
            Final merged DataFrame containing all data
        """
        if self.config['verbose']:
            print("\n🔗 Starting data merging pipeline...")
        
        # Find all processed leagues and seasons
        processed_dir = self.config['processed_dir']
        all_merged = []
        
        leagues = [
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ]
        
        for league in leagues:
            league_path = os.path.join(processed_dir, league)
            seasons = [
                d for d in os.listdir(league_path)
                if os.path.isdir(os.path.join(league_path, d))
            ]
            
            for season in seasons:
                try:
                    merged_season = self.merge_data_for_season(league, season)
                    all_merged.append(merged_season)
                    
                    if self.config['verbose']:
                        print(f"✅ Successfully merged {league} {season} "
                              f"({len(merged_season)} records)")
                except Exception as e:
                    if self.config['verbose']:
                        print(f"❌ Failed to merge {league} {season}: {str(e)}")
                    continue
        
        if not all_merged:
            raise ValueError("No data was successfully merged")
        
        # Combine all seasons
        final_merged = pd.concat(all_merged, ignore_index=True)
        
        # Save final dataset
        os.makedirs(os.path.dirname(self.config['final_output']), exist_ok=True)
        final_merged.to_csv(self.config['final_output'], index=False)
        
        if self.config['verbose']:
            print(f"\n🎉 Successfully created final merged dataset at {self.config['final_output']}")
            print(f"Total records: {len(final_merged)}")
        
        return final_merged
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run the complete pipeline (processing + merging).
        
        Returns:
            Final merged DataFrame
        """
        # Step 1: Process all raw data
        self.process_all_data()
        
        # Step 2: Merge all processed data
        final_data = self.merge_all_data()
        
        
        return final_data


if __name__ == "__main__":
    # Example configuration (can customize as needed)
    config = {
        'extracted_dir': 'data/extracted',
        'processed_dir': 'data/processed',
        'merged_dir': 'data/processed/merged',
        'final_output': 'data/processed/final_merged_dataset.csv',
        'leagues': ['Serie A'],  # Optional: specify leagues
        'seasons': ['2021'],  # Optional: specify seasons
        'verbose': True
    }
    
    # Create and run pipeline
    pipeline = FootballDataPipeline(config)
    final_dataset = pipeline.run_pipeline()
    
    
    print("\nPipeline completed successfully! 🚀")