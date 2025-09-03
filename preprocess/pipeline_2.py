import os
import pandas as pd
from typing import Dict, Optional, List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from preprocess.process_and_engineer import process_fixture_events, process_injuries, process_lineups, process_player_statistics, process_team_standings, process_team_statistics

class FootballDataPipeline:
    """
    End-to-end pipeline for processing and merging football data.
    Uses unified process_* functions for each data type.
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
            'verbose': True,
            'lag_window': 3,  # Default window for rolling features
            'league_avg_goals': 1.5,  # Default average goals for feature engineering
            'min_fixtures': 5  # Minimum fixtures required for rolling stats
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Processing function mappings
        self.process_functions = {
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
        self.processed_suffix = "_processed.csv"
        
        # Cache for fixtures data needed by other processors
        self._fixtures_cache = {}

    def process_data(
        self,
        input_path: str,
        output_path: str,
        process_function: callable,
        data_type: str,
        **kwargs
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
            
            # Special handling for data types that need fixtures data
            if data_type in ['team_stats', 'player_stats']:
                fixtures_data = self._get_fixtures_for_stats(os.path.dirname(input_path))
                if fixtures_data is not None:
                    kwargs['fixtures_df'] = fixtures_data
            
            # Process data - handle standings differently
            if data_type == 'standings':
                fixtures_data = self._get_fixtures_for_stats(os.path.dirname(input_path))
                if fixtures_data is not None:
                    processed_data = process_function(fixtures_data, output_path=output_path)
                else:
                    raise ValueError("Could not load fixtures data for standings processing")
            else:
                processed_data = process_function(data, **kwargs)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save results (except for standings which handles its own saving)
            if data_type != 'standings':
                processed_data.to_csv(output_path, index=False)
            
            if self.config['verbose']:
                print(f"✅ [{data_type.upper()}] Processed data saved to {output_path}")
            
            return processed_data
        
        except Exception as e:
            if self.config['verbose']:
                print(f"❌ [{data_type.upper()}] Error processing {os.path.basename(input_path)}: {str(e)}")
            raise
    
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
    
    def process_all_data(self) -> None:
        """
        Process all raw data files into enhanced versions.
        """
        if self.config['verbose']:
            print("⚙️ Starting data processing pipeline...")
        
        # Use default data types if not specified
        data_types = self.config['data_types'] or list(self.process_functions.keys())
        
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
                    
                    # Get the appropriate process function
                    process_func = self.process_functions[data_type]
                    
                    # Add config parameters to kwargs if needed
                    kwargs = {}
                    if data_type == 'fixtures':
                        kwargs.update({
                            'lag_window': self.config['lag_window'],
                            'league_avg_goals': self.config['league_avg_goals'],
                            #'min_fixtures': self.config['min_fixtures']
                        })
                    
                    # Process data
                    self.process_data(
                        input_path=input_file,
                        output_path=output_file,
                        process_function=process_func,
                        data_type=data_type,
                        **kwargs
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
        
        # Merge data using the optimized method
        merged = self._merge_datasets_optimized(**data)
        
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
        Debuggable version with merge step verification
        """
        if self.config['verbose']:
            print("\n🔄 Starting optimized merge process...")
        
        # 1. Prepare base fixtures data (380 records)
        fixtures = fixtures.copy()
        team_stats = team_stats.copy()
        print("🔍 Fixtures columns:", fixtures.columns.tolist())
        
        # 2. Get home/away team references from fixtures
        team_ref = fixtures[['fixture_id', 'home_team_id', 'away_team_id']]
        print("🔍 Team reference columns:", team_ref.columns.tolist())
        
        # 3. Process lineups (760 records - 2 per fixture)
        print("\n🔍 Before lineups merge - Lineups columns:", lineups.columns.tolist())
        lineups = lineups.merge(team_ref, on='fixture_id', how='left')
        print("✅ After adding team ref to lineups - Columns:", lineups.columns.tolist())
        print("   Contains home_team_id:", 'home_team_id' in lineups.columns)
        
        # 4. Process player stats (aggregate by fixture and team)
        player_agg = player_stats.groupby(['fixture_id', 'team_id']).agg({
            'rolling_5_games_rating': 'mean',
            'rolling_5_shots_on': 'mean',
            'weighted_5_games_rating': 'mean',
            'pos_D': 'sum',
            'pos_F': 'sum',
            'pos_G': 'sum',
            'pos_M': 'sum'
        }).reset_index()
        
        # 5. Create complete team data
        print("\n🔍 Before team_data merge - Team stats columns:", team_stats.columns.tolist())
        team_data = team_stats.merge(injuries, on=['fixture_id', 'team_id'], how='left')
        print("✅ After merging injuries - Columns:", team_data.columns.tolist())
        
        team_data = team_data.merge(player_agg, on=['fixture_id', 'team_id'], how='left', suffixes=('', '_player'))
        print("✅ After merging player stats - Columns:", team_data.columns.tolist())
        
        # 6. Merge team data with lineups
        print("\n🔍 Before lineups merge - Team data columns:", team_data.columns.tolist())
        team_data = team_data.merge(lineups, on=['fixture_id', 'team_id'], how='left', suffixes=('', '_lineup'))
        print("✅ After merging lineups - Columns:", team_data.columns.tolist())
        print("   Contains home_team_id:", 'home_team_id' in team_data.columns)
        
        # 7. Split into home and away datasets
        print("\n🔍 Before split - Checking team_data:")
        print("   home_team_id present:", 'home_team_id' in team_data.columns)
        print("   away_team_id present:", 'away_team_id' in team_data.columns)
        
        try:
            home_data = team_data[team_data['team_id'] == team_data['home_team_id']].copy()
            away_data = team_data[team_data['team_id'] == team_data['away_team_id']].copy()
  
            
             # Rename all columns except fixture_id in home_data
            home_data = home_data.rename(columns={
                col: f'home_{col}' for col in home_data.columns if col != 'fixture_id'
            })
            home_data.drop(columns=['home_home_team_id', 'home_away_team_id'], inplace=True)

            # Rename all columns except fixture_id in away_data
            away_data = away_data.rename(columns={
                col: f'away_{col}' for col in away_data.columns if col != 'fixture_id'
            })
            away_data.drop(columns=['away_away_team_id', 'away_home_team_id'], inplace=True)
            print("✅ Successfully split into home/away data")
        except Exception as e:
            print(f"❌ Split failed: {str(e)}")
            print("Current columns:", team_data.columns.tolist())
            raise
        
        # 8. Final merge with fixtures
        print("\n🔍 Before final merge - Home data columns:", home_data.columns.tolist())
        merged = fixtures.merge(home_data, on='fixture_id')
        merged = merged.merge(away_data, on='fixture_id')
        merged['home_team_id'] = merged['home_team_id_x']
        merged['away_team_id'] = merged['away_team_id_x']
        print("✅ After final merge - Columns:", merged.columns.tolist())
        
        # 9. Add standings data
        standings_cols = ['team_id', 'rank', 'points', 'form', 'goals_diff']
        home_standings = standings[standings_cols].add_prefix('home_standings_')
        away_standings = standings[standings_cols].add_prefix('away_standings_')
        
        merged = merged.merge(home_standings, left_on='home_team_id', right_on='home_standings_team_id', how='left')
        merged = merged.merge(away_standings, left_on='away_team_id', right_on='away_standings_team_id', how='left')
        print("✅ After standings merge - Columns:", merged.columns.tolist())
        
        # 10. Cleanup
        cols_to_drop = [
            'home_standings_team_id', 'away_standings_team_id',
            'home_home_team_id', 'away_away_team_id', 'away_team_id_x', 'home_team_id_x',
            'date_away', 'home_team_id_y', 'away_team_id_y'
        ]
        merged.drop(columns=[col for col in cols_to_drop if col in merged.columns], inplace=True)
        
        print("\n🔍 Final columns:", merged.columns.tolist())
        print(f"✅ Merge completed! Final shape: {merged.shape}")
        
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
        processed_dir = self.config['processed_dir']
        leagues = self.config['leagues'] or [
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ]
        
        all_merged = []
        
        for league in leagues:
            league_path = os.path.join(processed_dir, league)
            seasons = self.config['seasons'] or [
                d for d in os.listdir(league_path)
                if os.path.isdir(os.path.join(league_path, d))
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
        os.makedirs(os.path.dirname(self.config['final_output']), exist_ok=True)
        final_data.to_csv(self.config['final_output'], index=False)
        
        if self.config['verbose']:
            print(f"\n💾 Final merged dataset saved to {self.config['final_output']}")
            print(f"📊 Final dataset contains {len(final_data)} records")
        
        return final_data
    
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
        'merged_dir': 'data/merged',
        'final_output': 'data/processed/final_merged_dataset.csv',
        'leagues': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
        'seasons': ['2021', '2022'],
        'verbose': True,
        'lag_window': 5,
        'league_avg_goals': 1.5,
        'min_fixtures': 5
    }
    
    # Create and run pipeline
    pipeline = FootballDataPipeline(config)
    final_dataset = pipeline.run_pipeline()
    
    print("\nPipeline completed successfully! �")