
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Union, Dict, List
import logging

class ExtractPipeline:
    def __init__(self, log_dir="logs/extract", base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.extracted_dir = self.base_dir / "extracted"
        self.version_file = self.extracted_dir / ".extraction_versions.json"
        self.version_data = self._load_version_data()
        
        # Define extractors with their file patterns
        self.extractors = [
            ('fixture_events.json', self._extract_fixture_events),
            ('team_statistics.json', self._extract_team_statistics_multi),
            ('player_statistics.json', self._extract_player_statistics_multi),
            ('team_standings.json', self._extract_team_standings),
            ('injuries.json', self._extract_injuries_multi),
            ('lineups.json', self._extract_lineups_multi),
            ('odds.json', self._extract_odds_multi)
        ]

        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()



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
        log_file = os.path.join(self.log_dir, f"extract_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def _load_version_data(self) -> Dict:
        """Load the version tracking data"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_version_data(self) -> None:
        """Save the version tracking data"""
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.version_file, 'w') as f:
            json.dump(self.version_data, f, indent=2)

    def _file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of a file"""
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _file_needs_processing(self, filepath: Path, relative_path: str) -> bool:
        """Check if a file needs processing based on version data"""
        file_key = f"{relative_path}/{filepath.name}"
        
        # Get current file stats
        current_mtime = filepath.stat().st_mtime
        current_size = filepath.stat().st_size
        current_hash = self._file_hash(filepath)
        
        # Check if file is new or changed
        if relative_path not in self.version_data:
            return True
            
        if file_key not in self.version_data[relative_path]:
            return True
            
        stored_data = self.version_data[relative_path][file_key]
        return not (stored_data['mtime'] == current_mtime and
                   stored_data['size'] == current_size and
                   stored_data['hash'] == current_hash)

    def process_all_leagues_seasons(self) -> None:
        """Process all leagues and seasons with change detection"""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")

        processed_count = 0
        
        # Walk through all country/league/season directories
        for country in os.listdir(self.raw_dir):
            country_path = self.raw_dir / country
            if not country_path.is_dir():
                continue
                
            for league in os.listdir(country_path):
                league_path = country_path / league
                if not league_path.is_dir():
                    continue
                    
                for season in os.listdir(league_path):
                    season_path = league_path / season
                    if not season_path.is_dir():
                        continue
                        
                    # Create matching extracted directory
                    extracted_path = self.extracted_dir / country / league / season
                    extracted_path.mkdir(parents=True, exist_ok=True)
                    
                    self.logger.info(f"Checking {country}/{league}/{season}...")
                    processed_count += self._process_season(season_path, extracted_path)

        # Save version data after processing
        self._save_version_data()
        self.logger.info(f"\nProcessing complete. {processed_count} files were updated.")

    def _process_season(self, raw_path: Path, extracted_path: Path) -> int:
        """Process files in a single season directory"""
        processed_count = 0
        relative_path = str(raw_path.relative_to(self.raw_dir))
        
        # Initialize version tracking for this directory if needed
        if relative_path not in self.version_data:
            self.version_data[relative_path] = {}

        # Process each JSON file in the raw directory
        for json_file in raw_path.glob("*.json"):
            if not self._file_needs_processing(json_file, relative_path):
                continue
                
            # Find matching extractor
            for pattern, extractor in self.extractors:
                if json_file.name.startswith(pattern) or json_file.name == pattern:
                    self.logger.info(f"Processing {relative_path}/{json_file.name}...")
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        df = extractor(data)
                        if df is not None and not df.empty:
                            csv_file = extracted_path / f"{json_file.stem}.csv"
                            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                            
                            # Update version tracking
                            file_key = f"{relative_path}/{json_file.name}"
                            self.version_data[relative_path][file_key] = {
                                'mtime': json_file.stat().st_mtime,
                                'size': json_file.stat().st_size,
                                'hash': self._file_hash(json_file),
                                'processed': datetime.now().isoformat()
                            }
                            processed_count += 1
                            self.logger.info(f"Saved {csv_file.name}")
                        else:
                            self.logger.warning(f"No data extracted from {json_file.name}")
                    except Exception as e:
                        self.logger.error(f"Error processing {json_file.name}: {e}")
                    break
            else:
                self.logger.error(f"No extractor found for {json_file.name}, skipping")
        
        return processed_count

    def _extract_fixture_events(self, data: Union[Dict, List]) -> pd.DataFrame:
        """
        Extracts fixture details from the API response and returns a DataFrame.
        """
        fixtures_list = []

        if isinstance(data, dict):
            fixtures = data.get('response', [])
        elif isinstance(data, list):
            fixtures = data
        else:
            raise ValueError("Input data must be either a dictionary or list")

        for fixture in fixtures:
            try:
                # Ensure fixture is a dictionary before processing
                if not isinstance(fixture, dict):
                    continue                
                
                fixture_data = fixture.get('fixture', {})
                teams = fixture.get('teams', {})
                goals = fixture.get('goals', {})
                score = fixture.get('score', {})
                league_info = fixture.get('league', {})

                # Extracting fields with safe get()
                status = fixture_data.get('status', {})
                status_short = status.get('short', None)
                status_elapsed = status.get('elapsed', None)
                status_extra = status.get('extra', None)
                
                date_str = fixture_data.get('date', None)
                timestamp = fixture_data.get('timestamp', None)
                periods = fixture_data.get('periods', {})
                first_half = periods.get('first', None)
                second_half = periods.get('second', None)
                referee = fixture_data.get('referee', None)

                venue = fixture_data.get('venue', {})
                venue_name = venue.get('name', None)
                venue_city = venue.get('city', None)
                venue_id = venue.get('id', None)

                fixture_id = fixture_data.get('id', None)
                season = league_info.get('season', None)
                country = league_info.get('country', None)
                round_name = league_info.get('round', None)
                league_logo = league_info.get('logo', None)
                league_flag = league_info.get('flag', None)

                # Home team info
                home_team = teams.get('home', {})
                home_team_name = home_team.get('name', None)
                home_team_id = home_team.get('id', None)
                home_team_flag = home_team.get('logo', None)
                home_win = home_team.get('winner', None)
                home_goals = goals.get('home', None)
                home_score_ht = score.get('halftime', {}).get('home', None)
                home_score_ft = score.get('fulltime', {}).get('home', None)
                penalty_home = score.get('penalty', {}).get('home', None)

                # Away team info
                away_team = teams.get('away', {})
                away_team_name = away_team.get('name', None)
                away_team_id = away_team.get('id', None)
                away_team_flag = away_team.get('logo', None)
                away_win = away_team.get('winner', None)
                away_goals = goals.get('away', None)
                away_score_ht = score.get('halftime', {}).get('away', None)
                away_score_ft = score.get('fulltime', {}).get('away', None)
                penalty_away = score.get('penalty', {}).get('away', None)
                

                fixtures_list.append({
                    'fixture_id': fixture_id,
                    'date': date_str,
                    'timestamp': timestamp,
                    'status': status_short,
                    'maintime' : status_elapsed,
                    'first_half': first_half,
                    'second_half': second_half,
                    'extratime': status_extra,
                    'country': country,
                    'league_id': league_info.get('id', None),
                    'league_name': league_info.get('name', None),
                    'league_flag': league_flag,
                    'league_logo': league_logo,
                    'season': season,
                    'round': round_name,
                    'venue_name': venue_name,
                    'venue_city': venue_city,
                    'venue_id': venue_id,
                    'referee': referee,
                    'home_team': home_team_name,
                    'home_team_id': home_team_id,
                    'home_team_flag': home_team_flag,
                    'home_winner': home_win,
                    'away_team': away_team_name,
                    'away_team_id': away_team_id,
                    'away_team_flag': away_team_flag,
                    'away_winner': away_win,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'halftime_home': home_score_ht,
                    'halftime_away': away_score_ht,
                    'fulltime_home': home_score_ft,
                    'fulltime_away': away_score_ft,
                    'penalty_home': penalty_home,
                    'penalty_away': penalty_away
                })

            except Exception as e:
                self.logger.error(f"Error processing fixture: {e}")
                continue

        df = pd.DataFrame(fixtures_list)
        # Convert date string to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df


    def _extract_team_statistics(self, data: Dict) -> pd.DataFrame:
        """
        Extracts team statistics data from an API response and flattens it into a DataFrame.
        """
        team_stats_list = []

        # Get fixture ID from parameters if available
        fixture_id = data.get('parameters', {}).get('fixture', None)

        for team_stats in data.get('response', []):
            try:
                team = team_stats.get('team', {})
                team_id = team.get('id', None)
                team_name = team.get('name', None)
                team_logo = team.get('logo', None)

                # Handle missing or empty statistics
                statistics = team_stats.get('statistics', [])
                if statistics is None:
                    statistics = []

                # Convert list of stats to a dict
                stats_dict = {}
                for stat in statistics:
                    stat_type = stat.get('type', None)
                    stat_value = stat.get('value', None)
                    if stat_type:
                        stats_dict[stat_type] = stat_value

                # Build the row
                row = {
                    'fixture_id': fixture_id,
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_logo': team_logo
                }
                row.update(stats_dict)

                team_stats_list.append(row)

            except Exception as e:
                self.logger.error(f"Error processing team stats for fixture {fixture_id}: {e}")
                continue

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(team_stats_list)
        return df

    def _extract_team_statistics_multi(self, data: Dict) -> pd.DataFrame:
        import pandas as pd
        # If response is a list, process each; if dict, process just one
        if isinstance(data, list):
            dfs = [self._extract_team_statistics(r) for r in data]
            dfs = [df for df in dfs if not df.empty]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            return self._extract_team_statistics(data)


    def _extract_team_standings(self, data: Dict) -> pd.DataFrame:
        """
        Extracts standings data from an API response and flattens it into a DataFrame.

        Args:
            response (dict): The API response data, expected to be a dictionary containing the standings information.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted standings data.
        """

        flattened_data = []

        # Ensure 'response' is present and contains data
        if 'response' not in data or not data['response']:
            self.logger.warning("No standings data found in the response.")
            return pd.DataFrame()  # Return empty DataFrame if no standings data

        try:
            # Iterate through each standings group
            for round_data in data['response'][0]['league']['standings']:
                for team_data in round_data:
                    # Extract relevant data with safe dictionary access
                    team = team_data.get('team', {})
                    standings_info = {
                        'rank': team_data.get('rank'),
                        'team_id': team.get('id'),
                        'team_name': team.get('name'),
                        'team_logo': team.get('logo', None),
                        'points': team_data.get('points'),
                        'goals_diff': team_data.get('goalsDiff'),
                        'group': team_data.get('group', None),
                        'form': team_data.get('form', None),
                        'status': team_data.get('status'),
                        'description': team_data.get('description', None),
                        'played': team_data.get('all', {}).get('played'),
                        'wins': team_data.get('all', {}).get('win'),
                        'draws': team_data.get('all', {}).get('draw'),
                        'losses': team_data.get('all', {}).get('lose'),
                        'goals_for': team_data.get('all', {}).get('goals', {}).get('for'),
                        'goals_against': team_data.get('all', {}).get('goals', {}).get('against'),
                        'home_played': team_data.get('home', {}).get('played'),
                        'home_wins': team_data.get('home', {}).get('win'),
                        'home_draws': team_data.get('home', {}).get('draw'),
                        'home_losses': team_data.get('home', {}).get('lose'),
                        'home_goals_for': team_data.get('home', {}).get('goals', {}).get('for'),
                        'home_goals_against': team_data.get('home', {}).get('goals', {}).get('against'),
                        'away_played': team_data.get('away', {}).get('played'),
                        'away_wins': team_data.get('away', {}).get('win'),
                        'away_draws': team_data.get('away', {}).get('draw'),
                        'away_losses': team_data.get('away', {}).get('lose'),
                        'away_goals_for': team_data.get('away', {}).get('goals', {}).get('for'),
                        'away_goals_against': team_data.get('away', {}).get('goals', {}).get('against')
                    }

                    # Append the team's standings information to the flattened_data list
                    flattened_data.append(standings_info)

        except KeyError as e:
            self.logger.error(f"KeyError: {e} - Some expected keys are missing in the response.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {str(e)}")

        team_standings_df = pd.DataFrame(flattened_data)
        #team_standings_df.columns = team_standings_df.columns.str.lower().str.replace(' ', '_')

        # Convert the list of dictionaries into a DataFrame
        return team_standings_df


    def _extract_player_statistics(self, data: Dict) -> pd.DataFrame:
        """
        Extracts player statistics data from an API response and flattens it into a DataFrame.
        Enhanced with robust error handling and data validation.

        Args:
            response (dict): The API response data.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted player statistics data.
        """
        import pandas as pd

        # Initial response validation
        if not data or not isinstance(data, dict):
            self.logger.warning("Invalid response: Expected dictionary")
            return pd.DataFrame()

        # Get the fixture ID from the parameters with error handling
        fixture_id = None
        try:
            parameters = data.get('parameters', {}) or {}
            fixture_id = parameters.get('fixture')
        except Exception as e:
            self.logger.error(f"Error getting fixture ID: {e}")

        if not fixture_id:
            self.logger.warning("Warning: 'fixture' ID not found in the response parameters.")
            return pd.DataFrame()

        # Check response structure
        if 'response' not in data:
            self.logger.warning("No 'response' key found in the data")
            return pd.DataFrame()
        
        if not data['response']:
            #print("Empty player statistics data in the response")
            return pd.DataFrame()

        flattened_data = []

        # Process each team's data
        for team_data in data['response']:
            if not isinstance(team_data, dict):
                self.logger.warning("Skipping invalid team data (not a dictionary)")
                continue

            # Extract team information with defaults
            team = team_data.get('team', {}) or {}
            team_name = team.get('name', 'Unknown Team')
            team_id = team.get('id')

            # Process each player
            players = team_data.get('players', []) or []
            for player_data in players:
                if not isinstance(player_data, dict):
                    self.logger.warning("Skipping invalid player data (not a dictionary)")
                    continue

                try:
                    # Extract basic player info
                    player = player_data.get('player', {}) or {}
                    player_id = player.get('id')
                    player_name = player.get('name', 'Unknown Player')
                    player_photo = player.get('photo')

                    # Initialize player stats dictionary
                    player_stats = {
                        'fixture_id': fixture_id,
                        'team_name': team_name,
                        'team_id': team_id,
                        'player_id': player_id,
                        'player_name': player_name,
                        'player_photo': player_photo
                    }

                    # Process statistics
                    stats = player_data.get('statistics', []) or []
                    if not stats:
                        self.logger.warning(f"No statistics found for player {player_name} ({player_id})")
                        continue

                    for stat in stats:
                        if not isinstance(stat, dict):
                            continue
                            
                        # Flatten nested statistics
                        for stat_type, stat_value in stat.items():
                            if stat_value is None:
                                continue
                                
                            if isinstance(stat_value, dict):
                                # Handle nested statistics
                                for sub_key, sub_value in stat_value.items():
                                    if sub_value is not None:
                                        player_stats[f"{stat_type}_{sub_key}"] = sub_value
                            else:
                                # Handle flat statistics
                                player_stats[stat_type] = stat_value

                    flattened_data.append(player_stats)

                except Exception as e:
                    self.logger.error(f"Error processing player {player.get('id')}: {str(e)}")
                    continue

        # Create DataFrame if we have data
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            # Clean column names if needed
            # df.columns = df.columns.str.lower().str.replace(' ', '_')
            return df
        
        self.logger.warning("No valid player statistics data could be extracted")
        return pd.DataFrame()

    def _extract_player_statistics_multi(self, data: Dict) -> pd.DataFrame:
        """
        Handles both single and multiple player statistics responses.
        """
        import pandas as pd

        if not data:
            return pd.DataFrame()

        if isinstance(data, list):
            dfs = []
            for r in data:
                try:
                    if not r:  # Skip empty responses
                        continue
                    df = self._extract_player_statistics(r)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error processing player stats in multi: {e}")
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            return self._extract_player_statistics(data)


    def _extract_lineups(self, data: Dict) -> pd.DataFrame:
        """
        Extracts all fields from the fixtures/lineups API response and returns it as a DataFrame.
        Adds clear columns for coach id and name.
        """
        import pandas as pd

        # Initial response validation
        if not data or not isinstance(data, dict):
            self.logger.warning("Invalid data: Expected dictionary")
            return pd.DataFrame()
        
        # Check different possible response structures
        response_data = None
        
        # Structure 1: Direct response list
        if 'response' in data and isinstance(data['response'], list):
            response_data = data['response']
        # Structure 2: Nested under data (as mentioned in the error)
        elif 'data' in data and isinstance(data['data'], list):
            response_data = data['data']
        # Structure 3: Maybe the data is already the response list
        elif isinstance(data, list):
            response_data = data
        else:
            self.logger.warning("Invalid response structure: Could not find 'response' or 'data' key")
            return pd.DataFrame()

        if not response_data:
            self.logger.warning("Empty response data")
            return pd.DataFrame()

        lineup_data = []

        # Safely get fixture ID from parameters or try to extract from response
        fixture_id = None
        try:
            parameters = data.get('parameters', {}) or {}
            fixture_id = parameters.get('fixture')
            
            # If not in parameters, try to get from the first entry
            if not fixture_id and response_data:
                first_entry = response_data[0]
                if isinstance(first_entry, dict) and 'fixture' in first_entry:
                    fixture_id = first_entry['fixture'].get('id')
        except Exception as e:
            self.logger.error(f"Error getting fixture ID: {e}")

        if not fixture_id:
            self.logger.warning("Warning: 'fixture' ID not found in the response.")
            # We'll still process but fixture_id will be None

        for entry in response_data:
            if not isinstance(entry, dict):
                self.logger.warning("Skipping invalid lineup entry (not a dictionary)")
                continue
                
            try:
                # Try to get fixture ID from this specific entry if not already set
                entry_fixture_id = fixture_id
                if not entry_fixture_id and 'fixture' in entry:
                    entry_fixture_id = entry['fixture'].get('id')

                # Safely extract team info with defaults
                team = entry.get('team', {}) or {}
                team_colors = team.get('colors', {}) or {}
                player_colors = team_colors.get('player', {}) or {}
                gk_colors = team_colors.get('goalkeeper', {}) or {}
                
                team_id = team.get('id')
                team_name = team.get('name')
                team_logo = team.get('logo')

                # Safely extract coach info
                coach = entry.get('coach', {}) or {}
                coach_id = coach.get('id')
                coach_name = coach.get('name')
                coach_photo = coach.get('photo')

                formation = entry.get('formation')

                # Process starting lineup
                startXI = entry.get('startXI', []) or []
                for player in startXI:
                    if not isinstance(player, dict):
                        continue
                        
                    player_info = player.get('player', {}) or {}
                    lineup_data.append({
                        'fixture_id': entry_fixture_id,
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_logo': team_logo,
                        'team_colors_player_primary': player_colors.get('primary'),
                        'team_colors_player_number': player_colors.get('number'),
                        'team_colors_player_border': player_colors.get('border'),
                        'team_colors_goalkeeper_primary': gk_colors.get('primary'),
                        'team_colors_goalkeeper_number': gk_colors.get('number'),
                        'team_colors_goalkeeper_border': gk_colors.get('border'),
                        'coach_id': coach_id,
                        'coach_name': coach_name,
                        'coach_photo': coach_photo,
                        'formation': formation,
                        'player_id': player_info.get('id'),
                        'player_name': player_info.get('name'),
                        'player_number': player_info.get('number'),
                        'player_pos': player_info.get('pos'),
                        'player_grid': player_info.get('grid'),
                        'is_substitute': False
                    })

                # Process substitutes
                substitutes = entry.get('substitutes', []) or []
                for player in substitutes:
                    if not isinstance(player, dict):
                        continue
                        
                    player_info = player.get('player', {}) or {}
                    lineup_data.append({
                        'fixture_id': entry_fixture_id,
                        'team_id': team_id,
                        'team_name': team_name,
                        'team_logo': team_logo,
                        'team_colors_player_primary': player_colors.get('primary'),
                        'team_colors_player_number': player_colors.get('number'),
                        'team_colors_player_border': player_colors.get('border'),
                        'team_colors_goalkeeper_primary': gk_colors.get('primary'),
                        'team_colors_goalkeeper_number': gk_colors.get('number'),
                        'team_colors_goalkeeper_border': gk_colors.get('border'),
                        'coach_id': coach_id,
                        'coach_name': coach_name,
                        'coach_photo': coach_photo,
                        'formation': formation,
                        'player_id': player_info.get('id'),
                        'player_name': player_info.get('name'),
                        'player_number': player_info.get('number'),
                        'player_pos': player_info.get('pos'),
                        'player_grid': player_info.get('grid'),
                        'is_substitute': True
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing lineup entry: {e}")
                continue

        # Create DataFrame only if we have data
        if lineup_data:
            return pd.DataFrame(lineup_data)
        return pd.DataFrame()

    def _extract_lineups_multi(self, data: Dict) -> pd.DataFrame:
        """
        Handles both single and multiple lineup responses
        """
        import pandas as pd
        
        if not data:
            return pd.DataFrame()
            
        if isinstance(data, list):
            dfs = []
            for r in data:
                try:
                    df = self._extract_lineups(r)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error processing lineup in multi: {e}")
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            return self._extract_lineups(data)


    def _extract_injuries(self, data: Dict) -> pd.DataFrame:
        """
        Extracts injury data from the API response and returns it as a DataFrame.

        Args:
            response (dict): The API response containing injury data.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted injury data.

        Raises:
            ValueError: If the 'response' key is missing or is not a list.
        """
        # Ensure 'response' exists and is a list
        if 'response' not in data or not isinstance(data['response'], list):
            raise ValueError("Invalid API response structure. Expected 'response' key with a list of injuries.")

        # Extract injury data into a list of dictionaries
        injury_data = []

        for entry in data['response']:
            try:
                fixture = entry['fixture']
                player = entry['player']
                team = entry['team']
                league = entry['league']
                
                injury_info = {
                    'fixture_id': fixture.get('id'),
                    'date': fixture.get('date'),
                    'player_id': player.get('id'),
                    'player_name': player.get('name'),
                    'player_photo': player.get('photo'),
                    'type': player.get('type'),
                    'injury_reason': player.get('reason'),
                    'team_id': team.get('id'),
                    'team_name': team.get('name'),
                    'league_id': league.get('id'),
                    'league_name': league.get('name'),
                    'season': league.get('season')
                }
            
                injury_data.append(injury_info)
            except Exception as e:
                self.logger.error(f"Error processing injuries: {e}")
                continue

        # Convert the list of dictionaries into a DataFrame
        injuries_df = pd.DataFrame(injury_data)

        return injuries_df

    def _extract_injuries_multi(self, data: Dict) -> pd.DataFrame:
        import pandas as pd
        # If response is a list, process each; if dict, process just one
        if isinstance(data, list):
            dfs = [self._extract_injuries(r) for r in data]
            dfs = [df for df in dfs if not df.empty]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            return self._extract_injuries(data)

    def _extract_odds(self, response: Dict, bookmaker_id=8) -> pd.DataFrame:
        """
        Extracts odds data from an API response for a specific bookmaker and flattens it into a DataFrame.
        Specifically extracts various bets including 'Match Winner', 'Second Half Winner', 'Both Teams to Score', 
        'Both Teams to Score in First Half', 'Goals Over First Half', 'HT/FT Double', 'Double Chance', and 
        'Over/Under' with specific values (Over 2.5, Under 2.5, Over 4.5, Under 4.5).
        
        Args:
            response (dict): The API response data.
            bookmaker_id (int): The ID of the bookmaker to extract odds for (default is 8).

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted odds data.
        """
        import pandas as pd
        
        flattened_data = []

        # Ensure the 'response' key exists and contains data
        if 'response' not in response or not response['response']:
            print("No odds data found in the response.")
            return pd.DataFrame()  # Return empty DataFrame if no data

        # Iterate through matches in the response
        for match_data in response['response']:
            fixture_id = match_data.get('fixture', {}).get('id', None)
            if not fixture_id:
                print("Warning: 'fixture_id' not found in the response.")
                continue

            # Iterate through bookmakers in each match
            for bookmaker in match_data.get('bookmakers', []):
                if bookmaker.get('id') == bookmaker_id:
                    bookmaker_name = bookmaker.get('name', 'Unknown Bookmaker')

                    # Iterate through bets provided by the bookmaker
                    for bet in bookmaker.get('bets', []):
                        bet_name = bet.get('name', 'Unknown Bet')

                        # Only extract specific bet types
                        if bet_name in [
                            "Match Winner", 
                            "Second Half Winner", 
                            "First Half Winner",
                            "Both Teams to Score", 
                            "Both Teams to Score in First Half", 
                            "Goals Over/Under",
                            "Goals Over First Half", 
                            "HT/FT Double", 
                            "Double Chance"
                        ]:
                            for value_data in bet.get('values', []):
                                bet_value = value_data.get('value', 'Unknown Value')
                                bet_odd = value_data.get('odd', 'Unknown Odd')

                                odds_info = {
                                    'fixture_id': fixture_id,
                                    'bet_name': bet_name,
                                    'bet_value': bet_value,
                                    'bet_odd': bet_odd
                                }

                                # Append the odds information to the flattened data
                                flattened_data.append(odds_info)

        odds_df = pd.DataFrame(flattened_data)
        return odds_df       

    def _extract_odds_multi(self, data: Dict, bookmaker_id=8) -> pd.DataFrame:
        """
        Handles both single and multiple odds responses for a specific bookmaker.
        Extracts odds data from API responses and flattens it into a DataFrame.
        
        Args:
            data (dict or list): The API response data (single response or list of responses)
            bookmaker_id (int): The ID of the bookmaker to extract odds for (default is 8)

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted odds data.
        """
        import pandas as pd
        
        if not data:
            return pd.DataFrame()
        
        if isinstance(data, list):
            dfs = []
            for response in data:
                try:
                    df = self._extract_odds(response, bookmaker_id)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error processing odds in multi: {e}")
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            return self._extract_odds(bookmaker_id, data)





    