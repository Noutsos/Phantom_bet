import http.client
import json
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import socket
import pandas as pd
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Dict, List, Optional, Union, Set, Any
from src.utils import LEAGUES  # Assuming LEAGUES is defined in utils.py

API_HOST = "v3.football.api-sports.io"
API_KEY = "25c02ce9f07df0edc1e69866fbe7d156"

class RateLimiter:
    def __init__(self, daily_limit=100, per_minute_limit=10):
        self.daily_limit = daily_limit
        self.per_minute_limit = per_minute_limit
        self.remaining_daily = daily_limit
        self.remaining_per_minute = per_minute_limit
        self.minute_window_start = time.time()

    def check_rate_limit(self):
        """Check and handle rate limits before making a request."""
        current_time = time.time()

        # Reset per-minute window if a minute has passed
        elapsed = current_time - self.minute_window_start
        if elapsed >= 60:
            self.remaining_per_minute = self.per_minute_limit
            self.minute_window_start = current_time

        if self.remaining_daily <= 0:
            raise Exception("Daily API request limit reached.")

        if self.remaining_per_minute <= 0:
            sleep_time = 60 - elapsed
            if sleep_time > 0:
                print(f"Per-minute rate limit reached. Waiting for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            # Reset after waiting
            self.remaining_per_minute = self.per_minute_limit
            self.minute_window_start = time.time()

        # Decrement counters for this request
        self.remaining_daily -= 1
        self.remaining_per_minute -= 1

    def update_rate_limits(self, headers):
        """
        Update rate limits from response headers (expects a dict).
        Example header keys: 'x-ratelimit-requests-remaining', 'x-ratelimit-remaining'
        """
        if headers is None:
            return
        # Lowercase all header keys for case-insensitive access
        headers = {k.lower(): v for k, v in headers.items()}
        if 'x-ratelimit-requests-remaining' in headers:
            try:
                self.remaining_daily = int(headers['x-ratelimit-requests-remaining'])
            except Exception:
                pass
        if 'x-ratelimit-remaining' in headers:
            try:
                self.remaining_per_minute = int(headers['x-ratelimit-remaining'])
            except Exception:
                pass


rate_limiter = RateLimiter()

class FootballDataCollector:
    def __init__(self, api_key, base_path="data/raw", log_dir="logs"):
        self.API_HOST = "v3.football.api-sports.io"
        self.API_KEY = api_key
        self.base_path = base_path
        self.rate_limiter = RateLimiter()
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir

        
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
        log_file = os.path.join(self.log_dir, f"collection_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")   
    
    def get_league_info(self, league_id):
        """Find which country a league belongs to and its metadata"""
        league_id = str(league_id)
        
        # First check top-level leagues
        for country, leagues in LEAGUES.items():
            if isinstance(leagues, dict) and league_id in leagues:
                return country, leagues[league_id]
        
        # Check nested structures (like 'Top 5 European Leagues')
        for category, countries in LEAGUES.items():
            if isinstance(countries, dict):
                for country_name, leagues in countries.items():
                    if isinstance(leagues, dict) and league_id in leagues:
                        return country_name, leagues[league_id]
        
        raise ValueError(f"League ID {league_id} not found in any country")

    def calculate_season_dates(self, country_name: str, league_info: Dict, season_year: int) -> tuple:
        """Calculate automatic season dates based on league characteristics"""
        start_month = league_info.get('start_month', 8)
        start_date = datetime(season_year, start_month, 1).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
                   relativedelta(months=+league_info['season_months'])).strftime('%Y-%m-%d')
        season_name = f"{season_year}"
        return season_name, start_date, end_date
    
    def make_api_request(self, endpoint):
        """Make API request with rate limiting and logging"""
        self.logger.info(f"Making API request: {endpoint}")
        
        conn = http.client.HTTPSConnection(self.API_HOST)
        headers = {
            'x-rapidapi-host': self.API_HOST,
            'x-rapidapi-key': self.API_KEY
        }

        self.rate_limiter.check_rate_limit()

        try:
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            try:
                response_data = json.loads(data)
                self.logger.info(f"API response: {response_data.get('results', 0)} results")
            except Exception as e:
                self.logger.error(f"Failed to parse JSON: {e}")
                raise

            if res.status == 200:
                self.rate_limiter.update_rate_limits(dict(res.getheaders()))
                return response_data
            else:
                self.logger.error(f"API request failed with status {res.status}")
                raise Exception(f"API request failed with status {res.status}")
                
        except Exception as e:
            self.logger.error(f"API request failed: {endpoint} - {str(e)}")
            raise
        finally:
            conn.close()
    
    # API endpoint methods
    def fetch_fixture_events(self, league_id: int, season: int, 
                           start_date: str, end_date: str) -> Dict:
        endpoint = f"/fixtures?league={league_id}&season={season}&from={start_date}&to={end_date}"
        return self.make_api_request(endpoint)

    def fetch_team_statistics(self, fixture_id: int) -> Dict:
        endpoint = f"/fixtures/statistics?fixture={fixture_id}"
        return self.make_api_request(endpoint)

    def fetch_player_statistics(self, fixture_id: int) -> Dict:
        endpoint = f"/fixtures/players?fixture={fixture_id}"
        return self.make_api_request(endpoint)

    def fetch_odds(self, fixture_id: int, bookmaker_id: int = 8) -> Dict:
        endpoint = f"/odds?bookmaker={bookmaker_id}&fixture={fixture_id}"
        return self.make_api_request(endpoint)

    def fetch_team_standings(self, league_id: int, season: int) -> Dict:
        endpoint = f"/standings?league={league_id}&season={season}"
        return self.make_api_request(endpoint)

    def fetch_lineups(self, fixture_id: int) -> Dict:
        endpoint = f"/fixtures/lineups?fixture={fixture_id}"
        return self.make_api_request(endpoint)

    def fetch_injuries(self, fixture_id: int) -> Dict:
        endpoint = f"/injuries?fixture={fixture_id}"
        return self.make_api_request(endpoint)

    # Bulk fetch methods
    def fetch_all_team_statistics(self, fixture_ids: List[int]) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=5) as executor:
            return list(executor.map(self.fetch_team_statistics, fixture_ids))

    def fetch_all_player_statistics(self, fixture_ids: List[int]) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=5) as executor:
            return list(executor.map(self.fetch_player_statistics, fixture_ids))

    def fetch_all_odds(self, fixture_ids: List[int]) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=5) as executor:
            return list(executor.map(self.fetch_odds, fixture_ids))

    def fetch_all_lineups(self, fixture_ids: List[int]) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=5) as executor:
            return list(executor.map(self.fetch_lineups, fixture_ids))

    def fetch_all_injuries(self, fixture_ids: List[int]) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=5) as executor:
            return list(executor.map(self.fetch_injuries, fixture_ids))
        


    def save_progress(self, progress_data, filename='progress.json'):
        """Save progress to JSON file"""
        progress_file = os.path.join(self.base_path, filename)
        sanitized_data = self.sanitize_for_json(progress_data)
        try:
            with open(progress_file, "w") as file:
                json.dump(sanitized_data, file, indent=4)
            self.logger.info(f"Progress saved to {progress_file}")
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def load_progress(self, filename='progress.json'):
        """Load progress from JSON file"""
        progress_file = os.path.join(self.base_path, filename)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as file:
                    progress_data = json.load(file)
                self.logger.info(f"Progress loaded from {progress_file}")
                return progress_data
            except Exception as e:
                self.logger.error(f"Failed to load progress: {e}")
                return {}
        self.logger.info(f"No progress file found. Starting fresh.")
        return {}

    def sanitize_for_json(self, obj):
        """Recursively convert non-serializable types for JSON"""
        if isinstance(obj, dict):
            return {k: self.sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.sanitize_for_json(i) for i in obj]
        if hasattr(obj, 'item'):  # Handle numpy types
            return obj.item()
        return obj 
        
    
    def collect_league_data(self, league_id, season, data_types=None, keep_progress=False,
                            batch_size=50, progress_file="data_collection_progress.json", start_date=None, end_date=None):
        """
        Main collection method with automatic date calculation and file-only logging
        """
        try:
            # 1. Get league metadata
            country_name, league_info = self.get_league_info(league_id)
            league_name = league_info['name']

            # 2. Calculate dates automatically OR use custom dates
            if start_date and end_date:
                # Use custom dates
                self.logger.info(f"Using custom date range: {start_date} to {end_date}")
                # For custom dates, create a season name based on the dates
                season_name = f"{season}-custom"
            else:
                # Calculate dates automatically based on league characteristics
                season_name, start_date, end_date = self.calculate_season_dates(
                    country_name, league_info, season
                )
                self.logger.info(f"Using automatic date range: {start_date} to {end_date}")
            
            current_year = datetime.now().year
            is_current_season = str(current_year) in season_name

            # 3. Set default data types
            if data_types is None:
                data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                            'lineups', 'injuries', 'team_standings', 'odds']

            # 4. Generate storage paths
            raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
            os.makedirs(raw_path, exist_ok=True)

            # --- Progress Handling ---
            progress = {
                'league_id': league_id,
                'country': country_name,
                'league': league_name,
                'season': season_name,
                'is_current_season': is_current_season,
                'data_types': {dt: {} for dt in data_types},
                'last_updated': datetime.now().isoformat(),
                'fixture_ids': [],
                'date_range': f"{start_date} to {end_date}"
            }
            
            # Load existing progress if requested
            existing_progress = self.load_progress(progress_file) if keep_progress else None
            
            # Check if existing progress matches our current collection
            if existing_progress and (
                existing_progress.get('league_id') == league_id and 
                existing_progress.get('season') == season_name and
                existing_progress.get('date_range') == progress['date_range']
            ):
                # Use existing progress but update metadata
                progress.update(existing_progress)
                progress['last_updated'] = datetime.now().isoformat()
                self.logger.info(f"Resuming collection from existing progress for {league_name} {season_name}")
            else:
                if keep_progress and existing_progress:
                    self.logger.info(f"Existing progress doesn't match current collection (league_id: {league_id}, season: {season_name})")
                    self.logger.info("Starting fresh collection")
                self.save_progress(progress, progress_file)  # Initialize progress file

            # --- Fetch Fixtures ---
            self.logger.info(f"Fetching fixtures for {league_name} {season_name}...")
            self.logger.info(f"Date range: {start_date} to {end_date}")
            
            fixtures = self.fetch_fixture_events(
                league_id=league_id,
                season=season,
                start_date=start_date,
                end_date=end_date
            )
            
            if not fixtures or not fixtures.get('response'):
                self.logger.warning("No fixtures found in the specified date range")
                return {'status': 'error', 'message': 'No fixtures found', 'date_range': f"{start_date} to {end_date}"}
            
            current_fixtures = fixtures['response']
            all_fixture_ids = [str(f['fixture']['id']) for f in current_fixtures]
            progress['fixture_ids'] = all_fixture_ids

            # Track fixture_events in progress
            for fid in all_fixture_ids:
                if fid not in progress['data_types']['fixture_events']:
                    progress['data_types']['fixture_events'][fid] = 'completed'

            # Save fixtures
            if is_current_season:
                # Save daily snapshot
                daily_path = os.path.join(raw_path, f"fixture_events_{datetime.now().strftime('%Y-%m-%d')}.json")
                with open(daily_path, 'w') as f:
                    json.dump(current_fixtures, f, indent=2)
                
                # Merge with existing
                merged_path = os.path.join(raw_path, "fixture_events.json")
                if os.path.exists(merged_path):
                    with open(merged_path, 'r') as f:
                        existing = json.load(f)
                    existing_ids = {str(f['fixture']['id']) for f in existing}
                    current_fixtures = existing + [f for f in current_fixtures if str(f['fixture']['id']) not in existing_ids]
                
                with open(merged_path, 'w') as f:
                    json.dump(current_fixtures, f, indent=2)
                self.logger.info(f"Saved daily fixture events to {daily_path}")
            else:
                fixture_path = os.path.join(raw_path, 'fixture_events.json')
                with open(fixture_path, 'w') as f:
                    json.dump(current_fixtures, f, indent=2)
                self.logger.info(f"Saved fixture events to {fixture_path}")

            self.save_progress(progress, progress_file)

            # --- Process other data types ---
            collection = {dt: [] for dt in data_types if dt not in ['fixture_events', 'team_standings']}
            
            # Determine which fixtures need processing
            fixtures_to_process = {}
            for data_type in collection.keys():
                fixtures_to_process[data_type] = [
                    fid for fid in all_fixture_ids 
                    if progress['data_types'][data_type].get(fid) != 'completed'
                ]
                self.logger.info(f"Found {len(fixtures_to_process[data_type])} fixtures to process for {data_type}")

            # Mapping of data types to their fetch methods
            fetch_functions = {
                'team_statistics': self.fetch_team_statistics,
                'player_statistics': self.fetch_player_statistics,
                'lineups': self.fetch_lineups,
                'injuries': self.fetch_injuries,
                'odds': self.fetch_odds,
            }

            # --- Batch processing ---
            for data_type, fetch_func in fetch_functions.items():
                if data_type not in collection:
                    continue
                    
                fixture_ids = fixtures_to_process[data_type]
                if not fixture_ids:
                    self.logger.info(f"No fixtures need processing for {data_type}")
                    continue
                    
                self.logger.info(f"Processing {len(fixture_ids)} fixtures for {data_type}")
                
                for start in range(0, len(fixture_ids), batch_size):
                    batch = fixture_ids[start:start+batch_size]
                    self.logger.info(f"Processing batch {start//batch_size + 1} of {len(fixture_ids)//batch_size + 1}")
                    
                    for fixture_id in batch:
                        try:
                            data = fetch_func(int(fixture_id))
                            if data:
                                collection[data_type].append(data)
                                progress['data_types'][data_type][fixture_id] = 'completed'
                        except Exception as e:
                            self.logger.error(f"Failed {data_type} for fixture {fixture_id}: {str(e)}")
                            progress['data_types'][data_type][fixture_id] = f'failed: {str(e)}'
                    
                    self.save_progress(progress, progress_file)

            # Save collected data
            for data_type, data_list in collection.items():
                if not data_list:
                    continue
                    
                if is_current_season:
                    # Save daily file
                    daily_path = os.path.join(raw_path, f"{data_type}_{datetime.now().strftime('%Y-%m-%d')}.json")
                    with open(daily_path, 'w') as f:
                        json.dump(data_list, f, indent=2)
                    
                    # Merge with existing
                    merged_path = os.path.join(raw_path, f"{data_type}.json")
                    if os.path.exists(merged_path):
                        with open(merged_path, 'r') as f:
                            existing = json.load(f)
                        existing_ids = {str(item['fixture']['id']) for item in existing if 'fixture' in item}
                        data_list = existing + [item for item in data_list 
                                            if str(item.get('fixture', {}).get('id')) not in existing_ids]
                    
                    with open(merged_path, 'w') as f:
                        json.dump(data_list, f, indent=2)
                    self.logger.info(f"Saved {len(data_list)} {data_type} records (Daily: {daily_path})")
                else:
                    out_path = os.path.join(raw_path, f'{data_type}.json')
                    with open(out_path, 'w') as f:
                        json.dump(data_list, f, indent=2)
                    self.logger.info(f"Saved {len(data_list)} {data_type} records to {out_path}")

            # --- Handle Team Standings ---
            if 'team_standings' in data_types:
                try:
                    standings = self.fetch_team_standings(league_id, season)
                    
                    if is_current_season:
                        # Save daily snapshot
                        daily_path = os.path.join(raw_path, f"team_standings_{datetime.now().strftime('%Y-%m-%d')}.json")
                        with open(daily_path, 'w') as f:
                            json.dump(standings, f, indent=2)
                        
                        # Replace rather than merge for standings
                        merged_path = os.path.join(raw_path, "team_standings.json")
                        with open(merged_path, 'w') as f:
                            json.dump(standings, f, indent=2)
                        
                        self.logger.info(f"Saved team standings (Daily: {daily_path})")
                        progress['data_types']['team_standings'] = {
                            'last_updated': datetime.now().isoformat(),
                            'status': 'completed'
                        }
                    else:
                        # For past seasons
                        filepath = os.path.join(raw_path, 'team_standings.json')
                        with open(filepath, 'w') as f:
                            json.dump(standings, f, indent=2)
                        self.logger.info(f"Saved team standings to {filepath}")
                        progress['data_types']['team_standings'] = {'status': 'completed'}
                    
                except Exception as e:
                    self.logger.error(f"Failed to fetch/save team standings: {e}")
                    progress['data_types']['team_standings'] = {
                        'status': f'failed: {str(e)}',
                        'last_attempt': datetime.now().isoformat()
                    }
                finally:
                    self.save_progress(progress, progress_file)

            return {
                'status': 'success',
                'league': league_name,
                'season': season_name,
                'fixture_events_collected': len(all_fixture_ids),
                'data_collected': {
                    dt: sum(1 for status in progress['data_types'][dt].values() 
                        if status == 'completed') 
                    for dt in data_types if dt != 'fixture_events'
                },
                'storage_path': raw_path,
                'progress_file': progress_file,
                'date_range': f"{start_date} to {end_date}"
            }
        
        except Exception as e:
            self.logger.error(f"Collection failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'league_id': league_id,
                'season': season
            }
        
# Initialize collector
collector = FootballDataCollector(api_key=API_KEY, base_path="data/raw")



# Collect data with automatic date calculation
result = collector.collect_league_data(
    league_id=140,  # La Liga
    season=2022,    # 2022-2023 season
    data_types=['fixture_events', 'team_statistics'],
    keep_progress=True,
    start_date="2022-08-01",
    end_date= "2022-08-15"  # Custom date range for testing purposes

)

print(f"Result: {result}")