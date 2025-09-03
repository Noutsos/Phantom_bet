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
        
    def get_current_date_range(self, is_current_season: bool) -> tuple:
        """Get appropriate date range based on collection phase"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        if is_current_season:
            # For current season, collect up to yesterday for completed games
            return "2023-08-01", yesterday.strftime('%Y-%m-%d')
        else:
            # For past seasons, collect entire season
            return None, None

    def get_ns_date_range(self) -> tuple:
        """Get date range for NS (Not Started) games"""
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        future_date = today + timedelta(days=30)  # Look 30 days ahead
        return tomorrow.strftime('%Y-%m-%d'), future_date.strftime('%Y-%m-%d')

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
    
    def update_ns_fixtures(self, raw_path: str, progress: Dict):
        """Update previously collected NS fixtures that are now completed"""
        fixture_file = os.path.join(raw_path, "fixture_events.json")
        
        if not os.path.exists(fixture_file):
            self.logger.warning("No fixture_events.json found for update")
            return progress
        
        try:
            with open(fixture_file, 'r') as f:
                existing_fixtures = json.load(f)
            
            # Find NS fixtures that need updating
            ns_fixtures_to_update = []
            for fixture in existing_fixtures:
                fixture_id = str(fixture['fixture']['id'])
                status = fixture['fixture']['status']['short']
                
                if status == 'NS' and fixture_id in progress.get('ns_fixtures', []):
                    ns_fixtures_to_update.append(fixture_id)
            
            if not ns_fixtures_to_update:
                self.logger.info("No NS fixtures need updating")
                return progress
            
            self.logger.info(f"Found {len(ns_fixtures_to_update)} NS fixtures to update")
            
            # Fetch updated fixture data
            updated_fixtures = []
            for fixture_id in ns_fixtures_to_update:
                try:
                    # Fetch single fixture to check status
                    endpoint = f"/fixtures?id={fixture_id}"
                    response = self.make_api_request(endpoint)
                    
                    if response and response.get('response'):
                        updated_fixture = response['response'][0]
                        current_status = updated_fixture['fixture']['status']['short']
                        
                        if current_status != 'NS':
                            # Replace the NS fixture with updated data
                            updated_fixtures.append(updated_fixture)
                            self.logger.info(f"Updated fixture {fixture_id} from NS to {current_status}")
                        else:
                            self.logger.info(f"Fixture {fixture_id} is still NS")
                
                except Exception as e:
                    self.logger.error(f"Failed to update fixture {fixture_id}: {e}")
            
            # Replace NS fixtures with updated ones
            if updated_fixtures:
                # Create mapping for easy replacement
                updated_map = {str(f['fixture']['id']): f for f in updated_fixtures}
                
                new_fixtures = []
                for fixture in existing_fixtures:
                    fixture_id = str(fixture['fixture']['id'])
                    if fixture_id in updated_map:
                        new_fixtures.append(updated_map[fixture_id])
                    else:
                        new_fixtures.append(fixture)
                
                # Save updated fixtures
                with open(fixture_file, 'w') as f:
                    json.dump(new_fixtures, f, indent=2)
                
                self.logger.info(f"Updated {len(updated_fixtures)} fixtures in fixture_events.json")
                
                # Update progress
                for fixture_id in updated_map.keys():
                    if fixture_id in progress.get('ns_fixtures', []):
                        progress['ns_fixtures'].remove(fixture_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update NS fixtures: {e}")
        
        return progress
    
    def collect_league_data_simple(self, league_id, season, data_types=None, keep_progress=False,
                           batch_size=50, progress_file="data_collection_progress.json", 
                           start_date=None, end_date=None, collection_phase=1):
        """
        Main collection method with three-phase system
        
        Args:
            collection_phase: 1 = completed games, 2 = NS games, 3 = update NS games
        """
        try:
            # 1. Get league metadata
            country_name, league_info = self.get_league_info(league_id)
            league_name = league_info['name']
            
            current_year = datetime.now().year
            is_current_season = str(current_year) in str(season)

            # 2. Determine date range based on collection phase
            if collection_phase == 1:
                # Phase 1: Completed games
                if is_current_season:
                    # For current season: from start to yesterday
                    if not start_date or not end_date:
                        season_name, auto_start, auto_end = self.calculate_season_dates(
                            country_name, league_info, season
                        )
                        start_date = auto_start
                    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    self.logger.info(f"Phase 1: Collecting completed games from {start_date} to {end_date}")
                else:
                    # For past seasons: entire season
                    if not start_date or not end_date:
                        season_name, start_date, end_date = self.calculate_season_dates(
                            country_name, league_info, season
                        )
                    self.logger.info(f"Phase 1: Collecting entire season {start_date} to {end_date}")
            
            elif collection_phase == 2:
                # Phase 2: NS games
                if not is_current_season:
                    self.logger.info("Phase 2: Not applicable for past seasons")
                    return {'status': 'skipped', 'reason': 'Not current season'}
                
                start_date, end_date = self.get_ns_date_range()
                self.logger.info(f"Phase 2: Collecting NS games from {start_date} to {end_date}")
            
            elif collection_phase == 3:
                # Phase 3: Update previously collected NS games
                if not is_current_season:
                    self.logger.info("Phase 3: Not applicable for past seasons")
                    return {'status': 'skipped', 'reason': 'Not current season'}
                
                # This phase doesn't fetch new fixtures, just updates existing ones
                season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
                raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
                
                progress = self.load_progress(progress_file) if keep_progress else {}
                progress = self.update_ns_fixtures(raw_path, progress)
                self.save_progress(progress, progress_file)
                
                return {
                    'status': 'success',
                    'phase': 3,
                    'message': 'NS fixtures updated'
                }

            # 3. Set default data types
            if data_types is None:
                data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                            'lineups', 'injuries', 'team_standings', 'odds']

            # 4. Generate storage paths
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
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
                'ns_fixtures': [],  # Track NS fixtures for phase 3 updates
                'date_range': f"{start_date} to {end_date}",
                'collection_phase': collection_phase
            }
            
            # Load existing progress if requested
            existing_progress = self.load_progress(progress_file) if keep_progress else None
            
            if existing_progress and (
                existing_progress.get('league_id') == league_id and 
                existing_progress.get('season') == season_name
            ):
                progress.update(existing_progress)
                progress['last_updated'] = datetime.now().isoformat()
                progress['collection_phase'] = collection_phase
                self.logger.info(f"Resuming collection from existing progress")
            else:
                self.save_progress(progress, progress_file)

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
            
            # Track NS fixtures for phase 3 updates
            if collection_phase == 2:
                ns_fixtures = [str(f['fixture']['id']) for f in current_fixtures 
                             if f['fixture']['status']['short'] == 'NS']
                progress['ns_fixtures'] = list(set(progress.get('ns_fixtures', []) + ns_fixtures))
                self.logger.info(f"Found {len(ns_fixtures)} NS fixtures")
            
            progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + all_fixture_ids))

            # Track fixture_events in progress
            for fid in all_fixture_ids:
                if fid not in progress['data_types']['fixture_events']:
                    progress['data_types']['fixture_events'][fid] = 'completed'

            # Save fixtures with appropriate strategy
            fixture_path = os.path.join(raw_path, "fixture_events.json")
            
            if collection_phase == 1 or not os.path.exists(fixture_path):
                # Phase 1 or first time: create new file
                with open(fixture_path, 'w') as f:
                    json.dump(current_fixtures, f, indent=2)
                self.logger.info(f"Saved fixture events to {fixture_path}")
            else:
                # Phase 2: Merge with existing, avoiding duplicates
                with open(fixture_path, 'r') as f:
                    existing_fixtures = json.load(f)
                
                existing_ids = {str(f['fixture']['id']) for f in existing_fixtures}
                new_fixtures = [f for f in current_fixtures 
                              if str(f['fixture']['id']) not in existing_ids]
                
                if new_fixtures:
                    combined_fixtures = existing_fixtures + new_fixtures
                    with open(fixture_path, 'w') as f:
                        json.dump(combined_fixtures, f, indent=2)
                    self.logger.info(f"Added {len(new_fixtures)} new fixtures to {fixture_path}")
                else:
                    self.logger.info("No new fixtures to add")

            self.save_progress(progress, progress_file)

            # --- Process other data types (only for completed games in phase 1) ---
            if collection_phase == 1:
                collection = {dt: [] for dt in data_types if dt not in ['fixture_events', 'team_standings']}
                
                # Only process completed games
                completed_fixtures = [
                    str(f['fixture']['id']) for f in current_fixtures 
                    if f['fixture']['status']['short'] != 'NS'
                ]
                
                fixtures_to_process = {}
                for data_type in collection.keys():
                    fixtures_to_process[data_type] = [
                        fid for fid in completed_fixtures 
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
                        self.logger.info(f"Processing batch {start//batch_size + 1}")
                        
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
                        
                    out_path = os.path.join(raw_path, f'{data_type}.json')
                    
                    if os.path.exists(out_path):
                        with open(out_path, 'r') as f:
                            existing_data = json.load(f)
                        existing_ids = {str(item.get('fixture', {}).get('id')) for item in existing_data}
                        new_data = [item for item in data_list 
                                  if str(item.get('fixture', {}).get('id')) not in existing_ids]
                        data_list = existing_data + new_data
                    
                    with open(out_path, 'w') as f:
                        json.dump(data_list, f, indent=2)
                    self.logger.info(f"Saved {len(data_list)} {data_type} records to {out_path}")

                # --- Handle Team Standings ---
                if 'team_standings' in data_types:
                    try:
                        standings = self.fetch_team_standings(league_id, season)
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
                'phase': collection_phase,
                'league': league_name,
                'season': season_name,
                'fixture_events_collected': len(all_fixture_ids),
                'ns_fixtures_tracked': len(progress.get('ns_fixtures', [])),
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
                'season': season,
                'phase': collection_phase
            }

    def collect_league_data(self, league_id=None, season=None, data_types=None, keep_progress=False,
                        batch_size=50, progress_file="data_collection_progress.json", 
                        start_date=None, end_date=None, collection_phase=1,
                        filter_ids=None, filter_tiers=None, filter_cups=None, filter_categories=None):
        """
        Main collection method with three-phase system and advanced filtering
        
        Args:
            collection_phase: 1 = completed games, 2 = NS games, 3 = update NS games
            filter_ids: List of specific league IDs to filter by
            filter_tiers: List of tiers to filter by (e.g., ['top_tier', 'second_tier'])
            filter_cups: List of cup types to filter by (e.g., ['domestic_cup', 'league_cup'])
            filter_categories: List of league categories to filter by (e.g., ['european_elite', 'top_tier'])
        """
        try:
            # --- Filter leagues based on provided criteria ---
            filtered_leagues = self._filter_leagues(
                league_id=league_id,
                filter_ids=filter_ids,
                filter_tiers=filter_tiers,
                filter_cups=filter_cups,
                filter_categories=filter_categories
            )
            
            if not filtered_leagues:
                self.logger.warning("No leagues match the filtering criteria")
                return {'status': 'error', 'message': 'No leagues match filtering criteria'}
            
            results = []
            
            # Process each filtered league
            for country_name, league_info in filtered_leagues:
                current_league_id = league_info.get('id') if isinstance(league_info, dict) else league_info
                league_name = league_info['name'] if isinstance(league_info, dict) else self.get_league_info(current_league_id)[1]['name']
                
                result = self._process_single_league(
                    current_league_id, season, data_types, keep_progress,
                    batch_size, progress_file, start_date, end_date,
                    collection_phase, country_name, league_name
                )
                results.append(result)
            
            return {
                'status': 'success',
                'phase': collection_phase,
                'results': results,
                'total_leagues_processed': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Collection failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'league_id': league_id,
                'season': season,
                'phase': collection_phase
            }

    def get_available_categories(self):
        """
        Return all available categories for filtering
        """
        categories = set()
        regional_categories = set()
        
        def extract_categories(leagues_dict, path=None):
            for key, value in leagues_dict.items():
                if isinstance(value, dict):
                    if 'name' in value and 'category' in value:
                        # League with category
                        categories.add(value['category'])
                    else:
                        # Regional/country category
                        if path is None:
                            regional_categories.add(key)
                        else:
                            regional_categories.add(key)
                        extract_categories(value, path)
        
        extract_categories(LEAGUES)
        
        return {
            'league_categories': sorted(list(categories)),
            'regional_categories': sorted(list(regional_categories)),
            'all_categories': sorted(list(categories.union(regional_categories)))
        }    
    
    def _filter_leagues(self, league_id=None, filter_ids=None, filter_tiers=None, 
                    filter_cups=None, filter_categories=None):
        """
        Filter leagues based on various criteria including regional categories
        """
        filtered_leagues = []
        
        # If specific league_id is provided, use that
        if league_id:
            try:
                country_name, league_info = self.get_league_info(league_id)
                filtered_leagues.append((country_name, league_info))
                return filtered_leagues
            except:
                self.logger.warning(f"League ID {league_id} not found")
                return []
        
        # If no filters provided, return empty (or could return all leagues)
        if not any([filter_ids, filter_tiers, filter_cups, filter_categories]):
            self.logger.warning("No filtering criteria provided")
            return []
        
        # Recursively search through LEAGUES structure with path tracking
        def search_leagues(leagues_dict, current_path=None):
            found = []
            for key, value in leagues_dict.items():
                if isinstance(value, dict):
                    if 'name' in value and 'category' in value:
                        # This is a league entry
                        league_data = value.copy()
                        league_data['id'] = key
                        
                        # Get the full path (regional hierarchy)
                        full_path = current_path + [key] if current_path else [key]
                        
                        # Apply filters
                        matches = True
                        
                        # Filter by specific league IDs
                        if filter_ids and key not in filter_ids:
                            matches = False
                        
                        # Filter by tier (category)
                        if filter_tiers and league_data.get('category') not in filter_tiers:
                            matches = False
                        
                        # Filter by cup type (category)
                        if filter_cups and league_data.get('category') not in filter_cups:
                            matches = False
                        
                        # Filter by categories - check both league category and regional path
                        if filter_categories:
                            category_matches = False
                            
                            # Check league-level categories (top_tier, domestic_cup, etc.)
                            if league_data.get('category') in filter_categories:
                                category_matches = True
                            
                            # Check regional categories in the path
                            for category in filter_categories:
                                if category in full_path:
                                    category_matches = True
                                    break
                            
                            if not category_matches:
                                matches = False
                        
                        if matches:
                            # Use the country name from the path (usually the parent of the league)
                            country_name = full_path[-2] if len(full_path) >= 2 else full_path[-1]
                            found.append((country_name, league_data))
                    else:
                        # This is a category or country, search deeper
                        new_path = current_path + [key] if current_path else [key]
                        found.extend(search_leagues(value, new_path))
            return found
        
        return search_leagues(LEAGUES)

    def _process_single_league(self, league_id, season, data_types, keep_progress,
                            batch_size, progress_file, start_date, end_date,
                            collection_phase, country_name, league_name):
        """
        Process data collection for a single league
        """
        # 1. Get league metadata if not provided - ALWAYS get league_info
        if not country_name or not league_name:
            country_name, league_info = self.get_league_info(league_id)
            league_name = league_info['name']
        else:
            # Still need to get league_info for season date calculations
            _, league_info = self.get_league_info(league_id)
        
        current_year = datetime.now().year
        is_current_season = str(current_year) in str(season)

        # 2. Determine date range based on collection phase
        if collection_phase == 1:
            # Phase 1: Completed games
            if is_current_season:
                # For current season: from start to yesterday
                if not start_date or not end_date:
                    season_name, auto_start, auto_end = self.calculate_season_dates(
                        country_name, league_info, season
                    )
                    start_date = auto_start
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                self.logger.info(f"Phase 1: Collecting completed games from {start_date} to {end_date}")
            else:
                # For past seasons: entire season
                if not start_date or not end_date:
                    season_name, start_date, end_date = self.calculate_season_dates(
                        country_name, league_info, season
                    )
                self.logger.info(f"Phase 1: Collecting entire season {start_date} to {end_date}")
        
        elif collection_phase == 2:
            # Phase 2: NS games
            if not is_current_season:
                self.logger.info("Phase 2: Not applicable for past seasons")
                return {'status': 'skipped', 'reason': 'Not current season', 'league_id': league_id}
            
            start_date, end_date = self.get_ns_date_range()
            self.logger.info(f"Phase 2: Collecting NS games from {start_date} to {end_date}")
        
        elif collection_phase == 3:
            # Phase 3: Update previously collected NS games
            if not is_current_season:
                self.logger.info("Phase 3: Not applicable for past seasons")
                return {'status': 'skipped', 'reason': 'Not current season', 'league_id': league_id}
            
            # This phase doesn't fetch new fixtures, just updates existing ones
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
            raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
            
            progress = self.load_progress(progress_file) if keep_progress else {}
            progress = self.update_ns_fixtures(raw_path, progress)
            self.save_progress(progress, progress_file)
            
            return {
                'status': 'success',
                'phase': 3,
                'message': 'NS fixtures updated',
                'league_id': league_id
            }

        # 3. Set default data types
        if data_types is None:
            data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                        'lineups', 'injuries', 'team_standings', 'odds']

        # 4. Generate storage paths - ensure season_name is defined
        if 'season_name' not in locals():
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
        
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
            'ns_fixtures': [],  # Track NS fixtures for phase 3 updates
            'date_range': f"{start_date} to {end_date}",
            'collection_phase': collection_phase
        }
        
        # Load existing progress if requested
        existing_progress = self.load_progress(progress_file) if keep_progress else None
        
        if existing_progress and (
            existing_progress.get('league_id') == league_id and 
            existing_progress.get('season') == season_name
        ):
            progress.update(existing_progress)
            progress['last_updated'] = datetime.now().isoformat()
            progress['collection_phase'] = collection_phase
            self.logger.info(f"Resuming collection from existing progress")
        else:
            self.save_progress(progress, progress_file)

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
            return {'status': 'error', 'message': 'No fixtures found', 'date_range': f"{start_date} to {end_date}", 'league_id': league_id}
        
        current_fixtures = fixtures['response']
        all_fixture_ids = [str(f['fixture']['id']) for f in current_fixtures]
        
        # Track NS fixtures for phase 3 updates
        if collection_phase == 2:
            ns_fixtures = [str(f['fixture']['id']) for f in current_fixtures 
                        if f['fixture']['status']['short'] == 'NS']
            progress['ns_fixtures'] = list(set(progress.get('ns_fixtures', []) + ns_fixtures))
            self.logger.info(f"Found {len(ns_fixtures)} NS fixtures")
        
        progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + all_fixture_ids))

        # Track fixture_events in progress
        for fid in all_fixture_ids:
            if fid not in progress['data_types']['fixture_events']:
                progress['data_types']['fixture_events'][fid] = 'completed'

        # Save fixtures with appropriate strategy
        fixture_path = os.path.join(raw_path, "fixture_events.json")
        
        if collection_phase == 1 or not os.path.exists(fixture_path):
            # Phase 1 or first time: create new file
            with open(fixture_path, 'w') as f:
                json.dump(current_fixtures, f, indent=2)
            self.logger.info(f"Saved fixture events to {fixture_path}")
        else:
            # Phase 2: Merge with existing, avoiding duplicates
            with open(fixture_path, 'r') as f:
                existing_fixtures = json.load(f)
            
            existing_ids = {str(f['fixture']['id']) for f in existing_fixtures}
            new_fixtures = [f for f in current_fixtures 
                        if str(f['fixture']['id']) not in existing_ids]
            
            if new_fixtures:
                combined_fixtures = existing_fixtures + new_fixtures
                with open(fixture_path, 'w') as f:
                    json.dump(combined_fixtures, f, indent=2)
                self.logger.info(f"Added {len(new_fixtures)} new fixtures to {fixture_path}")
            else:
                self.logger.info("No new fixtures to add")

        self.save_progress(progress, progress_file)

        # --- Process other data types (only for completed games in phase 1) ---
        if collection_phase == 1:
            collection = {dt: [] for dt in data_types if dt not in ['fixture_events', 'team_standings']}
            
            # Only process completed games
            completed_fixtures = [
                str(f['fixture']['id']) for f in current_fixtures 
                if f['fixture']['status']['short'] != 'NS'
            ]
            
            fixtures_to_process = {}
            for data_type in collection.keys():
                fixtures_to_process[data_type] = [
                    fid for fid in completed_fixtures 
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
                    self.logger.info(f"Processing batch {start//batch_size + 1}")
                    
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
                    
                out_path = os.path.join(raw_path, f'{data_type}.json')
                
                if os.path.exists(out_path):
                    with open(out_path, 'r') as f:
                        existing_data = json.load(f)
                    existing_ids = {str(item.get('fixture', {}).get('id')) for item in existing_data}
                    new_data = [item for item in data_list 
                            if str(item.get('fixture', {}).get('id')) not in existing_ids]
                    data_list = existing_data + new_data
                
                with open(out_path, 'w') as f:
                    json.dump(data_list, f, indent=2)
                self.logger.info(f"Saved {len(data_list)} {data_type} records to {out_path}")

            # --- Handle Team Standings ---
            if 'team_standings' in data_types:
                try:
                    standings = self.fetch_team_standings(league_id, season)
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
            'phase': collection_phase,
            'league': league_name,
            'league_id': league_id,
            'season': season_name,
            'fixture_events_collected': len(all_fixture_ids),
            'ns_fixtures_tracked': len(progress.get('ns_fixtures', [])),
            'storage_path': raw_path,
            'progress_file': progress_file,
            'date_range': f"{start_date} to {end_date}"
        }
# Initialize collector
collector = FootballDataCollector(api_key=API_KEY, base_path="data/raw")

