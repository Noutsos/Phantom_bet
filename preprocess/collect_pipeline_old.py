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
from typing import Dict, List, Optional, Union, Set, Any
import numpy as np
from src.utils import LEAGUES  


class FootballDataCollector:
    """Class to handle automated football data collection with automatic mode detection"""
    
    def __init__(self, api_key: str, base_path: str = "data/raw", log_dir: str = "logs"):
        self.API_HOST = "v3.football.api-sports.io"
        self.API_KEY = api_key
        self.base_path = base_path
        self.log_dir = log_dir
        
        # Ensure directories exist
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiter (assuming this is defined elsewhere)
        self.rate_limiter = self.RateLimiter() if hasattr(self, 'RateLimiter') else None
        
        self.logger.info("FootballDataCollector initialized")
        self.logger.info(f"Base path: {base_path}")
        self.logger.info(f"Log directory: {log_dir}")
    
    def _setup_logging(self):
        """Setup logging to log folder only - no console output"""
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Main log file - captures ALL levels
        log_file = os.path.join(self.log_dir, f"football_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Separate error log file
        error_log_file = os.path.join(self.log_dir, "errors.log")
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Use the root logger for setup messages
        root_logger.info(f"Logging initialized. All logs: {log_file}")
        root_logger.info(f"Error logs: {error_log_file}")

    # ==================== CATEGORY-BASED COLLECTION METHODS ====================
    
    def get_all_categories(self) -> List[str]:
        """Get all available category names from LEAGUES structure"""
        # This would return categories like: ['Europe', 'Top 5 European Leagues', 'Western Europe', etc.]
        # For now, return dummy categories - you'll need to implement based on your LEAGUES structure
        return ['Europe', 'Top 5 European Leagues', 'Western Europe', 'Eastern Europe', 'Baltic & Nordic']
    
    def get_league_ids_by_category(self, category: str, league_types: List[str] = None) -> List[int]:
        """
        Get all league IDs in a specific category, optionally filtered by league types
        
        Args:
            category: Category name (e.g., 'Top 5 European Leagues')
            league_types: Filter by league types (e.g., ['top_tier', 'domestic_cup'])
        """
        self.logger.info(f"Getting league IDs for category: {category}, types: {league_types}")
        
        # This would be implemented based on your LEAGUES structure
        # For now, return dummy data - you'll need to implement this properly
        
        category_leagues = {
            'Top 5 European Leagues': [39, 140, 135, 78, 61],  # Top tier leagues
            'Europe': [2, 3, 848],  # International competitions
            'Western Europe': [88, 94, 144, 207, 218, 179],  # Other European leagues
        }
        
        league_ids = category_leagues.get(category, [])
        self.logger.info(f"Found {len(league_ids)} leagues in category '{category}'")
        return league_ids
    
    def collect_by_category(self, category: str, season: int, 
                          data_types: List[str] = None,
                          league_types: List[str] = None,
                          max_workers: int = 3) -> Dict[str, Any]:
        """
        Collect data for all leagues in a specific category
        
        Args:
            category: Category name (e.g., 'Top 5 European Leagues')
            season: Season year
            data_types: List of data types to collect
            league_types: Filter leagues by type (e.g., ['top_tier', 'domestic_cup'])
            max_workers: Maximum concurrent leagues to process
        """
        self.logger.info(f"Starting category collection: {category}, season: {season}")
        
        league_ids = self.get_league_ids_by_category(category, league_types)
        if not league_ids:
            self.logger.warning(f"No leagues found in category '{category}'")
            return {'status': 'error', 'message': f'No leagues found in category {category}'}
        
        results = {}
        
        # Process leagues with limited concurrency to avoid API overload
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_league = {
                executor.submit(self.automated_collection, league_id, season, data_types): league_id
                for league_id in league_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_league):
                league_id = future_to_league[future]
                try:
                    result = future.result()
                    results[league_id] = result
                    self.logger.info(f"Completed collection for league {league_id}: {result['status']}")
                except Exception as e:
                    self.logger.error(f"Failed to collect data for league {league_id}: {e}")
                    results[league_id] = {
                        'status': 'error',
                        'message': str(e)
                    }
        
        # Generate summary
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = sum(1 for r in results.values() if r.get('status') == 'error')
        
        summary = {
            'category': category,
            'season': season,
            'total_leagues': len(league_ids),
            'successful': successful,
            'failed': failed,
            'league_results': results
        }
        
        self.logger.info(f"Category collection completed: {successful}/{len(league_ids)} successful")
        return summary
    
    def collect_multiple_categories(self, categories: List[str], season: int,
                                  data_types: List[str] = None,
                                  league_types: List[str] = None,
                                  max_workers_per_category: int = 2) -> Dict[str, Any]:
        """
        Collect data for multiple categories sequentially
        
        Args:
            categories: List of category names
            season: Season year
            data_types: List of data types to collect
            league_types: Filter leagues by type
            max_workers_per_category: Maximum concurrent leagues per category
        """
        self.logger.info(f"Starting multiple category collection: {categories}")
        
        all_results = {}
        
        for category in categories:
            if category not in self.get_all_categories():
                self.logger.warning(f"Skipping unknown category: {category}")
                continue
                
            self.logger.info(f"Processing category: {category}")
            
            result = self.collect_by_category(
                category=category,
                season=season,
                data_types=data_types,
                league_types=league_types,
                max_workers=max_workers_per_category
            )
            
            all_results[category] = result
            
            # Add delay between categories to avoid rate limiting
            time.sleep(2)
        
        return all_results

    # ==================== CORE COLLECTION METHODS ====================
    
    def get_league_info(self, league_id: int) -> tuple:
        """Find which country a league belongs to and its metadata"""
        # This would be implemented based on your LEAGUES structure
        # For now, return dummy data - you'll need to implement this
        league_info = {
            'name': f'League {league_id}',
            'start_month': 8,
            'season_months': 10,
            'category': 'top_tier'
        }
        return "Unknown", league_info

    def calculate_season_dates(self, country_name: str, league_info: Dict, season_year: int) -> tuple:
        """Calculate automatic season dates based on league characteristics"""
        start_month = league_info.get('start_month', 8)
        start_date = datetime(season_year, start_month, 1).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
                   relativedelta(months=+league_info['season_months'])).strftime('%Y-%m-%d')
        season_name = f"{season_year}"
        return season_name, start_date, end_date

    class RateLimiter:
        def __init__(self, daily_limit=100, per_minute_limit=10):
            self.daily_limit = daily_limit
            self.per_minute_limit = per_minute_limit
            self.remaining_daily = daily_limit
            self.remaining_per_minute = per_minute_limit
            self.minute_window_start = time.time()
            self.logger = logging.getLogger(__name__ + ".RateLimiter")

        def check_rate_limit(self):
            current_time = time.time()
            elapsed = current_time - self.minute_window_start
            if elapsed >= 60:
                self.remaining_per_minute = self.per_minute_limit
                self.minute_window_start = current_time
                self.logger.debug("Reset per-minute rate limit")

            if self.remaining_daily <= 0:
                self.logger.error("Daily API request limit reached")
                raise Exception("Daily API request limit reached.")

            if self.remaining_per_minute <= 0:
                sleep_time = 60 - elapsed
                if sleep_time > 0:
                    self.logger.warning(f"Per-minute rate limit reached. Waiting for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                self.remaining_per_minute = self.per_minute_limit
                self.minute_window_start = time.time()

            self.remaining_daily -= 1
            self.remaining_per_minute -= 1
            self.logger.debug(f"Rate limits - Daily: {self.remaining_daily}, Minute: {self.remaining_per_minute}")

        def update_rate_limits(self, headers):
            if headers is None:
                return
            headers = {k.lower(): v for k, v in headers.items()}
            if 'x-ratelimit-requests-remaining' in headers:
                try:
                    self.remaining_daily = int(headers['x-ratelimit-requests-remaining'])
                    self.logger.debug(f"Updated daily rate limit: {self.remaining_daily}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse daily rate limit: {e}")
            if 'x-ratelimit-remaining' in headers:
                try:
                    self.remaining_per_minute = int(headers['x-ratelimit-remaining'])
                    self.logger.debug(f"Updated per-minute rate limit: {self.remaining_per_minute}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse per-minute rate limit: {e}")

    def make_api_request(self, endpoint: str) -> Dict:
        """Make API request with rate limiting"""
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
            except Exception as e:
                self.logger.error(f"Failed to parse JSON: {e} | Response: {data[:200]}")
                raise Exception(f"Failed to parse JSON: {e} | Response: {data[:200]}")

            if res.status == 200:
                self.rate_limiter.update_rate_limits(dict(res.getheaders()))
                self.logger.info(f"API request successful: {endpoint}")
                return response_data
            else:
                self.logger.error(f"API request failed with status {res.status}: {response_data}")
                raise Exception(f"API request failed with status {res.status}: {response_data}")
                
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

    def load_progress(self, league_id: int, season: str) -> Optional[Dict]:
        """Load progress for specific league and season"""
        progress_file = self._get_progress_file_path(league_id, season)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load progress: {e}")
        return None

    def save_progress(self, league_id: int, season: str, progress_data: Dict):
        """Save progress for specific league and season"""
        progress_file = self._get_progress_file_path(league_id, season)
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")

    def _get_progress_file_path(self, league_id: int, season: str) -> str:
        """Get path for progress file"""
        country, league_info = self.get_league_info(league_id)
        league_name = league_info['name']
        progress_dir = os.path.join(self.base_path, country, league_name, season, 'progress')
        os.makedirs(progress_dir, exist_ok=True)
        return os.path.join(progress_dir, f'progress_{league_id}_{season}.json')

    def _get_data_directory(self, league_id: int, season: str) -> str:
        """Get data directory path"""
        country, league_info = self.get_league_info(league_id)
        league_name = league_info['name']
        data_dir = os.path.join(self.base_path, country, league_name, season)
        os.makedirs(data_dir, exist_ok=True)
        self.logger.debug(f"Data directory: {data_dir}")
        return data_dir

    def determine_collection_mode(self, league_id: int, season: int) -> str:
        """
        Determine which collection mode to use based on current state
        Returns: 'past_season', 'current_completed', 'current_future', 'update_completed'
        """
        self.logger.info(f"Determining collection mode for league {league_id}, season {season}")
        
        current_year = datetime.now().year
        country, league_info = self.get_league_info(league_id)
        season_name, start_date, end_date = self.calculate_season_dates(country, league_info, season)
        
        # Check if this is current season
        is_current_season = str(current_year) in season_name
        
        if not is_current_season:
            self.logger.info("Mode: past_season (not current season)")
            return 'past_season'
        
        # Check progress to determine current season mode
        progress = self.load_progress(league_id, season_name)
        
        if not progress:
            self.logger.info("Mode: current_completed (first run for current season)")
            return 'current_completed'
        
        # Check if we have future games that need updating
        if 'future_fixtures' in progress and progress['future_fixtures']:
            future_ids = progress['future_fixtures']
            self.logger.info(f"Found {len(future_ids)} future fixtures in progress")
            
            current_fixtures = self._get_current_fixtures_status(league_id, season, future_ids)
            
            completed_future = [fid for fid, status in current_fixtures.items() 
                              if status != 'NS' and fid in future_ids]
            
            if completed_future:
                self.logger.info(f"Mode: update_completed ({len(completed_future)} future fixtures completed)")
                return 'update_completed'
        
        # Check if we should collect future games
        last_future_collection = progress.get('last_future_collection')
        if last_future_collection:
            last_date = datetime.fromisoformat(last_future_collection)
            if (datetime.now() - last_date).days >= 1:
                self.logger.info("Mode: current_future (time to collect future games)")
                return 'current_future'
        
        self.logger.info("Mode: current_completed (default)")
        return 'current_completed'

    def _get_current_fixtures_status(self, league_id: int, season: int, fixture_ids: List[int]) -> Dict:
        """Get current status of specific fixtures"""
        self.logger.info(f"Checking status of {len(fixture_ids)} fixtures")
        # This would typically involve checking the API for current fixture status
        # For now, return a mock response
        return {fid: 'FT' for fid in fixture_ids}

    def collect_past_season(self, league_id: int, season: int, 
                          data_types: List[str] = None) -> Dict:
        """Collect data for past seasons (complete season data)"""
        self.logger.info(f"Starting past season collection for league {league_id}, season {season}")
        
        country, league_info = self.get_league_info(league_id)
        season_name, start_date, end_date = self.calculate_season_dates(country, league_info, season)
        
        self.logger.info(f"Collecting past season data for {league_info['name']} {season_name}")
        
        return self._collect_data(
            league_id, season, start_date, end_date, data_types, 
            is_current_season=False
        )

    def collect_current_completed(self, league_id: int, season: int,
                               data_types: List[str] = None) -> Dict:
        """Collect completed games for current season"""
        self.logger.info(f"Starting current completed collection for league {league_id}, season {season}")
        
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        
        # Get dates for completed games (from season start to yesterday)
        season_start = datetime(season, league_info['start_month'], 1)
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        self.logger.info(f"Collecting completed games for {league_info['name']} {season_name}")
        self.logger.info(f"Date range: {season_start.strftime('%Y-%m-%d')} to {end_date}")
        
        return self._collect_data(
            league_id, season, season_start.strftime('%Y-%m-%d'), end_date, 
            data_types, is_current_season=True, status_filter=['FT', 'AET', 'PEN']
        )

    def collect_current_future(self, league_id: int, season: int,
                            data_types: List[str] = None, days_ahead: int = 3) -> Dict:
        """Collect future games (NS status) for current season"""
        self.logger.info(f"Starting current future collection for league {league_id}, season {season}")
        
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        
        # Get dates for next few days
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        self.logger.info(f"Collecting future games for {league_info['name']} {season_name}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        result = self._collect_data(
            league_id, season, start_date, end_date, 
            data_types, is_current_season=True, status_filter=['NS']
        )
        
        # Update progress with future fixture IDs
        if result['status'] == 'success':
            progress = self.load_progress(league_id, season_name) or {}
            future_fixtures = result.get('fixture_ids', [])
            progress['future_fixtures'] = future_fixtures
            progress['last_future_collection'] = datetime.now().isoformat()
            self.save_progress(league_id, season_name, progress)
            self.logger.info(f"Saved {len(future_fixtures)} future fixtures to progress")
        
        return result

    def update_completed_future(self, league_id: int, season: int,
                             data_types: List[str] = None) -> Dict:
        """Update previously collected future games that are now completed"""
        self.logger.info(f"Starting update completed future for league {league_id}, season {season}")
        
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        
        progress = self.load_progress(league_id, season_name)
        if not progress or 'future_fixtures' not in progress:
            self.logger.warning("No future fixtures to update")
            return {'status': 'error', 'message': 'No future fixtures to update'}
        
        future_fixtures = progress['future_fixtures']
        if not future_fixtures:
            self.logger.info("No future fixtures to update")
            return {'status': 'success', 'message': 'No future fixtures to update'}
        
        self.logger.info(f"Checking {len(future_fixtures)} future fixtures for completion")
        
        # Get current status of future fixtures
        current_status = self._get_current_fixtures_status(league_id, season, future_fixtures)
        completed_fixtures = [fid for fid, status in current_status.items() 
                            if status != 'NS' and fid in future_fixtures]
        
        if not completed_fixtures:
            self.logger.info("No future fixtures have been completed yet")
            return {'status': 'success', 'message': 'No future fixtures have been completed yet'}
        
        self.logger.info(f"Updating {len(completed_fixtures)} completed future games")
        
        # Collect data for completed fixtures
        result = self._collect_data_for_fixtures(
            league_id, season, completed_fixtures, data_types, is_current_season=True
        )
        
        # Update progress
        remaining_future = [fid for fid in future_fixtures if fid not in completed_fixtures]
        progress['future_fixtures'] = remaining_future
        self.save_progress(league_id, season_name, progress)
        self.logger.info(f"Updated progress: {len(remaining_future)} future fixtures remaining")
        
        return result

    def _collect_data(self, league_id: int, season: int, start_date: str, end_date: str,
                    data_types: List[str], is_current_season: bool, 
                    status_filter: List[str] = None) -> Dict:
        """Main data collection method"""
        self.logger.info(f"Starting data collection for league {league_id}, season {season}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Data types: {data_types}")
        self.logger.info(f"Is current season: {is_current_season}")
        
        if data_types is None:
            data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                         'lineups', 'injures', 'odds']

        try:
            # Fetch fixtures
            self.logger.info("Fetching fixture events...")
            fixtures_response = self.fetch_fixture_events(league_id, season, start_date, end_date)
            fixtures = fixtures_response.get('response', [])
            
            if not fixtures:
                self.logger.warning("No fixtures found in the specified date range")
                return {'status': 'success', 'message': 'No fixtures found', 'fixture_ids': []}
            
            self.logger.info(f"Found {len(fixtures)} fixtures")
            
            # Filter by status if specified
            if status_filter:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] in status_filter]
                self.logger.info(f"Filtered to {len(fixtures)} fixtures with status {status_filter}")
            
            fixture_ids = [f['fixture']['id'] for f in fixtures]
            
            # Get data directory
            country, league_info = self.get_league_info(league_id)
            season_name, _, _ = self.calculate_season_dates(country, league_info, season)
            data_dir = self._get_data_directory(league_id, season_name)
            
            # Save fixtures
            self.logger.info("Saving fixture events...")
            self._save_data(fixtures, 'fixture_events', data_dir, is_current_season)
            
            # Collect other data types
            results = {}
            for data_type in data_types:
                if data_type == 'fixture_events':
                    continue
                
                self.logger.info(f"Processing data type: {data_type}")
                
                if data_type == 'team_standings':
                    standings = self.fetch_team_standings(league_id, season)
                    self._save_data(standings, 'team_standings', data_dir, is_current_season)
                    results[data_type] = len(standings.get('response', []))
                    self.logger.info(f"Saved {results[data_type]} team standings")
                else:
                    # Map data type to fetch method
                    fetch_methods = {
                        'team_statistics': (self.fetch_all_team_statistics, fixture_ids),
                        'player_statistics': (self.fetch_all_player_statistics, fixture_ids),
                        'odds': (self.fetch_all_odds, fixture_ids),
                        'lineups': (self.fetch_all_lineups, fixture_ids),
                        'injuries': (self.fetch_all_injuries, fixture_ids),
                    }
                    
                    if data_type in fetch_methods:
                        method, ids = fetch_methods[data_type]
                        data = method(ids)
                        self._save_data(data, data_type, data_dir, is_current_season)
                        results[data_type] = len(data)
                        self.logger.info(f"Saved {results[data_type]} {data_type} records")
            
            self.logger.info("Data collection completed successfully")
            return {
                'status': 'success',
                'fixture_ids': fixture_ids,
                'data_collected': results,
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            self.logger.error(f"Collection failed: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _collect_data_for_fixtures(self, league_id: int, season: int, fixture_ids: List[int],
                                 data_types: List[str], is_current_season: bool) -> Dict:
        """Collect data for specific fixture IDs"""
        self.logger.info(f"Collecting data for {len(fixture_ids)} specific fixtures")
        
        if not fixture_ids:
            self.logger.warning("No fixtures to process")
            return {'status': 'success', 'message': 'No fixtures to process'}
        
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        data_dir = self._get_data_directory(league_id, season_name)
        
        results = {}
        for data_type in data_types:
            if data_type == 'fixture_events':
                continue
                
            self.logger.info(f"Processing {data_type} for {len(fixture_ids)} fixtures")
            
            if data_type == 'team_standings':
                standings = self.fetch_team_standings(league_id, season)
                self._save_data(standings, 'team_standings', data_dir, is_current_season)
                results[data_type] = len(standings.get('response', []))
                self.logger.info(f"Saved {results[data_type]} team standings")
            else:
                fetch_methods = {
                    'team_statistics': self.fetch_all_team_statistics,
                    'player_statistics': self.fetch_all_player_statistics,
                    'odds': self.fetch_all_odds,
                    'lineups': self.fetch_all_lineups,
                    'injuries': self.fetch_all_injuries,
                }
                
                if data_type in fetch_methods:
                    method = fetch_methods[data_type]
                    data = method(fixture_ids)
                    self._save_data(data, data_type, data_dir, is_current_season)
                    results[data_type] = len(data)
                    self.logger.info(f"Saved {results[data_type]} {data_type} records")
        
        self.logger.info("Fixture-specific data collection completed")
        return {
            'status': 'success',
            'fixture_ids': fixture_ids,
            'data_collected': results
        }

    def _save_data(self, data: any, data_type: str, data_dir: str, is_current_season: bool):
        """Save data with appropriate strategy (merge for current season)"""
        self.logger.debug(f"Saving {data_type} data (current season: {is_current_season})")
        
        os.makedirs(data_dir, exist_ok=True)
        
        if is_current_season:
            # Save daily snapshot
            daily_file = os.path.join(data_dir, f"{data_type}_{datetime.now().strftime('%Y%m%d')}.json")
            with open(daily_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved daily snapshot: {daily_file}")
            
            # Merge with main file
            main_file = os.path.join(data_dir, f"{data_type}.json")
            if os.path.exists(main_file):
                with open(main_file, 'r') as f:
                    existing_data = json.load(f)
                
                # Merge strategy depends on data type
                if data_type == 'fixture_events':
                    merged_data = self._merge_fixture_events(existing_data, data)
                else:
                    merged_data = self._merge_statistical_data(existing_data, data)
                    
                self.logger.debug(f"Merged {data_type} data with existing file")
            else:
                merged_data = data
                self.logger.debug(f"Created new {data_type} file")
            
            with open(main_file, 'w') as f:
                json.dump(merged_data, f, indent=2)
            self.logger.info(f"Saved merged {data_type} data: {main_file}")
        else:
            # For past seasons, just overwrite
            file_path = os.path.join(data_dir, f"{data_type}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {data_type} data: {file_path}")

    def _merge_fixture_events(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge fixture events, replacing old data with new"""
        existing_ids = {f['fixture']['id'] for f in existing}
        merged = existing.copy()
        
        for new_fixture in new:
            fixture_id = new_fixture['fixture']['id']
            if fixture_id in existing_ids:
                # Replace existing fixture
                merged = [f for f in merged if f['fixture']['id'] != fixture_id]
                self.logger.debug(f"Replaced fixture {fixture_id} in merge")
            merged.append(new_fixture)
        
        self.logger.info(f"Merged fixture events: {len(existing)} existing + {len(new)} new = {len(merged)} total")
        return merged

    def _merge_statistical_data(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge statistical data"""
        existing_ids = {item.get('fixture', {}).get('id') for item in existing if 'fixture' in item}
        
        merged = existing.copy()
        for new_item in new:
            fixture_id = new_item.get('fixture', {}).get('id')
            if fixture_id and fixture_id in existing_ids:
                # Replace existing data for this fixture
                merged = [item for item in merged 
                         if item.get('fixture', {}).get('id') != fixture_id]
                self.logger.debug(f"Replaced statistical data for fixture {fixture_id}")
            merged.append(new_item)
        
        self.logger.info(f"Merged statistical data: {len(existing)} existing + {len(new)} new = {len(merged)} total")
        return merged

    def automated_collection(self, league_id: int, season: int, 
                        data_types: List[str] = None,
                        start_date: str = None,
                        end_date: str = None) -> Dict:
        """
        Automated collection that determines the right mode and executes it
        """
        self.logger.info(f"Starting automated collection for league {league_id}, season {season}")
        
        # If custom dates provided, use them instead of automatic calculation
        if start_date and end_date:
            self.logger.info(f"Using custom date range: {start_date} to {end_date}")
            return self._collect_data(
                league_id, season, start_date, end_date, data_types, 
                is_current_season=False  # Treat as past season when using custom dates
            )
        
        mode = self.determine_collection_mode(league_id, season)
        self.logger.info(f"Selected collection mode: {mode}")
        
        if mode == 'past_season':
            result = self.collect_past_season(league_id, season, data_types)
        elif mode == 'current_completed':
            result = self.collect_current_completed(league_id, season, data_types)
        elif mode == 'current_future':
            result = self.collect_current_future(league_id, season, data_types)
        elif mode == 'update_completed':
            result = self.update_completed_future(league_id, season, data_types)
        else:
            result = {'status': 'error', 'message': f'Unknown collection mode: {mode}'}
        
        self.logger.info(f"Automated collection completed: {result['status']}")
        return result

# Usage examples
if __name__ == "__main__":
    # Initialize with silent logging (files only)
    collector = FootballDataCollector(api_key="your_api_key_here")
    
    
    # Example 3: Get available categories
    categories = collector.get_all_categories()
    #print("Available categories:", categories)
    
    # Example 4: Single league automated collection
    single_result = collector.automated_collection(
        league_id=140,  # Premier League
        season=2022,
        data_types=['fixture_events'],
        start_date="2022-08-01",  # Custom start date
    end_date="2023-06-01"     # Custom end date
    )

    # 2. Second run - collect completed games for current season
   # result2 = collector.automated_collection(league_id=39, season=2024)
   # print(f"Current season completed: {result2}")
    
    # 3. Third run - collect future games
   # result3 = collector.automated_collection(league_id=39, season=2024)
   # print(f"Future games: {result3}")
    
    # 4. Fourth run - update completed future games
    #result4 = collector.automated_collection(league_id=39, season=2024)
    #print(f"Updated completed: {result4}")