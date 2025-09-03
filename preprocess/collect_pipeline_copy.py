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
from typing import Dict, List, Optional, Union, Set
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming LEAGUES is defined in utils.py - adding a placeholder
LEAGUES = {
    'England': {
        '39': {'name': 'Premier League', 'start_month': 8, 'season_months': 9},
        '40': {'name': 'Championship', 'start_month': 8, 'season_months': 9}
    },
    # Add other countries and leagues as needed
}

class FootballDataCollector:
    """Class to handle automated football data collection with progress tracking"""
    
    def __init__(self, api_key: str, base_path: str = "data/raw"):
        self.API_HOST = "v3.football.api-sports.io"
        self.API_KEY = api_key
        self.base_path = base_path
        self.rate_limiter = self.RateLimiter()
        
        # Ensure base directory exists
        os.makedirs(base_path, exist_ok=True)
    
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
                    logger.info(f"Per-minute rate limit reached. Waiting for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                # Reset after waiting
                self.remaining_per_minute = self.per_minute_limit
                self.minute_window_start = time.time()

            # Decrement counters for this request
            self.remaining_daily -= 1
            self.remaining_per_minute -= 1

        def update_rate_limits(self, headers):
            """
            Update rate limits from response headers.
            """
            if headers is None:
                return
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

    def make_api_request(self, endpoint: str) -> Dict:
        """Make API request with rate limiting"""
        conn = http.client.HTTPSConnection(self.API_HOST)
        headers = {
            'x-rapidapi-host': self.API_HOST,
            'x-rapidapi-key': self.API_KEY
        }

        self.rate_limiter.check_rate_limit()

        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()

        try:
            response_data = json.loads(data)
        except Exception as e:
            raise Exception(f"Failed to parse JSON: {e} | Response: {data[:200]}")

        if res.status == 200:
            self.rate_limiter.update_rate_limits(dict(res.getheaders()))
            return response_data
        else:
            raise Exception(f"API request failed with status {res.status}: {response_data}")

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

    # Helper methods
    @staticmethod
    def get_league_info(league_id: int) -> tuple:
        league_id_str = str(league_id)
        for country, leagues in LEAGUES.items():
            if league_id_str in leagues:
                return country, leagues[league_id_str]
        raise ValueError(f"League ID {league_id} not found")

    @staticmethod
    def calculate_season_dates(country_name: str, league_info: Dict, season_year: int) -> tuple:
        start_month = league_info.get('start_month', 8)
        start_date = datetime(season_year, start_month, 1).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
                   relativedelta(months=+league_info['season_months'])).strftime('%Y-%m-%d')
        season_name = f"{season_year}-{season_year + 1}"
        return season_name, start_date, end_date

    def load_progress(self, league_id: int, season: str) -> Optional[Dict]:
        """Load progress for specific league and season"""
        progress_file = self._get_progress_file_path(league_id, season)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
        return None

    def save_progress(self, league_id: int, season: str, progress_data: Dict):
        """Save progress for specific league and season"""
        progress_file = self._get_progress_file_path(league_id, season)
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

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
        return os.path.join(self.base_path, country, league_name, season)

    def determine_collection_mode(self, league_id: int, season: int) -> str:
        """
        Determine which collection mode to use based on current state
        Returns: 'past_season', 'current_completed', 'current_future', 'update_completed'
        """
        current_year = datetime.now().year
        country, league_info = self.get_league_info(league_id)
        season_name, start_date, end_date = self.calculate_season_dates(country, league_info, season)
        
        # Check if this is current season
        is_current_season = str(current_year) in season_name
        
        if not is_current_season:
            return 'past_season'
        
        # Check progress to determine current season mode
        progress = self.load_progress(league_id, season_name)
        
        if not progress:
            # First time collecting current season
            return 'current_completed'
        
        # Check if we have future games that need updating
        if 'future_fixtures' in progress and progress['future_fixtures']:
            # Check if any future fixtures have become completed
            future_ids = progress['future_fixtures']
            current_fixtures = self._get_current_fixtures_status(league_id, season, future_ids)
            
            completed_future = [fid for fid, status in current_fixtures.items() 
                              if status != 'NS' and fid in future_ids]
            
            if completed_future:
                return 'update_completed'
        
        # Check if we should collect future games
        last_future_collection = progress.get('last_future_collection')
        if last_future_collection:
            last_date = datetime.fromisoformat(last_future_collection)
            if (datetime.now() - last_date).days >= 1:  # Collect future games daily
                return 'current_future'
        
        return 'current_completed'  # Default to collecting completed games

    def _get_current_fixtures_status(self, league_id: int, season: int, fixture_ids: List[int]) -> Dict:
        """Get current status of specific fixtures"""
        # This would typically involve checking the API for current fixture status
        # For simplicity, we'll return a mock response
        return {fid: 'FT' for fid in fixture_ids}  # Mock - all completed

    def collect_past_season(self, league_id: int, season: int, 
                          data_types: List[str] = None) -> Dict:
        """Collect data for past seasons (complete season data)"""
        country, league_info = self.get_league_info(league_id)
        season_name, start_date, end_date = self.calculate_season_dates(country, league_info, season)
        
        logger.info(f"Collecting past season data for {league_info['name']} {season_name}")
        
        return self._collect_data(
            league_id, season, start_date, end_date, data_types, 
            is_current_season=False
        )

    def collect_current_completed(self, league_id: int, season: int,
                               data_types: List[str] = None) -> Dict:
        """Collect completed games for current season"""
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        
        # Get dates for completed games (from season start to yesterday)
        season_start = datetime(season, league_info['start_month'], 1)
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Collecting completed games for {league_info['name']} {season_name}")
        
        return self._collect_data(
            league_id, season, season_start.strftime('%Y-%m-%d'), end_date, 
            data_types, is_current_season=True, status_filter=['FT', 'AET', 'PEN']
        )

    def collect_current_future(self, league_id: int, season: int,
                            data_types: List[str] = None, days_ahead: int = 3) -> Dict:
        """Collect future games (NS status) for current season"""
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        
        # Get dates for next few days
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        logger.info(f"Collecting future games for {league_info['name']} {season_name}")
        
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
        
        return result

    def update_completed_future(self, league_id: int, season: int,
                             data_types: List[str] = None) -> Dict:
        """Update previously collected future games that are now completed"""
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        
        progress = self.load_progress(league_id, season_name)
        if not progress or 'future_fixtures' not in progress:
            return {'status': 'error', 'message': 'No future fixtures to update'}
        
        future_fixtures = progress['future_fixtures']
        if not future_fixtures:
            return {'status': 'success', 'message': 'No future fixtures to update'}
        
        # Get current status of future fixtures
        current_status = self._get_current_fixtures_status(league_id, season, future_fixtures)
        completed_fixtures = [fid for fid, status in current_status.items() 
                            if status != 'NS' and fid in future_fixtures]
        
        if not completed_fixtures:
            return {'status': 'success', 'message': 'No future fixtures have been completed yet'}
        
        logger.info(f"Updating {len(completed_fixtures)} completed future games")
        
        # Collect data for completed fixtures
        result = self._collect_data_for_fixtures(
            league_id, season, completed_fixtures, data_types, is_current_season=True
        )
        
        # Update progress
        remaining_future = [fid for fid in future_fixtures if fid not in completed_fixtures]
        progress['future_fixtures'] = remaining_future
        self.save_progress(league_id, season_name, progress)
        
        return result

    def _collect_data(self, league_id: int, season: int, start_date: str, end_date: str,
                    data_types: List[str], is_current_season: bool, 
                    status_filter: List[str] = None) -> Dict:
        """Main data collection method"""
        if data_types is None:
            data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                         'lineups', 'injuries', 'odds']

        try:
            # Fetch fixtures
            fixtures_response = self.fetch_fixture_events(league_id, season, start_date, end_date)
            fixtures = fixtures_response.get('response', [])
            
            if not fixtures:
                return {'status': 'success', 'message': 'No fixtures found', 'fixture_ids': []}
            
            # Filter by status if specified
            if status_filter:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] in status_filter]
            
            fixture_ids = [f['fixture']['id'] for f in fixtures]
            
            # Get data directory
            country, league_info = self.get_league_info(league_id)
            season_name, _, _ = self.calculate_season_dates(country, league_info, season)
            data_dir = self._get_data_directory(league_id, season_name)
            
            # Save fixtures
            self._save_data(fixtures, 'fixture_events', data_dir, is_current_season)
            
            # Collect other data types
            results = {}
            for data_type in data_types:
                if data_type == 'fixture_events':
                    continue
                
                if data_type == 'team_standings':
                    standings = self.fetch_team_standings(league_id, season)
                    self._save_data(standings, 'team_standings', data_dir, is_current_season)
                    results[data_type] = len(standings.get('response', []))
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
            
            return {
                'status': 'success',
                'fixture_ids': fixture_ids,
                'data_collected': results,
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def _collect_data_for_fixtures(self, league_id: int, season: int, fixture_ids: List[int],
                                 data_types: List[str], is_current_season: bool) -> Dict:
        """Collect data for specific fixture IDs"""
        if not fixture_ids:
            return {'status': 'success', 'message': 'No fixtures to process'}
        
        country, league_info = self.get_league_info(league_id)
        season_name, _, _ = self.calculate_season_dates(country, league_info, season)
        data_dir = self._get_data_directory(league_id, season_name)
        
        results = {}
        for data_type in data_types:
            if data_type == 'fixture_events':
                continue
                
            if data_type == 'team_standings':
                standings = self.fetch_team_standings(league_id, season)
                self._save_data(standings, 'team_standings', data_dir, is_current_season)
                results[data_type] = len(standings.get('response', []))
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
        
        return {
            'status': 'success',
            'fixture_ids': fixture_ids,
            'data_collected': results
        }

    def _save_data(self, data: any, data_type: str, data_dir: str, is_current_season: bool):
        """Save data with appropriate strategy (merge for current season)"""
        os.makedirs(data_dir, exist_ok=True)
        
        if is_current_season:
            # Save daily snapshot
            daily_file = os.path.join(data_dir, f"{data_type}_{datetime.now().strftime('%Y%m%d')}.json")
            with open(daily_file, 'w') as f:
                json.dump(data, f, indent=2)
            
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
            else:
                merged_data = data
            
            with open(main_file, 'w') as f:
                json.dump(merged_data, f, indent=2)
        else:
            # For past seasons, just overwrite
            file_path = os.path.join(data_dir, f"{data_type}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

    def _merge_fixture_events(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge fixture events, replacing old data with new"""
        existing_ids = {f['fixture']['id'] for f in existing}
        merged = existing.copy()
        
        for new_fixture in new:
            fixture_id = new_fixture['fixture']['id']
            if fixture_id in existing_ids:
                # Replace existing fixture
                merged = [f for f in merged if f['fixture']['id'] != fixture_id]
            merged.append(new_fixture)
        
        return merged

    def _merge_statistical_data(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge statistical data"""
        # This is a simple implementation - you might want more sophisticated merging
        existing_ids = {item.get('fixture', {}).get('id') for item in existing if 'fixture' in item}
        
        merged = existing.copy()
        for new_item in new:
            fixture_id = new_item.get('fixture', {}).get('id')
            if fixture_id and fixture_id in existing_ids:
                # Replace existing data for this fixture
                merged = [item for item in merged 
                         if item.get('fixture', {}).get('id') != fixture_id]
            merged.append(new_item)
        
        return merged

    def automated_collection(self, league_id: int, season: int, 
                           data_types: List[str] = None) -> Dict:
        """
        Automated collection that determines the right mode and executes it
        """
        mode = self.determine_collection_mode(league_id, season)
        logger.info(f"Automated collection mode: {mode}")
        
        if mode == 'past_season':
            return self.collect_past_season(league_id, season, data_types)
        elif mode == 'current_completed':
            return self.collect_current_completed(league_id, season, data_types)
        elif mode == 'current_future':
            return self.collect_current_future(league_id, season, data_types)
        elif mode == 'update_completed':
            return self.update_completed_future(league_id, season, data_types)
        else:
            return {'status': 'error', 'message': f'Unknown collection mode: {mode}'}

# Usage example
if __name__ == "__main__":
    # Initialize collector
    collector = FootballDataCollector(api_key="25c02ce9f07df0edc1e69866fbe7d156")
    
    # Example usage scenarios:
    
    # 1. First run - collect past seasons
    result1 = collector.automated_collection(league_id=39, season=2017)
    print(f"Past season collection: {result1}")
    
