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
import calendar
import os
from typing import Dict, List, Optional, Union, Set, Any, Tuple
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

class CollectPipeline:
    def __init__(self, api_key, base_path="data/raw", log_dir="logs/collection"):
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
        if league_id is None:
            self.logger.error("League ID is None in get_league_info")
            raise ValueError("League ID cannot be None")
        
        league_id = str(league_id)
        
        # Debug: print what we're looking for
        self.logger.debug(f"Looking for league ID: {league_id}")
        
        # First check top-level leagues
        for country, leagues in LEAGUES.items():
            if isinstance(leagues, dict):
                if league_id in leagues:
                    self.logger.debug(f"Found in top-level: {country} -> {leagues[league_id]}")
                    return country, leagues[league_id]
        
        # Check nested structures
        for category, countries in LEAGUES.items():
            if isinstance(countries, dict):
                for country_name, leagues in countries.items():
                    if isinstance(leagues, dict) and league_id in leagues:
                        self.logger.debug(f"Found in nested: {category} -> {country_name} -> {leagues[league_id]}")
                        return country_name, leagues[league_id]
        
        raise ValueError(f"League ID {league_id} not found in any country")

    def get_all_leagues(self):
        """
        Return leagues grouped by region and then by country for unified filtering
        """
        regions = {}
        
        def extract_leagues(leagues_dict, current_region=None, current_country=None):
            for key, value in leagues_dict.items():
                if isinstance(value, dict):
                    if 'name' in value and 'category' in value:
                        # This is a league entry
                        league_data = {
                            'id': key,
                            'name': value['name'],
                            'category': value.get('category', ''),
                            'country': current_country if current_country else 'Unknown',
                            'region': current_region if current_region else 'Other'
                        }
                        
                        # Add to region structure
                        region_name = current_region if current_region else 'Other Regions'
                        if region_name not in regions:
                            regions[region_name] = {}
                        
                        country_name = current_country if current_country else 'Other Countries'
                        if country_name not in regions[region_name]:
                            regions[region_name][country_name] = []
                        
                        regions[region_name][country_name].append(league_data)
                    else:
                        # Determine if this is a region or country
                        if key in ['Top 5', 'Western Europe', 'Eastern Europe', 'Scandinavia', 'Americas', 'Asia']:
                            # This is a region
                            extract_leagues(value, key, None)
                        else:
                            # This is a country within the current region
                            extract_leagues(value, current_region, key)
        
        extract_leagues(LEAGUES)
        
        # Sort regions, countries, and leagues
        sorted_regions = {}
        for region in sorted(regions.keys()):
            sorted_regions[region] = {}
            for country in sorted(regions[region].keys()):
                sorted_leagues = sorted(regions[region][country], key=lambda x: x['name'])
                sorted_regions[region][country] = sorted_leagues
        
        return sorted_regions

    def get_league_dates(self, league_id, season):
        """Get proper date ranges for a specific league and season"""
        try:
            country, league_info = self.get_league_info(str(league_id))
            league_name = league_info.get('name', 'Unknown')
            
            self.logger.info(f"Getting dates for {league_name} ({country}) - Season {season}")
            
            # Check if league has specific date ranges defined (highest priority)
            if 'date_ranges' in league_info and str(season) in league_info['date_ranges']:
                date_range = league_info['date_ranges'][str(season)]
                start_date = date_range.get('start')
                end_date = date_range.get('end')
                
                if start_date and end_date:
                    self.logger.info(f"Using predefined dates for {league_name} {season}: {start_date} to {end_date}")
                    return start_date, end_date
            
            # Use automatic date calculation if league has the required metadata
            if all(key in league_info for key in ['start_month', 'season_months']):
                season_name, start_date, end_date = self.calculate_season_dates(country, league_info, season)
                self.logger.info(f"Using automatic dates for {league_name}: {start_date} to {end_date}")
                return start_date, end_date
            
            # Special handling for cup competitions (fallback)
            if league_info.get('category') == 'domestic_cup':
                # Cup competitions typically run from September to May/June
                start_date = f"{season}-09-01"  # September 1st
                end_date = f"{season+1}-05-31"   # May 31st of next year
                self.logger.info(f"Using cup competition dates for {league_name}: {start_date} to {end_date}")
                return start_date, end_date
            
            # Special handling for top tier leagues (fallback)
            elif league_info.get('category') == 'top_tier':
                # Most leagues run from August to May
                start_date = f"{season}-08-01"   # August 1st
                end_date = f"{season+1}-05-31"   # May 31st of next year
                self.logger.info(f"Using top tier league dates for {league_name}: {start_date} to {end_date}")
                return start_date, end_date
            
            # Default fallback
            start_date = f"{season}-08-01"
            end_date = f"{season+1}-05-31"
            self.logger.info(f"Using default dates for {league_name}: {start_date} to {end_date}")
            return start_date, end_date
            
        except Exception as e:
            self.logger.warning(f"Error getting dates for league {league_id}: {e}")
            # Fallback to default dates
            start_date = f"{season}-08-01"
            end_date = f"{season+1}-05-31"
            return start_date, end_date
        
    def calculate_season_dates(self, country_name: str, league_info: Dict, season_year: int) -> Tuple[str, str, str]:
        """Calculate automatic season dates based on league characteristics"""
        try:
            # Get start month (default to August if not specified)
            start_month = league_info.get('start_month', 8)
            
            # Use the 1st day of the month as start day
            start_day = 1
            
            # Calculate start date
            start_date = datetime(season_year, start_month, start_day)
            
            # Calculate end date by adding season_months and going to end of month
            season_months = league_info.get('season_months', 10)  # Default to 10 months
            end_date = start_date + relativedelta(months=+season_months, days=-1)
            
            # Format dates as strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Determine if season spans across two years
            if start_date.year != end_date.year:
                season_name = f"{season_year}"
            else:
                season_name = f"{season_year}"
            
            return season_name, start_date_str, end_date_str
            
        except Exception as e:
            self.logger.error(f"Error in calculate_season_dates: {e}")
            # Fallback to default calculation
            start_date_str = f"{season_year}-08-01"
            end_date_str = f"{season_year + 1}-05-31"
            return f"{season_year}", start_date_str, end_date_str        
        
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

    def fetch_fixture_by_id(self, fixture_id):
        """Fetch a single fixture by its ID"""
        try:
            endpoint = f"/fixtures?id={fixture_id}"
            response = self.make_api_request(endpoint)
            return response
        except Exception as e:
            self.logger.error(f"Failed to fetch fixture {fixture_id}: {str(e)}")
            return None

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
        future_date = today + timedelta(days=7)  # Look 30 days ahead
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
    



    def _update_fixture_in_file(self, fixture_path, updated_fixture):
        """
        Update a specific fixture in the fixture_events.json file with cleaned data
        """
        fixture_id = str(updated_fixture['fixture']['id'])
        
        # Load existing fixtures
        with open(fixture_path, 'r') as f:
            all_fixtures = json.load(f)
        
        # Find and replace the fixture
        for i, fixture in enumerate(all_fixtures):
            if str(fixture['fixture']['id']) == fixture_id:
                all_fixtures[i] = updated_fixture
                break
        
        # Save updated fixtures
        with open(fixture_path, 'w') as f:
            json.dump(all_fixtures, f, indent=2)
        
        self.logger.info(f"Updated fixture {fixture_id} in file")

    def collect_missing_data_for_fixtures(self, raw_path, fixture_ids, data_types=None, batch_size=10):
        """
        Collect missing data types for specific fixture IDs
        """
        if data_types is None:
            data_types = ['team_statistics', 'player_statistics', 'lineups', 'injuries', 'odds']
        
        # Mapping of data types to their fetch methods
        fetch_functions = {
            'team_statistics': self.fetch_team_statistics,
            'player_statistics': self.fetch_player_statistics,
            'lineups': self.fetch_lineups,
            'injuries': self.fetch_injuries,
            'odds': self.fetch_odds,
        }
        
        # Check which data types already exist
        existing_data = {}
        for data_type in data_types:
            file_path = os.path.join(raw_path, f"{data_type}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        existing_data[data_type] = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data[data_type] = []
            else:
                existing_data[data_type] = []
        
        # Collect missing data
        for data_type in data_types:
            if data_type not in fetch_functions:
                continue
                
            # Find fixtures that need this data type
            existing_fixture_ids = {str(item.get('fixture', {}).get('id')) for item in existing_data[data_type]}
            missing_fixtures = [fid for fid in fixture_ids if fid not in existing_fixture_ids]
            
            if not missing_fixtures:
                self.logger.info(f"All {len(fixture_ids)} fixtures already have {data_type} data")
                continue
                
            self.logger.info(f"Collecting {data_type} for {len(missing_fixtures)} fixtures")
            
            # Process in batches
            collected_data = []
            for i in range(0, len(missing_fixtures), batch_size):
                batch = missing_fixtures[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1} for {data_type}")
                
                for fixture_id in batch:
                    try:
                        data = fetch_functions[data_type](int(fixture_id))
                        if data:
                            collected_data.append(data)
                            self.logger.info(f"Collected {data_type} for fixture {fixture_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to fetch {data_type} for fixture {fixture_id}: {str(e)}")
            
            # Save collected data
            if collected_data:
                # Combine with existing data
                all_data = existing_data[data_type] + collected_data
                
                # Save to file
                file_path = os.path.join(raw_path, f"{data_type}.json")
                with open(file_path, 'w') as f:
                    json.dump(all_data, f, indent=2)
                
                self.logger.info(f"Saved {len(all_data)} {data_type} records to {file_path}")

            
    def update_ns_fixtures(self, raw_path, progress):
        """
        Update previously collected NS (Not Started) fixtures
        by checking their current status and fetching updated data
        
        Returns:
            tuple: (updated_progress, updated_fixture_ids) - list of fixture IDs that were updated
        """
        fixture_path = os.path.join(raw_path, "fixture_events.json")
        
        if not os.path.exists(fixture_path):
            self.logger.warning(f"No fixture file found at {fixture_path}")
            return progress, []
        
        # Load all fixtures from the file
        with open(fixture_path, 'r') as f:
            all_fixtures = json.load(f)
        
        # Find NS fixtures in the actual file data
        ns_fixtures_in_file = []
        for fixture in all_fixtures:
            if fixture['fixture']['status']['short'] == 'NS':
                fixture_id = str(fixture['fixture']['id'])
                ns_fixtures_in_file.append(fixture_id)
        
        if not ns_fixtures_in_file:
            self.logger.info("No NS fixtures found in the fixture file")
            return progress, []
        
        self.logger.info(f"Found {len(ns_fixtures_in_file)} NS fixtures in file: {ns_fixtures_in_file}")
        
        # Update progress with NS fixtures from file - initialize if empty
        if 'ns_fixtures' not in progress or not progress['ns_fixtures']:
            progress['ns_fixtures'] = ns_fixtures_in_file
            self.logger.info(f"Initialized ns_fixtures in progress with {len(ns_fixtures_in_file)} fixtures")
        else:
            # Add any new NS fixtures to progress tracking
            current_ns_fixtures = set(progress.get('ns_fixtures', []))
            new_ns_fixtures = set(ns_fixtures_in_file) - current_ns_fixtures
            
            if new_ns_fixtures:
                progress['ns_fixtures'] = list(current_ns_fixtures.union(new_ns_fixtures))
                self.logger.info(f"Added {len(new_ns_fixtures)} new NS fixtures to progress tracking")
        
        # Check which NS fixtures need updating (those that might have started)
        fixtures_to_update = []
        for fixture_id in progress['ns_fixtures']:
            # Check if fixture is still in the file and still NS
            fixture_in_file = next((f for f in all_fixtures if str(f['fixture']['id']) == fixture_id), None)
            if fixture_in_file and fixture_in_file['fixture']['status']['short'] == 'NS':
                fixtures_to_update.append(fixture_id)
        
        if not fixtures_to_update:
            self.logger.info("No NS fixtures need updating")
            return progress, []
        
        self.logger.info(f"Checking status of {len(fixtures_to_update)} NS fixtures")
        
        # Extract league_id and season from the first fixture (they should all be the same)
        first_fixture = next((f for f in all_fixtures if str(f['fixture']['id']) in fixtures_to_update), None)
        if not first_fixture:
            self.logger.warning("No valid fixtures found for updating")
            return progress, []
        
        league_id = first_fixture['league']['id']
        season = first_fixture['league']['season']
        
        # Instead of fetching by individual IDs, get date range from NS fixtures
        # Find the earliest and latest dates among NS fixtures
        ns_fixture_dates = []
        for fixture_id in fixtures_to_update:
            fixture_data = next((f for f in all_fixtures if str(f['fixture']['id']) == fixture_id), None)
            if fixture_data:
                fixture_date = fixture_data['fixture']['date']
                ns_fixture_dates.append(fixture_date)
        
        if not ns_fixture_dates:
            self.logger.info("No valid dates found for NS fixtures")
            return progress, []
        
        # Convert to datetime objects for comparison
        date_objects = [datetime.fromisoformat(date_str.replace('Z', '+00:00')) for date_str in ns_fixture_dates]
        earliest_date = min(date_objects)
        latest_date = max(date_objects)
        
        # Format dates for API call
        start_date_str = earliest_date.strftime('%Y-%m-%d')
        end_date_str = latest_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching fixtures by date range: {start_date_str} to {end_date_str} for league {league_id}, season {season}")
        
        try:
            # Fetch all fixtures in the date range for this league and season
            fixtures_data = self.fetch_fixture_events(
                league_id=league_id,
                season=season,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if not fixtures_data or not fixtures_data.get('response'):
                self.logger.warning("No fixtures found in the specified date range")
                return progress, []
            
            current_fixtures = fixtures_data['response']
            updated_count = 0
            updated_fixture_ids = []
            
            # Process each fixture in the response
            for current_fixture in current_fixtures:
                fixture_id = str(current_fixture['fixture']['id'])
                current_status = current_fixture['fixture']['status']['short']
                
                # Only process fixtures that were in our NS tracking
                if fixture_id in fixtures_to_update and current_status != 'NS':
                    # Fixture has started or finished, update our data
                    self._update_fixture_in_file(fixture_path, current_fixture)
                    
                    # Remove from NS tracking
                    if fixture_id in progress['ns_fixtures']:
                        progress['ns_fixtures'].remove(fixture_id)
                    
                    updated_count += 1
                    updated_fixture_ids.append(fixture_id)
                    self.logger.info(f"Updated fixture {fixture_id} - new status: {current_status}")
            
            self.logger.info(f"Updated {updated_count} fixtures using date range method")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch fixtures by date range: {str(e)}")
            # Fall back to individual fixture updates if date range method fails
            self.logger.info("Falling back to individual fixture updates")
            updated_count = 0
            updated_fixture_ids = []
            
            for fixture_id in fixtures_to_update:
                try:
                    # Fetch current fixture data - only basic fixture info, not statistics
                    fixture_data = self.fetch_fixture_by_id(int(fixture_id))
                    
                    if fixture_data and fixture_data.get('response'):
                        current_fixture = fixture_data['response'][0]
                        current_status = current_fixture['fixture']['status']['short']
                        
                        if current_status != 'NS':
                            # Fixture has started or finished, update our data
                            self._update_fixture_in_file(fixture_path, current_fixture)
                            
                            # Remove from NS tracking
                            if fixture_id in progress['ns_fixtures']:
                                progress['ns_fixtures'].remove(fixture_id)
                            
                            updated_count += 1
                            updated_fixture_ids.append(fixture_id)
                            self.logger.info(f"Updated fixture {fixture_id} - new status: {current_status}")
                        
                except Exception as inner_e:
                    self.logger.error(f"Failed to update fixture {fixture_id}: {str(inner_e)}")
        
        return progress, updated_fixture_ids


    def collect_league_data_filter(self, league_id=None, season=None, data_types=None, keep_progress=False,
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
                league_id = league_info['id']
                league_name = league_info['name']
                
                self.logger.info(f"Processing {league_name} ({country_name}) - ID: {league_id}")
                
                # Get proper date range for this specific league
                # Only use custom dates if start_date/end_date are None or empty
                if start_date is None or start_date == '' or end_date is None or end_date == '':
                    league_start_date, league_end_date = self.get_league_dates(league_id, season)
                    self.logger.info(f"Using automatic dates for {league_name}: {league_start_date} to {league_end_date}")
                else:
                    league_start_date, league_end_date = start_date, end_date
                    self.logger.info(f"Using manual dates for {league_name}: {league_start_date} to {league_end_date}")
                
                self.logger.info(f"Phase {collection_phase}: Collecting {season} season {league_start_date} to {league_end_date}")
                
                result = self._process_single_league(
                    league_id, season, data_types, keep_progress,
                    batch_size, progress_file, league_start_date, league_end_date,
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
        
        # Debug output
        self.logger.info(f"Filtering leagues with: league_id={league_id}, filter_ids={filter_ids}, "
                    f"filter_tiers={filter_tiers}, filter_cups={filter_cups}, "
                    f"filter_categories={filter_categories}")
        
        # If specific league_id is provided, use that
        if league_id is not None:
            try:
                league_id_str = str(league_id)
                country_name, league_info = self.get_league_info(league_id_str)
                if isinstance(league_info, dict):
                    league_info['id'] = league_id_str
                filtered_leagues.append((country_name, league_info))
                return filtered_leagues
            except Exception as e:
                self.logger.warning(f"League ID {league_id} not found: {str(e)}")
                return []
        
        # If only filter_ids are provided, return those leagues directly
        if filter_ids and not any([filter_tiers, filter_cups, filter_categories]):
            self.logger.info(f"Only filter_ids provided: {filter_ids}")
            for league_id in filter_ids:
                try:
                    league_id_str = str(league_id)
                    country_name, league_info = self.get_league_info(league_id_str)
                    if isinstance(league_info, dict):
                        league_info['id'] = league_id_str
                    filtered_leagues.append((country_name, league_info))
                except Exception as e:
                    self.logger.warning(f"League ID {league_id} not found: {str(e)}")
            return filtered_leagues
        
        # If no filters provided, return empty
        if not any([filter_ids, filter_tiers, filter_cups, filter_categories]):
            self.logger.warning("No filtering criteria provided")
            return []
        
        # Convert filter_ids to strings for consistent comparison
        filter_ids_str = [str(fid) for fid in filter_ids] if filter_ids else None
        
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
                        
                        # Filter by specific league IDs (compare as strings)
                        if filter_ids_str and key not in filter_ids_str:
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
        
        result = search_leagues(LEAGUES)
        self.logger.info(f"Found {len(result)} leagues matching criteria")
        return result

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
            # Phase 3: Update previously collected NS games and collect missing data
            if not is_current_season:
                self.logger.info("Phase 3: Not applicable for past seasons")
                return {'status': 'skipped', 'reason': 'Not current season', 'league_id': league_id}
            
            # This phase updates NS fixtures and collects missing data
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
            raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
            
            progress = self.load_progress(progress_file) if keep_progress else {}
            
            # Update NS fixtures and get the list of updated fixtures
            progress, updated_fixture_ids = self.update_ns_fixtures(raw_path, progress)
            
            if not updated_fixture_ids:
                self.logger.info("No fixtures were updated from NS status")
                return {
                    'status': 'success',
                    'phase': 3,
                    'message': 'No NS fixtures needed updating',
                    'league_id': league_id
                }
            
            self.logger.info(f"Found {len(updated_fixture_ids)} fixtures that were updated from NS status: {updated_fixture_ids}")
            
            # Set default data types if not provided
            if data_types is None:
                data_types = ['team_statistics', 'player_statistics', 'lineups', 'injuries', 'odds']
            
            # Collect missing data only for the updated fixtures
            self.collect_missing_data_for_fixtures(
                raw_path, updated_fixture_ids, data_types, batch_size
            )
            
            self.save_progress(progress, progress_file)
            
            return {
                'status': 'success',
                'phase': 3,
                'message': 'NS fixtures updated and missing data collected',
                'league_id': league_id,
                'updated_fixtures': len(updated_fixture_ids)
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
        
        # --- EXCLUDE ALREADY COLLECTED FIXTURES ---
        # Load existing fixture_events to identify already collected fixtures
        fixture_path = os.path.join(raw_path, "fixture_events.json")
        existing_fixture_ids = set()
        
        if os.path.exists(fixture_path):
            try:
                with open(fixture_path, 'r') as f:
                    existing_fixtures = json.load(f)
                    existing_fixture_ids = {str(f['fixture']['id']) for f in existing_fixtures}
                    self.logger.info(f"Found {len(existing_fixture_ids)} existing fixtures in file")
            except (json.JSONDecodeError, FileNotFoundError):
                existing_fixture_ids = set()
        
        # Filter out fixtures that are already collected
        new_fixtures = []
        for fixture in current_fixtures:
            fixture_id = str(fixture['fixture']['id'])
            if fixture_id not in existing_fixture_ids:
                new_fixtures.append(fixture)
        
        if not new_fixtures:
            self.logger.info("All fixtures in the date range are already collected")
            return {
                'status': 'success',
                'phase': collection_phase,
                'message': 'No new fixtures to collect',
                'league_id': league_id,
                'existing_fixtures': len(existing_fixture_ids)
            }
        
        self.logger.info(f"Found {len(new_fixtures)} new fixtures to collect (excluding {len(existing_fixture_ids)} already collected)")
        
        # --- ADD THIS CODE TO SAVE NEW FIXTURES ---
        # Save new fixtures to the fixture_events.json file
        if new_fixtures:
            if os.path.exists(fixture_path):
                # Load existing fixtures and append new ones
                with open(fixture_path, 'r') as f:
                    existing_fixtures = json.load(f)
                combined_fixtures = existing_fixtures + new_fixtures
            else:
                combined_fixtures = new_fixtures
            
            # Save back to file
            with open(fixture_path, 'w') as f:
                json.dump(combined_fixtures, f, indent=2)
            
            self.logger.info(f"Saved {len(new_fixtures)} new fixtures to {fixture_path} "
                        f"(total fixtures: {len(combined_fixtures)})")
            
            # Update progress with fixture IDs
            all_fixture_ids = [str(f['fixture']['id']) for f in new_fixtures]
            progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + all_fixture_ids))
            for fid in all_fixture_ids:
                progress['data_types']['fixture_events'][fid] = 'completed'
            
            self.save_progress(progress, progress_file)
        # --- END OF ADDED CODE ---
        
        all_fixture_ids = [str(f['fixture']['id']) for f in new_fixtures]

        # --- Process other data types (only for completed games in phase 1) ---
        if collection_phase == 1:
            collection = {dt: [] for dt in data_types if dt not in ['fixture_events', 'team_standings']}
            
            # Only process completed games from NEW fixtures
            completed_new_fixtures = [
                str(f['fixture']['id']) for f in new_fixtures 
                if f['fixture']['status']['short'] != 'NS'
            ]
            
            # Also check which fixtures already have data collected
            fixtures_to_process = {}
            for data_type in collection.keys():
                # Load existing data to check what's already collected
                data_path = os.path.join(raw_path, f"{data_type}.json")
                existing_data_ids = set()
                
                if os.path.exists(data_path):
                    try:
                        with open(data_path, 'r') as f:
                            existing_data = json.load(f)
                            existing_data_ids = {str(item.get('fixture', {}).get('id')) for item in existing_data}
                            self.logger.info(f"Found {len(existing_data_ids)} existing {data_type} records")
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_data_ids = set()
                
                # Only process fixtures that are completed AND not already collected
                fixtures_to_process[data_type] = [
                    fid for fid in completed_new_fixtures 
                    if fid not in existing_data_ids and 
                    progress['data_types'][data_type].get(fid) != 'completed'
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
            for data_type, new_data in collection.items():
                if not new_data:
                    self.logger.info(f"No new {data_type} data to save")
                    continue
                    
                out_path = os.path.join(raw_path, f'{data_type}.json')
                
                # Load existing data
                existing_data = []
                if os.path.exists(out_path):
                    with open(out_path, 'r') as f:
                        existing_data = json.load(f)
                    self.logger.info(f"Found {len(existing_data)} existing {data_type} records")
                else:
                    self.logger.info(f"No existing {data_type} file found")
                
                # Deduplicate
                existing_ids = {str(item.get('fixture', {}).get('id')) for item in existing_data}
                unique_new_data = []
                duplicate_count = 0
                
                for item in new_data:
                    fixture_id = str(item.get('fixture', {}).get('id'))
                    if fixture_id in existing_ids:
                        duplicate_count += 1
                    else:
                        unique_new_data.append(item)
                
                # Combine data
                combined_data = existing_data + unique_new_data
                
                # Save
                with open(out_path, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                
                # Detailed logging
                self.logger.info(f"{data_type}: {len(unique_new_data)} new, {duplicate_count} duplicates skipped")
                self.logger.info(f"Total {data_type} records: {len(combined_data)}")
                self.logger.info(f"Saved to {out_path}")

            # --- Handle Team Standings ---
            if 'team_standings' in data_types:
                try:
                    # Check if standings already exist and are current
                    standings_path = os.path.join(raw_path, 'team_standings.json')
                    needs_update = True
                    
                    if os.path.exists(standings_path):
                        try:
                            with open(standings_path, 'r') as f:
                                existing_standings = json.load(f)
                            # Check if standings are recent (within last 7 days)
                            if 'last_updated' in existing_standings:
                                last_updated = datetime.fromisoformat(existing_standings['last_updated'])
                                if (datetime.now() - last_updated).days < 7:
                                    needs_update = False
                                    self.logger.info("Team standings are recent, skipping update")
                        except (json.JSONDecodeError, FileNotFoundError):
                            pass
                    
                    if needs_update:
                        standings = self.fetch_team_standings(league_id, season)
                        if standings:
                            standings['last_updated'] = datetime.now().isoformat()
                            with open(standings_path, 'w') as f:
                                json.dump(standings, f, indent=2)
                            self.logger.info(f"Saved team standings to {standings_path}")
                            progress['data_types']['team_standings'] = {'status': 'completed'}
                    else:
                        progress['data_types']['team_standings'] = {'status': 'existing'}
                        
                except Exception as e:
                    self.logger.error(f"Failed to fetch/save team standings: {e}")
                    progress['data_types']['team_standings'] = {
                        'status': f'failed: {str(e)}',
                        'last_attempt': datetime.now().isoformat()
                    }
                finally:
                    self.save_progress(progress, progress_file)

        elif collection_phase == 2:
            # Phase 2: Collect specified data types for NS games
            ns_data_types = [dt for dt in data_types if dt != 'fixture_events']
            
            # FIRST: Save the NS fixtures to the fixture file
            if new_fixtures:
                # Get NS fixtures from the newly collected fixtures
                ns_fixtures_from_new = [
                    f for f in new_fixtures 
                    if f['fixture']['status']['short'] == 'NS'
                ]
                
                if ns_fixtures_from_new:
                    # Load existing fixtures
                    if os.path.exists(fixture_path):
                        with open(fixture_path, 'r') as f:
                            existing_fixtures = json.load(f)
                    else:
                        existing_fixtures = []
                    
                    # Remove any existing NS fixtures to avoid duplicates
                    existing_non_ns = [f for f in existing_fixtures if f['fixture']['status']['short'] != 'NS']
                    
                    # Combine non-NS existing fixtures with new NS fixtures
                    combined_fixtures = existing_non_ns + ns_fixtures_from_new
                    
                    # Save back to file
                    with open(fixture_path, 'w') as f:
                        json.dump(combined_fixtures, f, indent=2)
                    
                    self.logger.info(f"Saved {len(ns_fixtures_from_new)} NS fixtures to {fixture_path} "
                                f"(total fixtures: {len(combined_fixtures)})")
                    
                    # Update progress with NS fixture IDs
                    ns_fixture_ids = [str(f['fixture']['id']) for f in ns_fixtures_from_new]
                    progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + ns_fixture_ids))
                    for fid in ns_fixture_ids:
                        if fid not in progress['data_types']['fixture_events']:
                            progress['data_types']['fixture_events'][fid] = 'completed'
                    
                    self.save_progress(progress, progress_file)
            
            # SECOND: Process other data types for NS fixtures
            if ns_data_types:
                self.logger.info(f"Phase 2: Collecting {ns_data_types} data for NS fixtures")
                
                # Get ALL NS fixtures from the file (not just new ones)
                if os.path.exists(fixture_path):
                    with open(fixture_path, 'r') as f:
                        all_fixtures_data = json.load(f)
                    
                    ns_fixtures = [
                        str(f['fixture']['id']) for f in all_fixtures_data 
                        if f['fixture']['status']['short'] == 'NS'
                    ]
                else:
                    ns_fixtures = []
                
                if not ns_fixtures:
                    self.logger.info("No NS fixtures found for data collection")
                else:
                    self.logger.info(f"Found {len(ns_fixtures)} NS fixtures for data collection")
                    
                    # Mapping of data types to their fetch methods
                    fetch_functions = {
                        'team_statistics': self.fetch_team_statistics,
                        'player_statistics': self.fetch_player_statistics,
                        'lineups': self.fetch_lineups,
                        'injuries': self.fetch_injuries,
                        'odds': self.fetch_odds,
                    }
                    
                    for data_type in ns_data_types:
                        if data_type not in fetch_functions:
                            continue
                            
                        # Load existing data to avoid duplicates
                        data_path = os.path.join(raw_path, f"{data_type}.json")
                        existing_data_ids = set()
                        
                        if os.path.exists(data_path):
                            try:
                                with open(data_path, 'r') as f:
                                    existing_data = json.load(f)
                                    existing_data_ids = {str(item.get('fixture', {}).get('id')) for item in existing_data}
                                    self.logger.info(f"Found {len(existing_data_ids)} existing {data_type} records")
                            except (json.JSONDecodeError, FileNotFoundError):
                                existing_data_ids = set()
                        
                        # Only process fixtures that don't already have this data type
                        fixtures_to_process = [
                            fid for fid in ns_fixtures 
                            if fid not in existing_data_ids and 
                            progress['data_types'].get(data_type, {}).get(fid) != 'completed'
                        ]
                        
                        if fixtures_to_process:
                            self.logger.info(f"Processing {len(fixtures_to_process)} fixtures for {data_type}")
                            
                            collected_data = []
                            for fixture_id in fixtures_to_process:
                                try:
                                    data = fetch_functions[data_type](int(fixture_id))
                                    if data:
                                        collected_data.append(data)
                                        # Initialize data type in progress if not exists
                                        if data_type not in progress['data_types']:
                                            progress['data_types'][data_type] = {}
                                        progress['data_types'][data_type][fixture_id] = 'completed'
                                        self.logger.info(f"Collected {data_type} for fixture {fixture_id}")
                                except Exception as e:
                                    self.logger.error(f"Failed {data_type} collection for fixture {fixture_id}: {str(e)}")
                                    if data_type not in progress['data_types']:
                                        progress['data_types'][data_type] = {}
                                    progress['data_types'][data_type][fixture_id] = f'failed: {str(e)}'
                            
                            # Save collected data
                            if collected_data:
                                if os.path.exists(data_path):
                                    with open(data_path, 'r') as f:
                                        existing_data = json.load(f)
                                    # Remove any existing entries for the fixtures we just processed
                                    existing_data = [item for item in existing_data 
                                                if str(item.get('fixture', {}).get('id')) not in fixtures_to_process]
                                    combined_data = existing_data + collected_data
                                else:
                                    combined_data = collected_data
                                
                                with open(data_path, 'w') as f:
                                    json.dump(combined_data, f, indent=2)
                                self.logger.info(f"Saved {len(combined_data)} {data_type} records to {data_path}")
                            
                            self.save_progress(progress, progress_file)
                        else:
                            self.logger.info(f"All NS fixtures already have {data_type} data")
        
        return {
            'status': 'success',
            'phase': collection_phase,
            'league': league_name,
            'league_id': league_id,
            'season': season_name,
            'fixture_events_collected': len(all_fixture_ids),
            'existing_fixtures_skipped': len(existing_fixture_ids),
            'ns_fixtures_tracked': len(progress.get('ns_fixtures', [])),
            'storage_path': raw_path,
            'progress_file': progress_file,
            'date_range': f"{start_date} to {end_date}"
        }


    def collect_league_data_merge(self, league_id, season, start_date, end_date, 
                                data_types=None, keep_progress=False,
                                batch_size=50, progress_file="data_collection_progress.json"):
        """
        Specialized method to collect data for specific date ranges and merge with existing data.
        Perfect for filling in missing data gaps in your JSON files.
        
        Args:
            league_id: ID of the league to collect data for
            season: Season year (e.g., 2024)
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            data_types: List of data types to collect
            keep_progress: Whether to keep progress tracking
            batch_size: Batch size for API calls
            progress_file: Progress file name
        """
        try:
            # 1. Get league metadata
            country_name, league_info = self.get_league_info(league_id)
            league_name = league_info['name']
            
            # 2. Set default data types
            if data_types is None:
                data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                            'lineups', 'injuries', 'odds']
            
            # 3. Generate storage paths
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
            raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
            os.makedirs(raw_path, exist_ok=True)
            
            # 4. Load existing progress or create new
            progress = {
                'league_id': league_id,
                'country': country_name,
                'league': league_name,
                'season': season_name,
                'data_types': {dt: {} for dt in data_types},
                'last_updated': datetime.now().isoformat(),
                'fixture_ids': [],
                'date_range': f"{start_date} to {end_date}",
                'collection_type': 'merge'
            }
            
            existing_progress = self.load_progress(progress_file) if keep_progress else None
            if existing_progress and existing_progress.get('league_id') == league_id:
                progress.update(existing_progress)
            
            self.save_progress(progress, progress_file)
            
            # 5. Fetch fixtures for the specified date range
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
            
            # 6. Merge fixtures with existing data
            fixture_path = os.path.join(raw_path, "fixture_events.json")
            merged_fixtures = self._merge_fixture_data(fixture_path, current_fixtures)
            
            # 7. Update progress
            progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + all_fixture_ids))
            for fid in all_fixture_ids:
                if fid not in progress['data_types']['fixture_events']:
                    progress['data_types']['fixture_events'][fid] = 'completed'
            
            self.save_progress(progress, progress_file)
            
            # 8. Process other data types for completed games only
            collection = {dt: [] for dt in data_types if dt != 'fixture_events'}
            
            completed_fixtures = [
                str(f['fixture']['id']) for f in current_fixtures 
                if f['fixture']['status']['short'] != 'NS'
            ]
            
            # Mapping of data types to their fetch methods
            fetch_functions = {
                'team_statistics': self.fetch_team_statistics,
                'player_statistics': self.fetch_player_statistics,
                'lineups': self.fetch_lineups,
                'injuries': self.fetch_injuries,
                'odds': self.fetch_odds,
            }
            
            # Process each data type
            for data_type, fetch_func in fetch_functions.items():
                if data_type not in collection:
                    continue
                    
                fixtures_to_process = [
                    fid for fid in completed_fixtures 
                    if progress['data_types'][data_type].get(fid) != 'completed'
                ]
                
                if not fixtures_to_process:
                    self.logger.info(f"No fixtures need processing for {data_type}")
                    continue
                    
                self.logger.info(f"Processing {len(fixtures_to_process)} fixtures for {data_type}")
                
                for start in range(0, len(fixtures_to_process), batch_size):
                    batch = fixtures_to_process[start:start+batch_size]
                    self.logger.info(f"Processing batch {start//batch_size + 1} for {data_type}")
                    
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
            
            # 9. Save collected data with merge logic
            for data_type, data_list in collection.items():
                if not data_list:
                    continue
                    
                out_path = os.path.join(raw_path, f'{data_type}.json')
                self._merge_data_file(out_path, data_list, 'fixture')
            
            return {
                'status': 'success',
                'league': league_name,
                'season': season_name,
                'fixture_events_collected': len(all_fixture_ids),
                'completed_fixtures_processed': len(completed_fixtures),
                'storage_path': raw_path,
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            self.logger.error(f"Merge collection failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'league_id': league_id,
                'season': season,
                'date_range': f"{start_date} to {end_date}"
            }

    def _merge_fixture_data(self, file_path, new_fixtures):
        """Merge new fixtures with existing fixture data"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_fixtures = json.load(f)
            
            # Create a dictionary of existing fixtures for easy lookup
            existing_dict = {str(f['fixture']['id']): f for f in existing_fixtures}
            
            # Add or update fixtures
            for new_fixture in new_fixtures:
                fixture_id = str(new_fixture['fixture']['id'])
                existing_dict[fixture_id] = new_fixture
            
            # Convert back to list
            merged_fixtures = list(existing_dict.values())
            
            with open(file_path, 'w') as f:
                json.dump(merged_fixtures, f, indent=2)
            
            new_count = len(merged_fixtures) - len(existing_fixtures)
            self.logger.info(f"Merged {new_count} new fixtures. Total: {len(merged_fixtures)}")
            return merged_fixtures
        else:
            # First time: create new file
            with open(file_path, 'w') as f:
                json.dump(new_fixtures, f, indent=2)
            self.logger.info(f"Saved {len(new_fixtures)} fixture events")
            return new_fixtures

    def _merge_data_file(self, file_path, new_data, id_key='fixture'):
        """Merge new data with existing data file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
            
            # Create a dictionary of existing data for easy lookup
            existing_dict = {}
            for item in existing_data:
                item_id = str(item.get(id_key, {}).get('id'))
                if item_id:
                    existing_dict[item_id] = item
            
            # Add or update data
            for new_item in new_data:
                item_id = str(new_item.get(id_key, {}).get('id'))
                if item_id:
                    existing_dict[item_id] = new_item
            
            # Convert back to list
            merged_data = list(existing_dict.values())
            
            with open(file_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
            
            new_count = len(merged_data) - len(existing_data)
            self.logger.info(f"Merged {new_count} new records into {os.path.basename(file_path)}. Total: {len(merged_data)}")
        else:
            # First time: create new file
            with open(file_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            self.logger.info(f"Saved {len(new_data)} records to {os.path.basename(file_path)}")

    def find_missing_date_ranges(self, league_id, season):
        """
        Analyze existing data to find missing date ranges that need to be collected.
        This helps identify gaps in your data collection.
        """
        country_name, league_info = self.get_league_info(league_id)
        league_name = league_info['name']
        season_name, season_start, season_end = self.calculate_season_dates(country_name, league_info, season)
        
        raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
        fixture_path = os.path.join(raw_path, "fixture_events.json")
        
        if not os.path.exists(fixture_path):
            return [{'start': season_start, 'end': season_end, 'reason': 'No data exists'}]
        
        with open(fixture_path, 'r') as f:
            fixtures = json.load(f)
        
        # Extract all fixture dates
        fixture_dates = []
        for fixture in fixtures:
            fixture_date = fixture['fixture']['date'][:10]  # Extract YYYY-MM-DD
            fixture_dates.append(fixture_date)
        
        # Sort dates and find gaps
        fixture_dates.sort()
        missing_ranges = []
        
        # Convert to datetime objects for easier comparison
        date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in fixture_dates]
        season_start_dt = datetime.strptime(season_start, '%Y-%m-%d')
        season_end_dt = datetime.strptime(season_end, '%Y-%m-%d')
        
        # Check for gap at the beginning
        if date_objects and date_objects[0] > season_start_dt:
            missing_ranges.append({
                'start': season_start,
                'end': (date_objects[0] - timedelta(days=1)).strftime('%Y-%m-%d'),
                'reason': 'Missing early season data'
            })
        
        # Check for gaps between fixtures
        for i in range(1, len(date_objects)):
            gap = (date_objects[i] - date_objects[i-1]).days
            if gap > 7:  # More than a week gap suggests missing data
                missing_ranges.append({
                    'start': (date_objects[i-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'end': (date_objects[i] - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'reason': f'Gap of {gap} days between fixtures'
                })
        
        # Check for gap at the end
        if date_objects and date_objects[-1] < season_end_dt:
            missing_ranges.append({
                'start': (date_objects[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
                'end': season_end,
                'reason': 'Missing late season data'
            })
        
        return missing_ranges




    def collect_league_data_smart_merge(self, league_id, season, start_date, end_date, 
                                    data_types=None, keep_progress=False,
                                    batch_size=50, progress_file="data_collection_progress.json", collection_phase=1):
        """
        Smart collection method that only processes fixtures not already in the files.
        Efficiently merges new data without duplicating efforts.
        
        Args:
            league_id: ID of the league to collect data for
            season: Season year (e.g., 2024)
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            data_types: List of data types to collect
            keep_progress: Whether to keep progress tracking
            batch_size: Batch size for API calls
            progress_file: Progress file name
        """
        try:
            # 1. Get league metadata
            country_name, league_info = self.get_league_info(league_id)
            league_name = league_info['name']
            
            # 2. Set default data types
            if data_types is None:
                data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                            'lineups', 'injuries', 'odds']
            
            # 3. Generate storage paths
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
            raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
            os.makedirs(raw_path, exist_ok=True)
            
            # 4. Load existing fixture IDs to avoid duplicates
            fixture_path = os.path.join(raw_path, "fixture_events.json")
            existing_fixture_ids = set()
            
            if os.path.exists(fixture_path):
                with open(fixture_path, 'r') as f:
                    existing_fixtures = json.load(f)
                existing_fixture_ids = {str(f['fixture']['id']) for f in existing_fixtures}
                self.logger.info(f"Found {len(existing_fixture_ids)} existing fixtures")
            
            # 5. Fetch fixtures for the specified date range
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
            
            # 6. Identify new fixtures that aren't already in our files
            new_fixtures = []
            new_fixture_ids = []
            
            for fixture in current_fixtures:
                fixture_id = str(fixture['fixture']['id'])
                if fixture_id not in existing_fixture_ids:
                    new_fixtures.append(fixture)
                    new_fixture_ids.append(fixture_id)
            
            if not new_fixtures:
                self.logger.info("No new fixtures found in the specified date range")
                return {
                    'status': 'success', 
                    'message': 'All fixtures already exist in files',
                    'existing_fixtures': len(existing_fixture_ids),
                    'new_fixtures': 0
                }
            
            self.logger.info(f"Found {len(new_fixtures)} new fixtures to process")
            
            # 7. Merge new fixtures with existing data
            merged_fixtures = existing_fixtures + new_fixtures if os.path.exists(fixture_path) else new_fixtures
            with open(fixture_path, 'w') as f:
                json.dump(merged_fixtures, f, indent=2)
            
            # 8. Load progress and update it
            progress = {
                'league_id': league_id,
                'country': country_name,
                'league': league_name,
                'season': season_name,
                'data_types': {dt: {} for dt in data_types},
                'last_updated': datetime.now().isoformat(),
                'fixture_ids': list(existing_fixture_ids) + new_fixture_ids,
                'date_range': f"{start_date} to {end_date}",
                'collection_type': 'smart_merge'
            }
            
            existing_progress = self.load_progress(progress_file) if keep_progress else None
            if existing_progress and existing_progress.get('league_id') == league_id:
                progress.update(existing_progress)
                # Ensure we don't lose existing fixture IDs
                progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + new_fixture_ids))
            
            # 9. Process other data types only for new completed games
            collection = {dt: [] for dt in data_types if dt != 'fixture_events'}
            
            completed_new_fixtures = [
                str(f['fixture']['id']) for f in new_fixtures 
                if f['fixture']['status']['short'] != 'NS'
            ]
            
            if not completed_new_fixtures:
                self.logger.info("No completed new fixtures to process for additional data")
                self.save_progress(progress, progress_file)
                return {
                    'status': 'success',
                    'league': league_name,
                    'season': season_name,
                    'new_fixtures_added': len(new_fixtures),
                    'completed_new_fixtures': 0,
                    'total_fixtures': len(merged_fixtures),
                    'storage_path': raw_path
                }
            
            # Mapping of data types to their fetch methods
            fetch_functions = {
                'team_statistics': self.fetch_team_statistics,
                'player_statistics': self.fetch_player_statistics,
                'lineups': self.fetch_lineups,
                'injuries': self.fetch_injuries,
                'odds': self.fetch_odds,
            }
            
            # Process each data type for new fixtures
            for data_type, fetch_func in fetch_functions.items():
                if data_type not in collection:
                    continue
                    
                # Check which fixtures need processing for this data type
                fixtures_to_process = [
                    fid for fid in completed_new_fixtures 
                    if progress['data_types'][data_type].get(fid) != 'completed'
                ]
                
                if not fixtures_to_process:
                    self.logger.info(f"No new fixtures need processing for {data_type}")
                    continue
                    
                self.logger.info(f"Processing {len(fixtures_to_process)} new fixtures for {data_type}")
                
                data_list = []
                for start in range(0, len(fixtures_to_process), batch_size):
                    batch = fixtures_to_process[start:start+batch_size]
                    self.logger.info(f"Processing batch {start//batch_size + 1} for {data_type}")
                    
                    for fixture_id in batch:
                        try:
                            data = fetch_func(int(fixture_id))
                            if data:
                                data_list.append(data)
                                progress['data_types'][data_type][fixture_id] = 'completed'
                        except Exception as e:
                            self.logger.error(f"Failed {data_type} for fixture {fixture_id}: {str(e)}")
                            progress['data_types'][data_type][fixture_id] = f'failed: {str(e)}'
                    
                    self.save_progress(progress, progress_file)
                
                # Merge the new data with existing data file
                if data_list:
                    out_path = os.path.join(raw_path, f'{data_type}.json')
                    self._smart_merge_data_file(out_path, data_list, 'fixture')
            
            self.save_progress(progress, progress_file)
            
            return {
                'status': 'success',
                'league': league_name,
                'season': season_name,
                'new_fixtures_added': len(new_fixtures),
                'completed_new_fixtures': len(completed_new_fixtures),
                'total_fixtures': len(merged_fixtures),
                'storage_path': raw_path,
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            self.logger.error(f"Smart merge collection failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'league_id': league_id,
                'season': season,
                'date_range': f"{start_date} to {end_date}"
            }

    def _smart_merge_data_file(self, file_path, new_data, id_key='fixture'):
        """Smart merge that only adds new data without loading entire file if possible"""
        if not new_data:
            return
        
        if os.path.exists(file_path):
            # Load existing data
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
            
            # Create a set of existing IDs for quick lookup
            existing_ids = set()
            for item in existing_data:
                item_id = str(item.get(id_key, {}).get('id'))
                if item_id:
                    existing_ids.add(item_id)
            
            # Filter out data that already exists
            truly_new_data = []
            for item in new_data:
                item_id = str(item.get(id_key, {}).get('id'))
                if item_id and item_id not in existing_ids:
                    truly_new_data.append(item)
            
            if not truly_new_data:
                self.logger.info(f"No new data to add to {os.path.basename(file_path)}")
                return
            
            # Merge and save
            merged_data = existing_data + truly_new_data
            with open(file_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
            
            self.logger.info(f"Added {len(truly_new_data)} new records to {os.path.basename(file_path)}. Total: {len(merged_data)}")
        else:
            # First time: create new file
            with open(file_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            self.logger.info(f"Created new {os.path.basename(file_path)} with {len(new_data)} records")

    def get_existing_fixture_ids(self, league_id, season):
        """Get all fixture IDs that already exist in the data files"""
        country_name, league_info = self.get_league_info(league_id)
        league_name = league_info['name']
        season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
        
        raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
        fixture_path = os.path.join(raw_path, "fixture_events.json")
        
        if not os.path.exists(fixture_path):
            return set()
        
        with open(fixture_path, 'r') as f:
            fixtures = json.load(f)
        
        return {str(f['fixture']['id']) for f in fixtures}

    def collect_missing_fixtures_only(self, league_id, season, start_date, end_date, data_types=None):
        """
        High-level method to collect only fixtures that are missing from the specified date range
        """
        # Get existing fixture IDs
        existing_ids = self.get_existing_fixture_ids(league_id, season)
        
        # Fetch fixtures for the date range
        fixtures = self.fetch_fixture_events(
            league_id=league_id,
            season=season,
            start_date=start_date,
            end_date=end_date
        )
        
        if not fixtures or not fixtures.get('response'):
            return {'status': 'error', 'message': 'No fixtures found'}
        
        current_fixtures = fixtures['response']
        
        # Find missing fixtures
        missing_fixtures = []
        for fixture in current_fixtures:
            fixture_id = str(fixture['fixture']['id'])
            if fixture_id not in existing_ids:
                missing_fixtures.append(fixture)
        
        if not missing_fixtures:
            return {'status': 'success', 'message': 'No missing fixtures found'}
        
        # Process missing fixtures
        result = self.collect_league_data_smart_merge(
            league_id=league_id,
            season=season,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types
        )
        
        result['missing_fixtures_found'] = len(missing_fixtures)
        result['missing_fixture_ids'] = [str(f['fixture']['id']) for f in missing_fixtures]
        
        return result





    def collect_league_data_filter_smart_merge(self, league_id=None, season=None, data_types=None, keep_progress=False,
                                    batch_size=50, progress_file="data_collection_progress.json", 
                                    start_date=None, end_date=None, collection_phase=1,
                                    filter_ids=None, filter_tiers=None, filter_cups=None, filter_categories=None):
        """
        Smart collection method that automatically calculates dates and only merges new data.
        Perfect for filling missing data gaps efficiently.
        
        Args:
            collection_phase: 1 = completed games, 2 = NS games, 3 = update NS games
            filter_ids: List of specific league IDs to filter by
            filter_tiers: List of tiers to filter by (e.g., ['top_tier', 'second_tier'])
            filter_cups: List of cup types to filter by (e.g., ['domestic_cup', 'league_cup'])
            filter_categories: List of league categories to filter by (e.g., ['european_elite', 'top_tier'])
        """
        try:
            # Handle filtering if specified
            if any([filter_ids, filter_tiers, filter_cups, filter_categories]):
                return self._collect_multiple_leagues_smart_merge(
                    season, data_types, keep_progress, batch_size, progress_file,
                    start_date, end_date, collection_phase, filter_ids, filter_tiers,
                    filter_cups, filter_categories
                )
            
            # Single league collection with smart merge
            country_name, league_info = self.get_league_info(league_id)
            league_name = league_info['name']
            
            current_year = datetime.now().year
            is_current_season = str(current_year) in str(season)

            # Calculate automatic dates if not provided
            if not start_date or not end_date:
                season_name, auto_start, auto_end = self.calculate_season_dates(
                    country_name, league_info, season
                )
                start_date = start_date or auto_start
                end_date = end_date or auto_end
            
            # Set default data types
            if data_types is None:
                data_types = ['fixture_events', 'team_statistics', 'player_statistics', 
                            'lineups', 'injuries', 'team_standings', 'odds']

            # Generate storage paths
            season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
            raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
            os.makedirs(raw_path, exist_ok=True)

            # Load existing fixture IDs to avoid duplicates
            fixture_path = os.path.join(raw_path, "fixture_events.json")
            existing_fixture_ids = set()
            
            if os.path.exists(fixture_path):
                with open(fixture_path, 'r') as f:
                    existing_fixtures = json.load(f)
                existing_fixture_ids = {str(f['fixture']['id']) for f in existing_fixtures}
                self.logger.info(f"Found {len(existing_fixture_ids)} existing fixtures")

            # Fetch fixtures for the date range
            self.logger.info(f"Smart merge: Fetching fixtures for {league_name} {season_name}...")
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
            
            # Identify new fixtures that aren't already in our files
            new_fixtures = []
            new_fixture_ids = []
            
            for fixture in current_fixtures:
                fixture_id = str(fixture['fixture']['id'])
                if fixture_id not in existing_fixture_ids:
                    new_fixtures.append(fixture)
                    new_fixture_ids.append(fixture_id)
            
            if not new_fixtures:
                self.logger.info("No new fixtures found - all data already exists")
                return {
                    'status': 'success', 
                    'message': 'All fixtures already exist in files',
                    'existing_fixtures': len(existing_fixture_ids),
                    'new_fixtures': 0,
                    'league': league_name,
                    'season': season_name
                }
            
            self.logger.info(f"Found {len(new_fixtures)} new fixtures to process")
            
            # Merge new fixtures with existing data
            merged_fixtures = existing_fixtures + new_fixtures if os.path.exists(fixture_path) else new_fixtures
            with open(fixture_path, 'w') as f:
                json.dump(merged_fixtures, f, indent=2)
            
            # Load progress and update it
            progress = {
                'league_id': league_id,
                'country': country_name,
                'league': league_name,
                'season': season_name,
                'data_types': {dt: {} for dt in data_types},
                'last_updated': datetime.now().isoformat(),
                'fixture_ids': list(existing_fixture_ids) + new_fixture_ids,
                'date_range': f"{start_date} to {end_date}",
                'collection_type': 'smart_merge'
            }
            
            existing_progress = self.load_progress(progress_file) if keep_progress else None
            if existing_progress and existing_progress.get('league_id') == league_id:
                progress.update(existing_progress)
                progress['fixture_ids'] = list(set(progress.get('fixture_ids', []) + new_fixture_ids))
            
            self.save_progress(progress, progress_file)
            
            # Process other data types only for new completed games
            collection = {dt: [] for dt in data_types if dt != 'fixture_events'}
            
            completed_new_fixtures = [
                str(f['fixture']['id']) for f in new_fixtures 
                if f['fixture']['status']['short'] != 'NS'
            ]
            
            if not completed_new_fixtures:
                self.logger.info("No completed new fixtures to process for additional data")
                return {
                    'status': 'success',
                    'league': league_name,
                    'season': season_name,
                    'new_fixtures_added': len(new_fixtures),
                    'completed_new_fixtures': 0,
                    'total_fixtures': len(merged_fixtures),
                    'storage_path': raw_path
                }
            
            # Process each data type for new fixtures
            fetch_functions = {
                'team_statistics': self.fetch_team_statistics,
                'player_statistics': self.fetch_player_statistics,
                'lineups': self.fetch_lineups,
                'injuries': self.fetch_injuries,
                'odds': self.fetch_odds,
                'team_standings': lambda league_id, season: self.fetch_team_standings(league_id, season)
            }
            
            # Store all collected data by type
            all_collected_data = {dt: [] for dt in collection.keys()}
            
            for data_type in collection.keys():
                if data_type not in fetch_functions:
                    continue
                    
                # Check which fixtures need processing for this data type
                fixtures_to_process = [
                    fid for fid in completed_new_fixtures 
                    if progress['data_types'][data_type].get(fid) != 'completed'
                ]
                
                if not fixtures_to_process:
                    self.logger.info(f"No new fixtures need processing for {data_type}")
                    continue
                
                self.logger.info(f"Processing {len(fixtures_to_process)} new fixtures for {data_type}")
                
                data_list = []
                if data_type == 'team_standings':
                    # Team standings is league-level, not fixture-level
                    try:
                        data = fetch_functions[data_type](league_id, season)
                        if data:
                            data_list.append(data)
                            progress['data_types'][data_type] = {'status': 'completed'}
                            all_collected_data[data_type] = data_list
                    except Exception as e:
                        self.logger.error(f"Failed {data_type}: {str(e)}")
                        progress['data_types'][data_type] = {'status': f'failed: {str(e)}'}
                else:
                    # Fixture-level data
                    for start in range(0, len(fixtures_to_process), batch_size):
                        batch = fixtures_to_process[start:start+batch_size]
                        
                        for fixture_id in batch:
                            try:
                                data = fetch_functions[data_type](int(fixture_id))
                                if data:
                                    data_list.append(data)
                                    progress['data_types'][data_type][fixture_id] = 'completed'
                            except Exception as e:
                                self.logger.error(f"Failed {data_type} for fixture {fixture_id}: {str(e)}")
                                progress['data_types'][data_type][fixture_id] = f'failed: {str(e)}'
                        
                        self.save_progress(progress, progress_file)
                    
                    all_collected_data[data_type] = data_list
            
            # Now merge all collected data at once
            for data_type, data_list in all_collected_data.items():
                if not data_list:
                    continue
                    
                out_path = os.path.join(raw_path, f'{data_type}.json')
                self._smart_merge_data_file(out_path, data_list, 'fixture' if data_type != 'team_standings' else 'league')
            
            self.save_progress(progress, progress_file)
            
            return {
                'status': 'success',
                'league': league_name,
                'season': season_name,
                'new_fixtures_added': len(new_fixtures),
                'completed_new_fixtures': len(completed_new_fixtures),
                'total_fixtures': len(merged_fixtures),
                'storage_path': raw_path,
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            self.logger.error(f"Smart merge collection failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'league_id': league_id,
                'season': season,
                'date_range': f"{start_date} to {end_date}" if 'start_date' in locals() and 'end_date' in locals() else 'N/A'
            }

    def _collect_multiple_leagues_smart_merge(self, season, data_types, keep_progress, batch_size,
                                            progress_file, start_date, end_date, collection_phase,
                                            filter_ids, filter_tiers, filter_cups, filter_categories):
        """
        Handle smart merge collection for multiple leagues based on filtering criteria
        """
        filtered_leagues = self._filter_leagues(
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
            league_id = league_info.get('id')
            league_name = league_info['name']
            
            result = self.collect_league_data_smart_merge(
                league_id=league_id,
                season=season,
                data_types=data_types,
                keep_progress=keep_progress,
                batch_size=batch_size,
                progress_file=progress_file,
                start_date=start_date,
                end_date=end_date,
                collection_phase=collection_phase
            )
            results.append(result)
        
        return {
            'status': 'success',
            'phase': collection_phase,
            'results': results,
            'total_leagues_processed': len(results)
        }

    def _smart_merge_data_file(self, file_path, new_data, id_key='fixture'):
        """Smart merge that properly handles new data addition"""
        if not new_data:
            self.logger.info(f"No data to process for {os.path.basename(file_path)}")
            return
        
        # For team standings, replace entirely rather than merge
        if id_key == 'league':
            with open(file_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            self.logger.info(f"Updated {os.path.basename(file_path)} with new standings")
            return
        
        # For fixture data, check if file exists and merge appropriately
        if os.path.exists(file_path):
            try:
                # Load existing data
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Create a dictionary of existing data for easy lookup
                existing_dict = {}
                for item in existing_data:
                    item_id = str(item.get(id_key, {}).get('id', ''))
                    if item_id:
                        existing_dict[item_id] = item
                
                # Add or update data
                new_items_added = 0
                for new_item in new_data:
                    item_id = str(new_item.get(id_key, {}).get('id', ''))
                    if item_id:
                        if item_id not in existing_dict:
                            # New item, add it
                            existing_dict[item_id] = new_item
                            new_items_added += 1
                        else:
                            # Item exists, but we might want to update it
                            # For now, we'll keep the existing one to avoid duplicates
                            pass
                
                if new_items_added > 0:
                    # Convert back to list and save
                    merged_data = list(existing_dict.values())
                    with open(file_path, 'w') as f:
                        json.dump(merged_data, f, indent=2)
                    self.logger.info(f"Added {new_items_added} new records to {os.path.basename(file_path)}. Total: {len(merged_data)}")
                else:
                    self.logger.info(f"No new data to add to {os.path.basename(file_path)}")
                    
            except Exception as e:
                self.logger.error(f"Error merging data for {os.path.basename(file_path)}: {str(e)}")
                # Fallback: append new data
                try:
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                    merged_data = existing_data + new_data
                    with open(file_path, 'w') as f:
                        json.dump(merged_data, f, indent=2)
                    self.logger.info(f"Fallback merge: Added {len(new_data)} records to {os.path.basename(file_path)}")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback merge also failed: {str(fallback_error)}")
        else:
            # First time: create new file
            with open(file_path, 'w') as f:
                json.dump(new_data, f, indent=2)
            self.logger.info(f"Created new {os.path.basename(file_path)} with {len(new_data)} records")

    def get_existing_fixture_ids(self, league_id, season):
        """Get all fixture IDs that already exist in the data files"""
        country_name, league_info = self.get_league_info(league_id)
        league_name = league_info['name']
        season_name, _, _ = self.calculate_season_dates(country_name, league_info, season)
        
        raw_path = os.path.join(self.base_path, country_name, league_name, season_name)
        fixture_path = os.path.join(raw_path, "fixture_events.json")
        
        if not os.path.exists(fixture_path):
            return set()
        
        with open(fixture_path, 'r') as f:
            fixtures = json.load(f)
        
        return {str(f['fixture']['id']) for f in fixtures}



