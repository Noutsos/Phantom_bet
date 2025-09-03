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



# Configure logging
logging.basicConfig(level=logging.INFO) 


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



def make_api_request(endpoint):
    conn = http.client.HTTPSConnection(API_HOST)
    headers = {
        'x-rapidapi-host': API_HOST,
        'x-rapidapi-key': API_KEY
    }

    rate_limiter.check_rate_limit()

    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()

    try:
        response_data = json.loads(data)
    except Exception as e:
        raise Exception(f"Failed to parse JSON: {e} | Response: {data[:200]}")

    if res.status == 200:
        rate_limiter.update_rate_limits(dict(res.getheaders()))
        return response_data
    else:
        raise Exception(f"API request failed with status {res.status}: {response_data}")


def fetch_fixture_events(league, season, start_date, end_date):
    endpoint = f"/fixtures?league={league}&season={season}&from={start_date}&to={end_date}"
    return make_api_request(endpoint)

def fetch_team_statistics(fixture_id):
    endpoint = f"/fixtures/statistics?fixture={fixture_id}"
    return make_api_request(endpoint)

def fetch_player_statistics(fixture_id):
    endpoint = f"/fixtures/players?fixture={fixture_id}"
    return make_api_request(endpoint)

def fetch_odds(fixture_id, bookmaker_id=8):
    endpoint = f"/odds?bookmaker={bookmaker_id}&fixture={fixture_id}"
    return make_api_request(endpoint)


def fetch_team_standings(league, season):
    endpoint = f"/standings?league={league}&season={season}"
    return make_api_request(endpoint)

def fetch_top_scorers(league, season):
    endpoint = f"/players/topscorers?season={season}&league={league}"
    return make_api_request(endpoint)

def fetch_top_assists(league, season):
    endpoint = f"/players/topassists?season={season}&league={league}"
    return make_api_request(endpoint)

def fetch_top_yellowcards(league, season):
    endpoint = f"/players/topyellowcards?season={season}&league={league}"
    return make_api_request(endpoint)

def fetch_top_redcards(league, season):
    endpoint = f"/players/topredcards?season={season}&league={league}"
    return make_api_request(endpoint)

def fetch_predictions(fixture_id):
    endpoint = f"/predictions?fixture={fixture_id}"
    return make_api_request(endpoint)

def fetch_lineups(fixture_id):
    endpoint = f"/fixtures/lineups?fixture={fixture_id}"
    return make_api_request(endpoint)

def fetch_injuries(fixture_id):
    endpoint = f"/injuries?fixture={fixture_id}"
    return make_api_request(endpoint)


def fetch_all_fixture_events(league, season, start_date, end_date):
    """Fetch fixture events for a range of dates."""
    response = fetch_fixture_events(league, season, start_date, end_date)
    return response['response']

def fetch_all_team_statistics(fixture_ids):
    """Fetch team statistics for multiple fixtures in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers to avoid overwhelming the API
        team_stats = list(executor.map(fetch_team_statistics, fixture_ids))
    return team_stats

def fetch_all_player_statistics(fixture_ids):
    """Fetch player statistics for multiple fixtures in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers to avoid overwhelming the API
        player_stats = list(executor.map(fetch_player_statistics, fixture_ids))
    return player_stats

def fetch_all_odds(fixture_ids):
    """Fetch odds for multiple fixtures in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers to avoid overwhelming the API
        odds = list(executor.map(fetch_odds, fixture_ids))
    return odds

def fetch_all_team_standings(league, season):
    """Fetch team standings for a league and season."""
    response = fetch_team_standings(league, season)
    return response['response']

def fetch_all_lineups(fixture_ids):
    """Fetch lineups for multiple fixtures in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers to avoid overwhelming the API
        lineups = list(executor.map(fetch_lineups, fixture_ids))
    return lineups

def fetch_all_injuries(fixture_ids):
    """Fetch injuries for multiple fixtures in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers to avoid overwhelming the API
        injuries = list(executor.map(fetch_injuries, fixture_ids))
    return injuries



def sanitize_for_json(obj):
    """Recursively convert np.int64, np.int32, etc. to int for JSON serialization."""
    try:
        import numpy as np
    except ImportError:
        np = None

    if np:
        if isinstance(obj, (np.integer,)):
            return int(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    return obj

def save_progress(progress_data, filename='progress.json', folder='src'):
    """
    Save the progress to a JSON file in the specified folder.
    Handles nested numpy types for JSON serialization.
    """
    save_path = os.path.join(folder)
    os.makedirs(save_path, exist_ok=True)
    progress_file = os.path.join(save_path, filename)
    sanitized_data = sanitize_for_json(progress_data)
    try:
        with open(progress_file, "w") as file:
            json.dump(sanitized_data, file, indent=4)
        # print(f"Progress saved to {progress_file}.")
    except Exception as e:
        print(f"Failed to save progress: {e}")

def load_progress(filename='progress.json', folder='src'):
    """
    Load progress from a JSON file in the specified folder.
    Returns an empty dict if not found.
    """
    progress_file = os.path.join(folder, filename)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as file:
                progress_data = json.load(file)
            print(f"Progress loaded from {progress_file}.")
            return progress_data
        except Exception as e:
            print(f"Failed to load progress: {e}")
            return {}
    print(f"No progress file found in {folder}. Starting fresh.")
    return {}

def clear_progress(filename='progress.json', folder='src'):
    """
    Delete the progress file if it exists.
    """
    progress_file = os.path.join(folder, filename)
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            print(f"Progress file {progress_file} has been cleared.")
        except Exception as e:
            print(f"Failed to clear progress: {e}")
    else:
        print(f"No progress file found in {folder} to clear.")

# League dictionary organized by country
LEAGUES = {
    # Europe (international competitions)
    'Europe': {
        '2': {'name': 'Champions League', 'season_months': 9},
        '3': {'name': 'Europa League', 'season_months': 9},
        '848': {'name': 'Conference League', 'season_months': 9}
    },
    # Spain
    'Spain': {
        '140': {'name': 'La Liga', 'season_months': 10},
        '143': {'name': 'Copa del Rey', 'season_months': 8}
    },
    # Italy
    'Italy': {
        '135': {'name': 'Serie A', 'season_months': 10},
        '137': {'name': 'Coppa Italia', 'season_months': 8}
    },
    # Germany
    'Germany': {
        '78': {'name': 'Bundesliga', 'season_months': 10},
        '81': {'name': 'DFB Pokal', 'season_months': 8}
    },
    # France
    'France': {
        '61': {'name': 'Ligue 1', 'season_months': 10},
        '66': {'name': 'Coupe de France', 'season_months': 8}
    },
    # England
    'England': {
        '39': {'name': 'Premier League', 'season_months': 10},
        '45': {'name': 'FA Cup', 'season_months': 8},
        '46': {'name': 'EFL Trophy', 'season_months': 8}
    },
    'Switzerland': {
        '207': {'name': 'Super League', 'season_months': 10},
        '209': {'name': 'Cup', 'season_months': 8}
    },
    'Netherlands': {
        '88': {'name': 'Eredivisie', 'season_months': 10},
        '90': {'name': 'KNVB Beker', 'season_months': 8}
    },
    'Turkey': {
        '203': {'name': 'Super Lig', 'season_months': 10},
        '206': {'name': 'Cup', 'season_months': 8}
    },
    'Belgium': {
        '144': {'name': 'Jupiler Pro League', 'season_months': 10},
        '147': {'name': 'Cup', 'season_months': 8}
    },
    'Scotland': {
        '179': {'name': 'Premiership', 'season_months': 10},
        '181': {'name': 'FA Cup', 'season_months': 8},
        '185': {'name': 'Cup', 'season_months': 8}
    },
    'Portugal': {
        '94': {'name': 'Primeira Liga', 'season_months': 10},
        '97': {'name': 'Taca de Portugal', 'season_months': 8}
    },
    'Austria': {
        '218': {'name': 'Bundesliga', 'season_months': 10},
        '220': {'name': 'Cup', 'season_months': 8}
    },
    'Greece': {
        '197': {'name': 'Super League', 'season_months': 10},
        '199': {'name': 'Cup', 'season_months': 8}
    },    
    
    # Nordic countries (shorter seasons)
    'Sweden': {
        '113': {'name': 'Allsvenskan', 'season_months': 8, 'start_month': 3},
        '115': {'name': 'Svenska Cupen', 'season_months': 8, 'start_month': 2}
    },
    'Norway': {
        '103': {'name': 'Eliteserien', 'season_months': 8, 'start_month': 3},
        '105': {'name': 'NM Cupen', 'season_months': 8, 'start_month': 2}
    },
    'Denmark': {
        '119': {'name': 'Superliga', 'season_months': 8, 'start_month': 3},
        '121': {'name': 'DBU Pokalen', 'season_months': 8, 'start_month': 2}
    },
    'Finland': {
        '244': {'name': 'Veikkausliiga', 'season_months': 8, 'start_month': 4},
        '246': {'name': 'Suomen Cup', 'season_months': 8, 'start_month': 2},
        '700': {'name': 'Kakkosen Cup', 'season_months': 8, 'start_month': 2}
    },
    'Estonia': {
        '329': {'name': 'Meistriliiga', 'season_months': 8, 'start_month': 4},
        '657': {'name': 'Cup', 'season_months': 8, 'start_month': 2}
    },
    'Latvia': {
        '365': {'name': 'Virsliga', 'season_months': 8, 'start_month': 4},
        '658': {'name': 'Cup', 'season_months': 8, 'start_month': 2}
    }

}


def get_league_info(league_id):
    """Find which country a league belongs to and its metadata"""
    league_id = str(league_id)  # Ensure string comparison
    for country, leagues in LEAGUES.items():
        if league_id in leagues:
            return country, leagues[league_id]
    raise ValueError(f"League ID {league_id} not found in any country")

def calculate_season_dates(country_name, league_info, season_year):
    """Calculate automatic season dates based on league characteristics"""
    start_month = league_info.get('start_month', 8)  # Default to August
    
    if country_name in ['Sweden', 'Norway', 'Denmark', 'Finland'] and start_month < 6:
        start_date = datetime(season_year, start_month, 1).strftime('%Y-%m-%d')
        season_name = f"{season_year}"
    else:
        start_date = datetime(season_year, start_month, 1).strftime('%Y-%m-%d')
        season_name = f"{season_year}-{season_year+1}"
    
    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + 
               relativedelta(months=+league_info['season_months'])).strftime('%Y-%m-%d')
    
    return season_name, start_date, end_date


def check_existing_data(raw_path, data_types):
    """Check if data already exists in the target folder"""
    existing_data = {}
    for dtype in data_types:
        file_path = os.path.join(raw_path, f"{dtype}.json")
        if os.path.exists(file_path):
            with open(file_path) as f:
                existing_data[dtype] = len(json.load(f))
        else:
            existing_data[dtype] = 0
    return existing_data



def collect_league_data(
    league_id,
    season_year,
    data_types=None,
    keep_progress=False,
    batch_size=50,
    base_path="data/raw"
):
    """
    Unified function to collect football data with automatic season calculation
    Args:
        league_id: ID of the league (from LEAGUES dictionary)
        season_year: Starting year of the season
        data_types: List of data types to collect (default: all)
        keep_progress: Whether to resume interrupted collection
        batch_size: Number of fixtures to process at once
        base_path: Root directory for data storage
    Returns:
        Dictionary with status and collection results
    """
    try:
        # 1. Get league metadata
        country_name, league_info = get_league_info(league_id)
        league_name = league_info['name']
        
        # 2. Calculate season parameters
        season_name, start_date, end_date = calculate_season_dates(
            country_name, league_info, season_year
        )
        


        
        # 4. Generate storage path
        raw_path = os.path.join(
            base_path,
            country_name,
            league_name,
            season_name
        )
        os.makedirs(raw_path, exist_ok=True)
        
        # --- Progress Handling ---
        if not keep_progress:
            clear_progress()
            progress = {
                'league_id': league_id,
                'country': country_name,
                'league': league_name,
                'season': season_name,
                'start_date': start_date,
                'end_date': end_date,
                'data_types': {dt: {} for dt in data_types},
                'collected_fixtures': []
            }
        else:
            progress = load_progress()
            if not progress:
                print("No previous progress found. Starting fresh.")
                progress = {
                    'league_id': league_id,
                    'country': country_name,
                    'league': league_name,
                    'season': season_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'data_types': {dt: {} for dt in data_types},
                    'collected_fixtures': []
                }
            else:
                print(f"Resuming progress for {progress.get('league')} season {progress.get('season')}")

        def save_json(data, filename):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        # --- Fetch Fixtures ---
        print(f"Fetching fixtures for {league_name} {season_name}...")
        fixtures = fetch_fixture_events(league_id, season_year, start_date, end_date)
        
        if not fixtures or not fixtures.get('response'):
            print("No fixture events returned by the API.")
            return {
                'status': 'error',
                'message': 'No fixtures found',
                'parameters': {
                    'league_id': league_id,
                    'season': season_name,
                    'dates': f"{start_date} to {end_date}"
                }
            }

        fixture_ids = [f['fixture']['id'] for f in fixtures['response']]
        print(f"Found {len(fixture_ids)} fixtures")
        
        # Save fixtures
        fixture_json_path = os.path.join(raw_path, 'fixtures.json')
        save_json(fixtures['response'], fixture_json_path)
        print(f"Saved fixtures to {fixture_json_path}")

        # --- Collect all results for each data_type ---
        collection = {dt: [] for dt in data_types if dt != 'team_standings'}
        
        # --- Batch processing ---
        total = len(fixture_ids)
        for start in range(0, total, batch_size):
            batch = fixture_ids[start:start+batch_size]
            print(f"Processing fixtures {start+1} to {min(start+batch_size, total)} of {total}")
            
            for fixture_id in batch:
                if fixture_id in progress['collected_fixtures']:
                    continue
                
                # Team Statistics
                if 'team_statistics' in collection:
                    try:
                        data = fetch_team_statistics(fixture_id)
                        collection['team_statistics'].append(data)
                        progress['data_types']['team_statistics'][str(fixture_id)] = 'completed'
                    except Exception as e:
                        print(f"Failed team stats for fixture {fixture_id}: {e}")
                
                # Player Statistics
                if 'player_statistics' in collection:
                    try:
                        data = fetch_player_statistics(fixture_id)
                        collection['player_statistics'].append(data)
                        progress['data_types']['player_statistics'][str(fixture_id)] = 'completed'
                    except Exception as e:
                        print(f"Failed player stats for fixture {fixture_id}: {e}")
                
                # Injuries
                if 'injuries' in collection:
                    try:
                        data = fetch_injuries(fixture_id)
                        collection['injuries'].append(data)
                        progress['data_types']['injuries'][str(fixture_id)] = 'completed'
                    except Exception as e:
                        print(f"Failed injuries for fixture {fixture_id}: {e}")
                
                # Lineups
                if 'lineups' in collection:
                    try:
                        data = fetch_lineups(fixture_id)
                        collection['lineups'].append(data)
                        progress['data_types']['lineups'][str(fixture_id)] = 'completed'
                    except Exception as e:
                        print(f"Failed lineups for fixture {fixture_id}: {e}")
                
                progress['collected_fixtures'].append(fixture_id)
                save_progress(progress)
        
        # --- Save combined files ---
        for dtype, data_list in collection.items():
            if data_list:  # Only save if we have data
                out_path = os.path.join(raw_path, f'{dtype}.json')
                save_json(data_list, out_path)
                print(f"Saved {len(data_list)} {dtype} records to {out_path}")

        # --- Standings (once per league/season) ---
        if 'team_standings' in data_types:
            try:
                standings = fetch_team_standings(league_id, season_year)
                filepath = os.path.join(raw_path, 'standings.json')
                save_json(standings, filepath)
                print(f"Saved team standings to {filepath}")
                progress['data_types']['team_standings'] = 'completed'
            except Exception as e:
                print(f"Failed to fetch/save team standings: {e}")
            finally:
                save_progress(progress)

        # Return success results
        result = {
            'status': 'success',
            'league': league_name,
            'country': country_name,
            'season': season_name,
            'fixtures_processed': len(fixture_ids),
            'data_collected': {k: len(v) for k, v in collection.items()},
            'storage_path': raw_path
        }
        
        print(f"\n✅ Successfully collected data for {league_name} {season_name}")
        print(f"📁 Data saved to: {raw_path}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Collection failed: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'league_id': league_id,
            'season_year': season_year
        }



    
# Custom date range for current season

collect_league_data(
    league_id=103,
    season_year=2025,
    custom_start_date=datetime(2025, 3, 1),
    custom_end_date=datetime(2025, 8, 13),
    keep_progress=False
)









