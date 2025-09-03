import pandas as pd
import logging
import os
import json
from pathlib import Path

def extract_fixture_events(response):
    """
    Extracts fixture details from the API response and returns a DataFrame.
    """
    fixtures_list = []

    for fixture in response.get('response', []):
        try:
            fixture_data = fixture.get('fixture', {})
            teams = fixture.get('teams', {})
            goals = fixture.get('goals', {})
            score = fixture.get('score', {})
            league_info = fixture.get('league', {})

            # Extracting fields with safe get()
            status = fixture_data.get('status', {}).get('short', None)
            date_str = fixture_data.get('date', None)
            referee = fixture_data.get('referee', None)

            venue = fixture_data.get('venue', {})
            venue_name = venue.get('name', None)
            venue_city = venue.get('city', None)
            venue_id = venue.get('id', None)

            fixture_id = fixture_data.get('id', None)
            season = league_info.get('season', None)
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
                'status': status,
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
            print(f"Error processing fixture: {e}")
            continue

    df = pd.DataFrame(fixtures_list)
    # Convert date string to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def extract_team_statistics(response):
    """
    Extracts team statistics data from an API response and flattens it into a DataFrame.
    """
    team_stats_list = []

    # Get fixture ID from parameters if available
    fixture_id = response.get('parameters', {}).get('fixture', None)

    for team_stats in response.get('response', []):
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
            print(f"Error processing team stats for fixture {fixture_id}: {e}")
            continue

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(team_stats_list)
    return df

def extract_team_statistics_multi(response):
    import pandas as pd
    # If response is a list, process each; if dict, process just one
    if isinstance(response, list):
        dfs = [extract_team_statistics(r) for r in response]
        dfs = [df for df in dfs if not df.empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        return extract_team_statistics(response)


def extract_team_standings(response):
    """
    Extracts standings data from an API response and flattens it into a DataFrame.

    Args:
        response (dict): The API response data, expected to be a dictionary containing the standings information.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted standings data.
    """

    flattened_data = []

    # Ensure 'response' is present and contains data
    if 'response' not in response or not response['response']:
        print("No standings data found in the response.")
        return pd.DataFrame()  # Return empty DataFrame if no standings data

    try:
        # Iterate through each standings group
        for round_data in response['response'][0]['league']['standings']:
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
        print(f"KeyError: {e} - Some expected keys are missing in the response.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    team_standings_df = pd.DataFrame(flattened_data)
    #team_standings_df.columns = team_standings_df.columns.str.lower().str.replace(' ', '_')

    # Convert the list of dictionaries into a DataFrame
    return team_standings_df


def extract_player_statistics(response):
    """
    Extracts player statistics data from an API response and flattens it into a DataFrame.

    Args:
        response (dict): The API response data.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted player statistics data.
    """

    flattened_data = []

    # Get the fixture ID from the parameters
    fixture_id = response.get('parameters', {}).get('fixture', None)
    if not fixture_id:
        print("Warning: 'fixture' ID not found in the response parameters.")
        return pd.DataFrame()  # Return an empty DataFrame if no fixture_id

    # Check if the response contains any data
    if 'response' not in response or not response['response']:
        print("No player statistics data found in the response.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    # Loop over each team in the response
    for team_data in response['response']:
        team = team_data.get('team', {})
        team_name = team.get('name', 'Unknown Team')
        team_id = team.get('id', None)

        # Loop over each player in the team's data
        for player_data in team_data.get('players', []):
            try:
                # Extract player information
                player = player_data.get('player', {})
                player_id = player.get('id', None)
                player_name = player.get('name', 'Unknown Player')
                player_photo = player.get('photo', None)

                # Initialize the dictionary to store player stats
                player_stats = {
                    'fixture_id': fixture_id,
                    'team_name': team_name,
                    'team_id': team_id,
                    'player_id': player_id,
                    'player_name': player_name,
                    'player_photo': player_photo
                }

                # Loop over the player's statistics and flatten them
                for stat in player_data.get('statistics', []):
                    for stat_type, stat_value in stat.items():
                        if isinstance(stat_value, dict):
                            # Flatten sub-statistics if they exist
                            for sub_stat_type, sub_stat_value in stat_value.items():
                                player_stats[f'{stat_type}_{sub_stat_type}'] = sub_stat_value
                        else:
                            player_stats[stat_type] = stat_value

                # Append the flattened player data to the list
                flattened_data.append(player_stats)

            except KeyError as e:
                print(f"Key error: {e} - Player data might be missing a field.")
            except Exception as e:
                print(f"An error occurred while processing player data: {str(e)}")
    
    player_stats_df = pd.DataFrame(flattened_data)
    #player_stats_df.columns = player_stats_df.columns.str.lower().str.replace(' ', '_')

    # Convert the list of dictionaries into a DataFrame
    return player_stats_df

def extract_player_statistics_multi(response):
    import pandas as pd
    # If response is a list, process each; if dict, process just one
    if isinstance(response, list):
        dfs = [extract_player_statistics(r) for r in response]
        dfs = [df for df in dfs if not df.empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        return extract_player_statistics(response)


def extract_lineups(response):
    """
    Extracts all fields from the fixtures/lineups API response and returns it as a DataFrame.
    Adds clear columns for coach id and name.
    """
    import pandas as pd

    if 'response' not in response or not isinstance(response['response'], list):
        raise ValueError("Invalid API response structure. Expected 'response' key with a list of lineups.")

    lineup_data = []

    # Get the fixture ID from the parameters
    fixture_id = response.get('parameters', {}).get('fixture', None)
    if not fixture_id:
        print("Warning: 'fixture' ID not found in the response parameters.")
        return pd.DataFrame()  # Return an empty DataFrame if no fixture_id
    
    for entry in response['response']:
        try:
            # Extract team and color info
            team = entry.get('team', {})
            team_colors = team.get('colors', {})
            team_id = team.get('id', None)
            team_name = team.get('name', None)
            team_logo = team.get('logo', None)

            # Coach info
            coach = entry.get('coach', {})
            coach_id = coach.get('id', None)
            coach_name = coach.get('name', None)
            coach_photo = coach.get('photo', None)

            # Formation
            formation = entry.get('formation', None)

            # StartXI
            startXI = entry.get('startXI', [])
            for player in startXI:
                player_info = player.get('player', {})
                lineup_data.append({
                    'fixture_id': fixture_id,
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_logo': team_logo,
                    'team_colors_player_primary': team_colors.get('player', {}).get('primary', None),
                    'team_colors_player_number': team_colors.get('player', {}).get('number', None),
                    'team_colors_player_border': team_colors.get('player', {}).get('border', None),
                    'team_colors_goalkeeper_primary': team_colors.get('goalkeeper', {}).get('primary', None),
                    'team_colors_goalkeeper_number': team_colors.get('goalkeeper', {}).get('number', None),
                    'team_colors_goalkeeper_border': team_colors.get('goalkeeper', {}).get('border', None),
                    'coach_id': coach_id,
                    'coach_name': coach_name,
                    'coach_photo': coach_photo,
                    'formation': formation,
                    'player_id': player_info.get('id', None),
                    'player_name': player_info.get('name', None),
                    'player_number': player_info.get('number', None),
                    'player_pos': player_info.get('pos', None),
                    'player_grid': player_info.get('grid', None),
                    'is_substitute': False
                })

            # Substitutes
            substitutes = entry.get('substitutes', [])
            for player in substitutes:
                player_info = player.get('player', {})
                lineup_data.append({
                    'fixture_id': fixture_id,
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_logo': team_logo,
                    'team_colors_player_primary': team_colors.get('player', {}).get('primary', None),
                    'team_colors_player_number': team_colors.get('player', {}).get('number', None),
                    'team_colors_player_border': team_colors.get('player', {}).get('border', None),
                    'team_colors_goalkeeper_primary': team_colors.get('goalkeeper', {}).get('primary', None),
                    'team_colors_goalkeeper_number': team_colors.get('goalkeeper', {}).get('number', None),
                    'team_colors_goalkeeper_border': team_colors.get('goalkeeper', {}).get('border', None),
                    'coach_id': coach_id,
                    'coach_name': coach_name,
                    'coach_photo': coach_photo,
                    'formation': formation,
                    'player_id': player_info.get('id', None),
                    'player_name': player_info.get('name', None),
                    'player_number': player_info.get('number', None),
                    'player_pos': player_info.get('pos', None),
                    'player_grid': player_info.get('grid', None),
                    'is_substitute': True
                })
        except Exception as e:
            print(f"Error processing lineup: {e}")
            continue

    df = pd.DataFrame(lineup_data)
    return df


def extract_lineups_multi(response):
    import pandas as pd
    # If response is a list, process each; if dict, process just one
    if isinstance(response, list):
        dfs = [extract_lineups(r) for r in response]
        dfs = [df for df in dfs if not df.empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        return extract_lineups(response)


def extract_injuries(response):
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
    if 'response' not in response or not isinstance(response['response'], list):
        raise ValueError("Invalid API response structure. Expected 'response' key with a list of injuries.")

    # Extract injury data into a list of dictionaries
    injury_data = []

    for entry in response['response']:
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
            print(f"Error processing injuries: {e}")
            continue

    # Convert the list of dictionaries into a DataFrame
    injuries_df = pd.DataFrame(injury_data)

    return injuries_df

def extract_injuries_multi(response):
    import pandas as pd
    # If response is a list, process each; if dict, process just one
    if isinstance(response, list):
        dfs = [extract_injuries(r) for r in response]
        dfs = [df for df in dfs if not df.empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        return extract_injuries(response)




def process_all_leagues_seasons(base_dir="data"):
    """
    Processes all leagues and seasons found in the raw directory structure.
    Creates matching extracted directories for output.
    """
    raw_base = os.path.join(base_dir, "raw")
    extracted_base = os.path.join(base_dir, "extracted")
    
    if not os.path.exists(raw_base):
        raise FileNotFoundError(f"Raw data directory not found: {raw_base}")
    
    # Walk through all country directories
    for country in os.listdir(raw_base):
        country_raw_path = os.path.join(raw_base, country)
        if not os.path.isdir(country_raw_path):
            continue
            
        # Process each league in the country
        for league in os.listdir(country_raw_path):
            league_raw_path = os.path.join(country_raw_path, league)
            if not os.path.isdir(league_raw_path):
                continue
                
            # Process each season in the league
            for season in os.listdir(league_raw_path):
                season_raw_path = os.path.join(league_raw_path, season)
                if not os.path.isdir(season_raw_path):
                    continue
                    
                # Create matching extracted directory
                season_extracted_path = os.path.join(extracted_base, country, league, season)
                os.makedirs(season_extracted_path, exist_ok=True)
                
                print(f"Processing {country}/{league}/{season}...")
                extract_and_save_all_jsons(season_raw_path, season_extracted_path)

def extract_and_save_all_jsons(raw_dir, out_dir):
    """
    Processes all JSON files in a directory using the appropriate extractors.
    """
    # Map file patterns to extract functions
    extractors = [
        ('fixture_events.json', extract_fixture_events),
        ('team_statistics.json', extract_team_statistics_multi),
        ('player_statistics.json', extract_player_statistics_multi),
        ('team_standings.json', extract_team_standings),
        ('injuries.json', extract_injuries_multi),
        ('lineups.json', extract_lineups_multi),

    ]

    # Process each JSON file in the raw directory
    for filename in os.listdir(raw_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(raw_dir, filename)
        csv_filename = filename.replace('.json', '.csv')
        csv_path = os.path.join(out_dir, csv_filename)
        
        # Skip if CSV already exists and is newer than JSON
        if os.path.exists(csv_path):
            json_mtime = os.path.getmtime(json_path)
            csv_mtime = os.path.getmtime(csv_path)
            if csv_mtime > json_mtime:
                continue
                
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        # Find the right extractor
        for pattern, extractor in extractors:
            if filename.startswith(pattern) or filename == pattern:
                print(f"Extracting {filename}...")
                try:
                    df = extractor(data)
                    if df is not None and not df.empty:
                        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                        print(f"Saved {csv_filename}")
                    else:
                        print(f"No data extracted from {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                break
        else:
            print(f"No extractor found for {filename}, skipping")

if __name__ == "__main__":
    # Process all leagues and seasons when run directly
    process_all_leagues_seasons()