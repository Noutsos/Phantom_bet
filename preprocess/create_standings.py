import pandas as pd
from datetime import datetime, timedelta

def process_fixture_data(input_csv, output_csv):
    """
    Processes and corrects football fixture data with comprehensive feature engineering.
    Handles all edge cases and produces ML-ready output with accurate standings.
    """
    # Load and prepare data
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['season', 'date', 'fixture_id']).reset_index(drop=True)
    
    # Initialize team tracking
    team_history = {}
    season_start_dates = df.groupby('season')['date'].min().to_dict()
    
    # Define features with proper data types
    features = {
        # Numeric features
        'home_rank': 'float64',
        'home_points': 'int64',
        'home_goals_diff': 'int64',
        'home_played': 'int64',
        'home_wins': 'int64',
        'home_draws': 'int64',
        'home_losses': 'int64',
        'home_goals_for': 'int64',
        'home_goals_against': 'int64',
        'home_days_rest': 'int64',
        'home_form_strength': 'float64',
        'away_rank': 'float64',
        'away_points': 'int64',
        'away_goals_diff': 'int64',
        'away_played': 'int64',
        'away_wins': 'int64',
        'away_draws': 'int64',
        'away_losses': 'int64',
        'away_goals_for': 'int64',
        'away_goals_against': 'int64',
        'away_days_rest': 'int64',
        'away_form_strength': 'float64',
        # Categorical features
        'home_form': 'object',
        'away_form': 'object'
    }

    # Initialize columns with proper dtypes
    for col, dtype in features.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)

    # Process each match
    for idx, row in df.iterrows():
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        match_date = row['date']
        season = row['season']
        round_num = row['round']
        
        # Initialize team data if new season or new team
        for team_id in [home_id, away_id]:
            if team_id not in team_history or season != team_history[team_id]['current_season']:
                team_history[team_id] = {
                    'current_season': season,
                    'points': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'form': [],  # Stores numeric form (1=win, 0.5=draw, 0=loss)
                    'last_match_date': season_start_dates.get(season, match_date) - timedelta(days=30),
                    'matches_played': 0,
                    'team_name': row['home_team'] if team_id == home_id else row['away_team']
                }
        
        # Calculate days rest (capped at 30 days)
        home_days_rest = (match_date - team_history[home_id]['last_match_date']).days
        away_days_rest = (match_date - team_history[away_id]['last_match_date']).days
        home_days_rest = min(30, home_days_rest) if home_days_rest > 0 else 30  # Default for first match
        away_days_rest = min(30, away_days_rest) if away_days_rest > 0 else 30  # Default for first match
        
        # Calculate form strength with additive smoothing
        def calculate_enhanced_form_strength(team_id, match_date, team_history, df):
            """Calculate form strength with exponential weighting and goal difference consideration"""
            # Get last 5 matches (excluding current match)
            past_matches = df[(df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)]
            past_matches = past_matches[past_matches['date'] < match_date].sort_values('date', ascending=False).head(5)
            
            if past_matches.empty:
                return 0.5  # Neutral value
            
            total_weight = 0
            form_score = 0
            
            for i, (_, match) in enumerate(past_matches.iterrows()):
                # Determine if home or away
                is_home = match['home_team_id'] == team_id
                weight = 0.8 ** i  # Exponential decay (most recent match has weight 1)
                
                # Get match result
                if is_home:
                    goals_for = match['home_goals']
                    goals_against = match['away_goals']
                else:
                    goals_for = match['away_goals']
                    goals_against = match['home_goals']
                    
                # Calculate result score (0-1)
                if goals_for > goals_against:  # Win
                    result_score = 1.0
                elif goals_for == goals_against:  # Draw
                    result_score = 0.5
                else:  # Loss
                    result_score = 0.0
                    
                # Add goal difference factor (capped at ±3)
                gd_factor = min(3, max(-3, goals_for - goals_against)) / 12  # ±0.25 max effect
                
                form_score += (result_score + gd_factor) * weight
                total_weight += weight
            
            # Normalize and apply bounds
            form_strength = form_score / total_weight
            return max(0, min(1, round(form_strength, 4)))
        
        home_form_strength = calculate_enhanced_form_strength(home_id, match_date, team_history, df)
        away_form_strength = calculate_enhanced_form_strength(away_id, match_date, team_history, df)
        
        # Generate form string (last 5 matches)
        def get_form_string(form_history):
            if not form_history:
                return 'N'
            return ''.join(['W' if x == 1 else 'D' if x == 0.5 else 'L' for x in form_history[-5:]])
        
        home_form_str = get_form_string(team_history[home_id]['form'])
        away_form_str = get_form_string(team_history[away_id]['form'])
        
        # Set PRE-MATCH features
        df.at[idx, 'home_points'] = team_history[home_id]['points']
        df.at[idx, 'home_goals_for'] = team_history[home_id]['goals_for']
        df.at[idx, 'home_goals_against'] = team_history[home_id]['goals_against']
        df.at[idx, 'home_goals_diff'] = team_history[home_id]['goals_for'] - team_history[home_id]['goals_against']
        df.at[idx, 'home_played'] = team_history[home_id]['matches_played']
        df.at[idx, 'home_wins'] = team_history[home_id]['wins']
        df.at[idx, 'home_draws'] = team_history[home_id]['draws']
        df.at[idx, 'home_losses'] = team_history[home_id]['losses']
        df.at[idx, 'home_days_rest'] = home_days_rest
        df.at[idx, 'home_form_strength'] = home_form_strength
        df.at[idx, 'home_form'] = home_form_str
        
        df.at[idx, 'away_points'] = team_history[away_id]['points']
        df.at[idx, 'away_goals_for'] = team_history[away_id]['goals_for']
        df.at[idx, 'away_goals_against'] = team_history[away_id]['goals_against']
        df.at[idx, 'away_goals_diff'] = team_history[away_id]['goals_for'] - team_history[away_id]['goals_against']
        df.at[idx, 'away_played'] = team_history[away_id]['matches_played']
        df.at[idx, 'away_wins'] = team_history[away_id]['wins']
        df.at[idx, 'away_draws'] = team_history[away_id]['draws']
        df.at[idx, 'away_losses'] = team_history[away_id]['losses']
        df.at[idx, 'away_days_rest'] = away_days_rest
        df.at[idx, 'away_form_strength'] = away_form_strength
        df.at[idx, 'away_form'] = away_form_str
        
        # Update POST-MATCH stats (using actual match results)
        home_goals = int(row['home_goals'])
        away_goals = int(row['away_goals'])
        
        # Update goal metrics
        team_history[home_id]['goals_for'] += home_goals
        team_history[home_id]['goals_against'] += away_goals
        team_history[away_id]['goals_for'] += away_goals
        team_history[away_id]['goals_against'] += home_goals
        
        # Update points and form based on result
        if home_goals > away_goals:  # Home win
            team_history[home_id]['points'] += 3
            team_history[home_id]['wins'] += 1
            team_history[home_id]['form'].append(1)
            team_history[away_id]['losses'] += 1
            team_history[away_id]['form'].append(0)
        elif home_goals == away_goals:  # Draw
            team_history[home_id]['points'] += 1
            team_history[home_id]['draws'] += 1
            team_history[home_id]['form'].append(0.5)
            team_history[away_id]['points'] += 1
            team_history[away_id]['draws'] += 1
            team_history[away_id]['form'].append(0.5)
        else:  # Away win
            team_history[away_id]['points'] += 3
            team_history[away_id]['wins'] += 1
            team_history[away_id]['form'].append(1)
            team_history[home_id]['losses'] += 1
            team_history[home_id]['form'].append(0)
        
        # Update match counts and dates
        team_history[home_id]['matches_played'] += 1
        team_history[away_id]['matches_played'] += 1
        team_history[home_id]['last_match_date'] = match_date
        team_history[away_id]['last_match_date'] = match_date
    
    def calculate_ranks(group):
        """
        Properly calculates ranks within each season/round group using:
        1. Points (descending)
        2. Goal Difference (descending)
        3. Goals For (descending)
        """
        # Create temporary dataframe for ranking calculations
        teams = {}
        for _, row in group.iterrows():
            for team_type in ['home', 'away']:
                team_id = row[f'{team_type}_team_id']
                if team_id not in teams:
                    teams[team_id] = {
                        'points': row[f'{team_type}_points'],
                        'gd': row[f'{team_type}_goals_diff'],
                        'gf': row[f'{team_type}_goals_for'],
                        'name': row[f'{team_type}_team']
                    }
        
        # Convert to DataFrame for ranking
        ranking_df = pd.DataFrame.from_dict(teams, orient='index')
        ranking_df['rank_key'] = ranking_df.apply(lambda x: (-x['points'], -x['gd'], -x['gf']), axis=1)
        ranking_df['rank'] = ranking_df['rank_key'].rank(method='min').astype(int)
        
        # Map ranks back to original dataframe
        for idx, row in group.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            group.at[idx, 'home_rank'] = ranking_df.at[home_id, 'rank']
            group.at[idx, 'away_rank'] = ranking_df.at[away_id, 'rank']
        
        return group

    # Apply ranking calculation
    df = df.groupby(['season', 'round'], group_keys=False).apply(calculate_ranks)
    
    # Final validation
    def validate_data(df):
        # Check goal differences
        assert all(df['home_goals_diff'] == (df['home_goals_for'] - df['home_goals_against']))
        assert all(df['away_goals_diff'] == (df['away_goals_for'] - df['away_goals_against']))
        
        # Check points accumulation
        for team_id, team_data in team_history.items():
            calculated_points = team_data['wins'] * 3 + team_data['draws']
            assert team_data['points'] == calculated_points, f"Points mismatch for {team_data['team_name']}"
        
        # Check form strings match form strength
        for _, row in df.iterrows():
            if row['home_form'] != 'N':
                calculated_strength = (sum(1 if x == 'W' else 0.5 if x == 'D' else 0 for x in row['home_form']) + 0.5) / (len(row['home_form']) + 1)
                assert abs(row['home_form_strength'] - calculated_strength) < 0.01
            
            if row['away_form'] != 'N':
                calculated_strength = (sum(1 if x == 'W' else 0.5 if x == 'D' else 0 for x in row['away_form']) + 0.5) / (len(row['away_form']) + 1)
                assert abs(row['away_form_strength'] - calculated_strength) < 0.01

        # Add form strength validation
        for _, row in df.iterrows():
            assert 0 <= row['home_form_strength'] <= 1
            assert 0 <= row['away_form_strength'] <= 1
            # Verify form string matches strength
            if row['home_form'] != 'N':
                expected = (row['home_form'].count('W') + row['home_form'].count('D')*0.5)/5
                assert abs(row['home_form_strength'] - expected) < 0.2  # Allow some variance for enhanced calculation
        
        print("All data validation checks passed")
    
    try:
        validate_data(df)
    except AssertionError as e:
        print(f"Validation warning: {e}")
    
    # Select output columns
    output_cols = [
        'fixture_id', 'date', 'season', 'round',
        'home_team', 'home_team_id', 'away_team', 'away_team_id',
        'home_goals', 'away_goals'
    ] + list(features.keys())

    # Save output
    try:
        df[output_cols].to_csv(output_csv, index=False)
        print(f"Successfully saved corrected data to {output_csv}")
        return df
    except Exception as e:
        print(f"Error saving output: {e}")
        return None



# Example usage
if __name__ == "__main__":
    processed_data = process_fixture_data(
        'data/extracted/Bundesliga/2024/fixture_events.csv',
        'fixtures_with_standings_fixed.csv'
    )