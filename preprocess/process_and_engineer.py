import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def process_fixture_events(df, lag_window=3, league_avg_goals=1.5):
    """
    Processes fixture data with:
    - Time-lagged features
    - First-match handling
    - Opponent-aware stats
    - Leakage prevention
    - Multi-class outcome column
    
    Outcome encoding:
    - 0: Away win
    - 1: Draw
    - 2: Home win
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['home_team_id', 'date'])
    
    # --- Create Outcome Column (MUST BE FIRST STEP TO PREVENT LEAKAGE) ---
    df['outcome'] = np.select(
        condlist=[
            df['home_goals'] > df['away_goals'],  # Home win
            df['home_goals'] == df['away_goals']  # Draw
        ],
        choicelist=[2, 1],  # 2=home win, 1=draw
        default=0  # 0=away win
    )
    
    # --- Static Features ---
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # --- Team-Level Features ---
    for team_type in ['home', 'away']:
        team_id = f'{team_type}_team_id'
        opp_type = 'away' if team_type == 'home' else 'home'
        
        # Goals scored (fill first match with league average)
        df[f'{team_type}_goals_avg'] = (df.groupby(team_id)[f'{team_type}_goals']
                                      .shift(1)
                                      .rolling(lag_window, min_periods=1)
                                      .mean()
                                      .fillna(league_avg_goals))
        
        # Goals conceded
        df[f'{team_type}_goals_conceded_avg'] = (df.groupby(team_id)[f'{opp_type}_goals']
                                               .shift(1)
                                               .rolling(lag_window, min_periods=1)
                                               .mean()
                                               .fillna(league_avg_goals))
        
        # Win/loss stats
        df[f'{team_type}_winner'] = (df[f'{team_type}_goals'] > df[f'{opp_type}_goals']).astype(int)
        df[f'{team_type}_win_streak'] = (df.groupby(team_id)[f'{team_type}_winner']
                                        .shift(1)
                                        .rolling(lag_window, min_periods=1)
                                        .sum()
                                        .fillna(0))
    
    # --- Head-to-Head Features ---
    df = df.sort_values(['home_team_id', 'away_team_id', 'date'])
    df['h2h_home_wins'] = (df.groupby(['home_team_id', 'away_team_id'])['home_winner']
                           .shift(1)
                           .rolling(5, min_periods=1)
                           .sum()
                           .fillna(0))
    
    # --- First-Match Flags ---
    df['is_home_first_match'] = ~df.duplicated(['home_team_id'], keep='first')
    df['is_away_first_match'] = ~df.duplicated(['away_team_id'], keep='first')
    
    # --- Cleanup ---
    to_drop = ['home_goals', 'away_goals', 'halftime_home', 'halftime_away',
               'fulltime_home', 'fulltime_away', 'home_winner', 'away_winner']
    return df.drop(columns=to_drop, errors='ignore')

# Example usage:
raw_fixture_events = pd.read_csv("data/extracted/Serie A/2021/fixture_events.csv")
#engineered_data = engineer_fixture_features(raw_fixture_events, lag_window=5)
# Save the processed data
#engineered_data.to_csv("data/processed/Serie A/2021/fixtures_enhanced.csv", index=False)


def process_injuries(injuries_df):
    """
    Final debugged version with proper column handling.
    Returns engineered features at (fixture_id × team_id) level.
    """
    df = injuries_df.copy()
    
    # Validate required columns exist
    required_cols = ['fixture_id', 'date', 'team_id', 'player_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team_id', 'date'])
    
    # --- Feature 1: Current Match Injury Counts ---
    current_features = df.groupby(['fixture_id', 'team_id', 'date']).agg({
        'player_id': 'count',
        'type': lambda x: x.str.contains('Muscle|Fracture|ACL|Rupture', case=False, na=False).sum()
    }).rename(columns={
        'player_id': 'current_injuries',
        'type': 'current_serious_injuries'
    }).reset_index()
    
    # --- Feature 2: Rolling Injury Burden ---
    # Create daily injury counts
    daily_counts = (
        df.groupby(['team_id', pd.Grouper(key='date', freq='D')])
        .size()
        .rename('daily_injuries')
        .reset_index()
    )
    
    # Calculate rolling averages - fixed implementation
    def add_rolling_features(group):
        group = group.set_index('date')
        group['injuries_7d_avg'] = group['daily_injuries'].rolling('7D', min_periods=1).mean()
        group['injuries_30d_avg'] = group['daily_injuries'].rolling('30D', min_periods=1).mean()
        return group.reset_index()
    
    rolling_features = (
        daily_counts
        .groupby('team_id', group_keys=False)
        .apply(add_rolling_features)
        .reset_index(drop=True)
    )
    
    # --- Feature 3: Key Injury Types ---
    if 'type' in df.columns:
        # Create injury type flags
        injury_types = df[df['type'].notna()].copy()
        injury_types['is_muscle'] = injury_types['type'].str.contains('Muscle', case=False, na=False)
        injury_types['is_serious'] = injury_types['type'].str.contains('Fracture|ACL|Rupture', case=False, na=False)
        
        # Calculate daily type counts
        type_counts = (
            injury_types.groupby(['team_id', pd.Grouper(key='date', freq='D')])
            [['is_muscle', 'is_serious']]
            .sum()
            .reset_index()
        )
        
        # Add rolling type counts
        def add_type_rolling(group):
            group = group.set_index('date')
            group['muscle_30d'] = group['is_muscle'].rolling('30D', min_periods=1).sum()
            group['serious_30d'] = group['is_serious'].rolling('30D', min_periods=1).sum()
            return group.reset_index()
        
        type_features = (
            type_counts
            .groupby('team_id', group_keys=False)
            .apply(add_type_rolling)
            .reset_index(drop=True)
        )
    else:
        # Create empty type features if no type data
        unique_dates = daily_counts['date'].unique()
        type_features = pd.DataFrame({
            'team_id': np.repeat(daily_counts['team_id'].unique(), len(unique_dates)),
            'date': np.tile(unique_dates, len(daily_counts['team_id'].unique())),
            'muscle_30d': 0,
            'serious_30d': 0
        })
    
    # --- Combine All Features ---
    # First merge current with rolling
    features = current_features.merge(
        rolling_features,
        on=['team_id', 'date'],
        how='left'
    )
    
    # Then merge with type features
    features = features.merge(
        type_features[['team_id', 'date', 'muscle_30d', 'serious_30d']],
        on=['team_id', 'date'],
        how='left'
    ).fillna(0)
    
    # Calculate composite injury burden score
    features['injury_burden'] = (
        0.4 * features['current_injuries'] +
        0.3 * features['injuries_7d_avg'] +
        0.2 * features['muscle_30d'] +
        0.1 * features['serious_30d']
    )
    
    # Select final features
    final_features = [
        'fixture_id', 'team_id', 'date',
        'current_injuries', 'current_serious_injuries',
        'injuries_7d_avg', 'injuries_30d_avg',
        'muscle_30d', 'serious_30d',
        'injury_burden'
    ]
    
    return features[final_features].drop_duplicates()

# Example usage:
#raw_injuries = pd.read_csv("data/extracted/Serie A/2021/injuries.csv")
#injury_features = engineer_injury_features(raw_injuries)


# Save the processed data
#injury_features.to_csv("data/processed/Serie A/2021/injuries_enhanced.csv", index=False)

def process_lineups(lineup_df):
    """
    Processes lineup data into team-level features while avoiding data leakage.
    Returns engineered features at (fixture_id × team_id) level.
    
    Parameters:
        lineup_df (pd.DataFrame): Raw lineup data with multiple rows per fixture-team
        
    Returns:
        pd.DataFrame: Engineered features ready for modeling
    """
    df = lineup_df.copy()
    
    # Validate required columns
    required_cols = ['fixture_id', 'team_id', 'player_id', 'player_pos', 'is_substitute']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # --- Feature 1: Basic Team Stats ---
    team_features = df.groupby(['fixture_id', 'team_id']).agg({
        'player_id': 'count',
        'is_substitute': 'sum',
        'formation': 'first',
        'coach_id': 'first'
    }).rename(columns={
        'player_id': 'squad_size',
        'is_substitute': 'substitutes_count',
        'formation': 'formation',
        'coach_id': 'coach_id'
    }).reset_index()
    
    # --- Feature 2: Position Distribution ---
    position_counts = (
        df[df['is_substitute'] == False]  # Only starting players
        .groupby(['fixture_id', 'team_id', 'player_pos'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('pos_')
        .reset_index()
    )
    
    # --- Feature 3: Player Experience (would need external data) ---
    # This would require merging with player stats data
    # For now just include placeholder
    #team_features['avg_player_experience'] = 0  # Placeholder
    
    # --- Feature 4: Formation Features ---
    if 'formation' in df.columns:
        team_features['formation'] = team_features['formation'].fillna('Unknown')
        # One-hot encode common formations
        common_formations = ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1']
        for formation in common_formations:
            team_features[f'formation_{formation}'] = (
                team_features['formation'].str.contains(formation).astype(int)
            )
    else:
        team_features['formation'] = 'Unknown'
    
    # --- Feature 5: Substitute Quality ---
    # Calculate average position strength of substitutes
    if 'player_pos' in df.columns:
        pos_strength = {
            'G': 1, 'D': 2, 'M': 3, 'F': 4  # Simple position importance scale
        }
        df['pos_strength'] = df['player_pos'].map(pos_strength).fillna(0)
        
        sub_strength = (
            df[df['is_substitute'] == True]
            .groupby(['fixture_id', 'team_id'])['pos_strength']
            .mean()
            .reset_index()
            .rename(columns={'pos_strength': 'substitute_quality'})
        )
        
        team_features = team_features.merge(
            sub_strength, 
            on=['fixture_id', 'team_id'], 
            how='left'
        ).fillna({'substitute_quality': 0})
    else:
        team_features['substitute_quality'] = 0
    
    # --- Combine All Features ---
    # Merge position counts with team features
    final_features = team_features.merge(
        position_counts,
        on=['fixture_id', 'team_id'],
        how='left'
    ).fillna(0)
    
    # Add time-safe features (would need fixture dates)
    # final_features = add_time_features(final_features, fixtures_df)
    
    return final_features



#raw_lineups = pd.read_csv("data/extracted/Serie A/2021/lineups.csv")

#lineup_features = process_lineup_features(raw_lineups)
# Save the processed data
#lineup_features.to_csv("data/processed/Serie A/2021/lineups_enhanced.csv", index=False)





def process_team_statistics(stats_df, fixtures_df):
    """
    Processes team stats keeping ONLY 5-match rolling averages (no simple match stats).
    Requires fixture dates for proper time-series calculation.
    """
    # Validate inputs
    required_stats = ['fixture_id', 'team_id']
    missing = [col for col in required_stats if col not in stats_df.columns]
    if missing:
        raise ValueError(f"Missing columns in stats_df: {missing}")
    
    required_fixtures = ['fixture_id', 'date']
    missing = [col for col in required_fixtures if col not in fixtures_df.columns]
    if missing:
        raise ValueError(f"Missing columns in fixtures_df: {missing}")

    # Create working copy and convert percentages
    df = stats_df.copy()
    for col in df.columns:
        if '%' in str(col):
            df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100

    # Get numeric columns (excluding IDs)
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                  if col not in required_stats]

    # Merge with fixtures and sort
    fixtures_df = fixtures_df[required_fixtures].copy()
    fixtures_df['date'] = pd.to_datetime(fixtures_df['date'])
    df = df.merge(fixtures_df, on='fixture_id').sort_values(['team_id', 'date'])

    # --- Calculate 5-Match Rolling Averages ---
    rolling_features = []
    for team in df['team_id'].unique():
        team_data = df[df['team_id'] == team].copy()
        
        # Calculate rolling stats (excluding current match)
        for col in numeric_cols:
            team_data[f'rolling_5_{col}'] = (
                team_data[col].shift(1)
                .rolling(5, min_periods=1)
                .mean()
            )
        rolling_features.append(team_data)
    
    # Combine all teams
    df = pd.concat(rolling_features).sort_values(['team_id', 'date'])

    # --- Create Derived Features from Rolling Averages ---
    # Shot accuracy (from rolling averages)
    if all(f'rolling_5_{col}' in df.columns for col in ['Shots on Goal', 'Total Shots']):
        df['rolling_shot_accuracy'] = (
            df['rolling_5_Shots on Goal'] / df['rolling_5_Total Shots'].replace(0, 1)
        ).clip(0, 1)

    # Defensive intensity (from rolling averages)
    defensive_cols = []
    for col in ['Fouls', 'Blocked Shots', 'Goalkeeper Saves']:
        if f'rolling_5_{col}' in df.columns:
            defensive_cols.append(f'rolling_5_{col}')
    if defensive_cols:
        df['rolling_defensive_actions'] = df[defensive_cols].sum(axis=1)

    # Team style (from rolling possession)
    if 'rolling_5_Ball Possession' in df.columns:
        df['style'] = pd.cut(
            df['rolling_5_Ball Possession'],
            bins=[0, 40, 60, 100],
            labels=['counter', 'balanced', 'possession']
        )
        df = pd.get_dummies(df, columns=['style'], prefix='style')

    # Select only rolling features
    keep_cols = ['fixture_id', 'team_id'] + \
               [col for col in df.columns if col.startswith('rolling_5_')] + \
               [col for col in df.columns if col.startswith('rolling_') and not col.startswith('rolling_5_')] + \
               [col for col in df.columns if col.startswith('style_')]

    return df[keep_cols].fillna(0)

raw_team_stats = pd.read_csv("data/extracted/Serie A/2021/team_statistics.csv")


# Save the processed data

#team_stats_features = process_team_stats(raw_team_stats, raw_fixture_events)
#team_stats_features.to_csv("data/processed/Serie A/2021/team_statistics_enhanced.csv", index=False)

def process_player_statistics(player_stats_df, fixtures_df):
    """
    Processes player stats with 5-match rolling averages.
    Returns ONLY rolling features (no single-match stats).
    
    Parameters:
        player_stats_df: DataFrame with player match stats
        fixtures_df: DataFrame with fixture dates (must contain fixture_id and date)
    
    Returns:
        DataFrame with rolling features at (fixture_id × player_id) level
    """
    # Validate inputs
    required_player_cols = ['fixture_id', 'team_id', 'player_id', 'games_rating']
    missing = [col for col in required_player_cols if col not in player_stats_df.columns]
    if missing:
        raise ValueError(f"Missing columns in player_stats_df: {missing}")
    
    if not all(col in fixtures_df.columns for col in ['fixture_id', 'date']):
        raise ValueError("fixtures_df must contain 'fixture_id' and 'date'")

    # Create working copy and convert percentages
    df = player_stats_df.copy()
    for col in df.columns:
        if '%' in str(col):
            df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100

    # Merge with fixtures and sort chronologically
    fixtures_df = fixtures_df[['fixture_id', 'date']].copy()
    fixtures_df['date'] = pd.to_datetime(fixtures_df['date'])
    df = df.merge(fixtures_df, on='fixture_id').sort_values(['player_id', 'date'])

    # Select numeric performance columns (excluding IDs and metadata)
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col not in ['fixture_id', 'team_id', 'player_id', 'games_number', 'games_minutes']]

    # --- Calculate 5-Match Rolling Averages ---
    rolling_features = []
    for player in df['player_id'].unique():
        player_data = df[df['player_id'] == player].copy()
        
        # Calculate rolling stats (excluding current match)
        for col in numeric_cols:
            player_data[f'rolling_5_{col}'] = (
                player_data[col].shift(1)
                .rolling(5, min_periods=1)
                .mean()
            )
        
        # Add minutes-weighted features for key metrics
        if 'games_minutes' in player_data.columns:
            for metric in ['games_rating', 'passes_accuracy', 'shots_on']:
                if metric in numeric_cols:
                    player_data[f'weighted_5_{metric}'] = (
                        (player_data[metric] * player_data['games_minutes']).shift(1)
                        .rolling(5, min_periods=1)
                        .sum() / 
                        player_data['games_minutes'].shift(1)
                        .rolling(5, min_periods=1)
                        .sum()
                    ).fillna(0)
        
        rolling_features.append(player_data)

    # Combine all players
    df = pd.concat(rolling_features).sort_values(['player_id', 'date'])

    # --- Create Derived Features ---
    # Performance composites
    if all(col in df.columns for col in ['rolling_5_shots_on', 'rolling_5_shots_total']):
        df['rolling_shot_accuracy'] = (
            df['rolling_5_shots_on'] / df['rolling_5_shots_total'].replace(0, 1)
        ).clip(0, 1)
    
    if all(col in df.columns for col in ['rolling_5_tackles_total', 'rolling_5_interceptions']):
        df['rolling_defensive_impact'] = (
            df['rolling_5_tackles_total'] + df['rolling_5_interceptions']
        )
    
    # Positional performance indicators
    if 'games_position' in df.columns:
        position_dummies = pd.get_dummies(df['games_position'], prefix='pos')
        df = pd.concat([df, position_dummies], axis=1)

    # --- Select Final Features ---
    rolling_feature_cols = [col for col in df.columns if col.startswith('rolling_5_')]
    weighted_feature_cols = [col for col in df.columns if col.startswith('weighted_5_')]
    derived_feature_cols = [col for col in df.columns if col.startswith('rolling_') and 
                          not col.startswith('rolling_5_') and
                          not col.startswith('weighted_5_')]
    position_cols = [col for col in df.columns if col.startswith('pos_')]
    
    keep_cols = ['fixture_id', 'team_id', 'player_id'] + \
               rolling_feature_cols + weighted_feature_cols + \
               derived_feature_cols + position_cols

    return df[keep_cols].fillna(0)


raw_player_stats = pd.read_csv("data/extracted/Serie A/2021/player_statistics.csv")


# Save the processed data

#player_features = process_player_stats(raw_player_stats, raw_fixture_events)
#player_features.to_csv("data/processed/Serie A/2021/player_statistics_enhanced.csv", index=False)




raw_team_standings = pd.read_csv("data/extracted/Serie A/2021/team_standings.csv")


# Save the processed data

#standings_features = process_standings(raw_team_standings, raw_fixture_events)

#standings_features.to_csv("data/processed/Serie A/2021/team_standings_enhanced.csv", index=False)




def process_team_standings(fixtures_df, output_path='data/processed/Serie A/2021/team_standings_enhanced.csv'):
    """
    Generates complete standings CSV from fixture events data alone.
    Final fixed version with robust form string calculation.
    """
    # Extract basic match results
    matches = fixtures_df[[
        'fixture_id', 'date', 
        'home_team_id', 'home_team', 'home_team_flag',
        'away_team_id', 'away_team', 'away_team_flag',
        'home_goals', 'away_goals'
    ]].copy()
    
    # Convert dates and sort chronologically
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date')
    
    # Create home and away records
    home_records = matches.rename(columns={
        'home_team_id': 'team_id',
        'home_team': 'team_name',
        'home_team_flag': 'team_logo',
        'away_team_id': 'opponent_id',
        'home_goals': 'goals_for',
        'away_goals': 'goals_against'
    })
    home_records['is_home'] = True
    
    away_records = matches.rename(columns={
        'away_team_id': 'team_id',
        'away_team': 'team_name',
        'away_team_flag': 'team_logo',
        'home_team_id': 'opponent_id',
        'away_goals': 'goals_for',
        'home_goals': 'goals_against'
    })
    away_records['is_home'] = False
    
    # Combine all matches
    all_matches = pd.concat([home_records, away_records])
    
    # Calculate match outcomes
    all_matches['points'] = all_matches.apply(
        lambda x: 3 if x['goals_for'] > x['goals_against'] else 
        1 if x['goals_for'] == x['goals_against'] else 0,
        axis=1
    )
    all_matches['wins'] = (all_matches['points'] == 3).astype(int)
    all_matches['draws'] = (all_matches['points'] == 1).astype(int)
    all_matches['losses'] = (all_matches['points'] == 0).astype(int)
    
    # Calculate cumulative stats for each team
    standings = []
    form_map = {3: 'W', 1: 'D', 0: 'L'}
    
    for team in all_matches['team_id'].unique():
        team_matches = all_matches[all_matches['team_id'] == team].copy()
        team_matches = team_matches.sort_values('date')
        
        # Cumulative totals
        cum_cols = ['points', 'goals_for', 'goals_against', 'wins', 'draws', 'losses']
        for col in cum_cols:
            team_matches[f'cum_{col}'] = team_matches[col].cumsum()
        
        team_matches['cum_goal_diff'] = team_matches['cum_goals_for'] - team_matches['cum_goals_against']
        
        # Home/away splits
        for venue in ['home', 'away']:
            venue_matches = team_matches[team_matches['is_home'] == (venue == 'home')]
            for col in ['points', 'goals_for', 'goals_against', 'wins', 'draws', 'losses']:
                team_matches[f'cum_{venue}_{col}'] = venue_matches[col].cumsum()
        
        # Calculate form (last 5 matches) - ROBUST VERSION
        team_matches['form_letter'] = team_matches['points'].map(form_map)
        
        # Custom function to calculate form string with proper length handling
        def calculate_form(series):
            form_str = []
            for i in range(len(series)):
                start_idx = max(0, i - 5)
                # Get last 5 non-null results (or fewer if not enough matches)
                last_results = [x for x in series.iloc[start_idx:i+1] if pd.notna(x)]
                form_str.append(''.join(last_results[-5:]))  # Take at most last 5
            return form_str
        
        team_matches['form'] = calculate_form(team_matches['form_letter'])
        
        standings.append(team_matches)
    
    # Combine all teams and get final standings
    final_standings = pd.concat(standings)
    last_standings = final_standings.sort_values(['team_id', 'date']).groupby('team_id').last().reset_index()
    
    # Calculate league rank
    last_standings = last_standings.sort_values(
        ['cum_points', 'cum_goal_diff', 'cum_goals_for'],
        ascending=False
    )
    last_standings['rank'] = range(1, len(last_standings)+1)
    
    # Create output in exact required format
    output_standings = pd.DataFrame({
        'rank': last_standings['rank'],
        'team_id': last_standings['team_id'],
        'team_name': last_standings['team_name'],
        'team_logo': last_standings['team_logo'],
        'points': last_standings['cum_points'],
        'goals_diff': last_standings['cum_goal_diff'],
        'group': 'MAIN',
        'form': last_standings['form'].fillna(''),
        'status': 'FINISHED',
        'description': '',
        'played': last_standings['cum_wins'] + last_standings['cum_draws'] + last_standings['cum_losses'],
        'wins': last_standings['cum_wins'],
        'draws': last_standings['cum_draws'],
        'losses': last_standings['cum_losses'],
        'goals_for': last_standings['cum_goals_for'],
        'goals_against': last_standings['cum_goals_against'],
        'home_played': last_standings['cum_home_wins'] + last_standings['cum_home_draws'] + last_standings['cum_home_losses'],
        'home_wins': last_standings['cum_home_wins'],
        'home_draws': last_standings['cum_home_draws'],
        'home_losses': last_standings['cum_home_losses'],
        'home_goals_for': last_standings['cum_home_goals_for'],
        'home_goals_against': last_standings['cum_home_goals_against'],
        'away_played': last_standings['cum_away_wins'] + last_standings['cum_away_draws'] + last_standings['cum_away_losses'],
        'away_wins': last_standings['cum_away_wins'],
        'away_draws': last_standings['cum_away_draws'],
        'away_losses': last_standings['cum_away_losses'],
        'away_goals_for': last_standings['cum_away_goals_for'],
        'away_goals_against': last_standings['cum_away_goals_against']
    })
    
    # Save to CSV
    output_standings.to_csv(output_path, index=False)
    print(f"Standings successfully saved to {output_path}")
    return output_standings

#standings = generate_standings_from_fixtures(fixture_events_df=raw_fixture_events)

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