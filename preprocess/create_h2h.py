import pandas as pd
from tqdm import tqdm  # For progress tracking (optional)

def calculate_h2h_features(df, recent_matches_window=5):
    """
    Calculate Head-to-Head (H2H) features for each fixture.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical match data.
        recent_matches_window (int): Number of recent matches to consider for recent form (default: 5).
    
    Returns:
        pd.DataFrame: DataFrame with added H2H features.
    """
    # Ensure date is in datetime format and sort
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Initialize new H2H columns
    h2h_features = [
        'h2h_matches', 
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'h2h_home_goals', 'h2h_away_goals', 'h2h_goal_diff',
        'h2h_home_win_pct', 'h2h_away_win_pct',
        f'h2h_recent_home_wins_last{recent_matches_window}',
        f'h2h_recent_away_wins_last{recent_matches_window}',
        f'h2h_recent_draws_last{recent_matches_window}',
        f'h2h_recent_avg_goals_last{recent_matches_window}',
        'h2h_streak'  # Current unbeaten streak (e.g., "home_team_unbeaten_in_3")
    ]
    
    for col in h2h_features:
        df[col] = 0 if col != 'h2h_streak' else ""
    
    # Use tqdm for progress tracking (optional)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating H2H features"):
        home_id, away_id, date = row['home_team_id'], row['away_team_id'], row['date']
        
        # Get all past matches between these two teams (regardless of home/away)
        past_matches = df[
            ((df['home_team_id'] == home_id) & (df['away_team_id'] == away_id) |
            ((df['home_team_id'] == away_id) & (df['away_team_id'] == home_id)))
        ].query('date < @date')
        
        # Skip if no past matches
        if len(past_matches) == 0:
            continue
        
        # --- Total H2H Stats ---
        df.at[idx, 'h2h_matches'] = len(past_matches)
        
        # Home wins (current home team won in any past H2H)
        home_wins = len(past_matches[
            ((past_matches['home_team_id'] == home_id) & (past_matches['home_winner'] == True)) |
            ((past_matches['away_team_id'] == home_id) & (past_matches['away_winner'] == True))
        ])
        df.at[idx, 'h2h_home_wins'] = home_wins
        
        # Away wins (current away team won in any past H2H)
        away_wins = len(past_matches[
            ((past_matches['home_team_id'] == away_id) & (past_matches['home_winner'] == True)) |
            ((past_matches['away_team_id'] == away_id) & (past_matches['away_winner'] == True))
        ])
        df.at[idx, 'h2h_away_wins'] = away_wins
        
        # Draws
        draws = len(past_matches[past_matches['home_winner'].isna()])
        df.at[idx, 'h2h_draws'] = draws
        
        # Win percentages
        df.at[idx, 'h2h_home_win_pct'] = home_wins / len(past_matches) if len(past_matches) > 0 else 0
        df.at[idx, 'h2h_away_win_pct'] = away_wins / len(past_matches) if len(past_matches) > 0 else 0
        
        # --- Goals ---
        # Home goals (goals scored by current home team in past H2H)
        home_goals = past_matches.apply(
            lambda x: x['home_goals'] if x['home_team_id'] == home_id else x['away_goals'], axis=1
        ).mean()
        df.at[idx, 'h2h_home_goals'] = home_goals
        
        # Away goals (goals scored by current away team in past H2H)
        away_goals = past_matches.apply(
            lambda x: x['away_goals'] if x['home_team_id'] == home_id else x['home_goals'], axis=1
        ).mean()
        df.at[idx, 'h2h_away_goals'] = away_goals
        
        # Goal difference
        df.at[idx, 'h2h_goal_diff'] = home_goals - away_goals
        
        # --- Recent Form (Last N Matches) ---
        recent_matches = past_matches.tail(recent_matches_window)
        if len(recent_matches) > 0:
            # Recent home wins
            recent_home_wins = len(recent_matches[
                ((recent_matches['home_team_id'] == home_id) & (recent_matches['home_winner'] == True)) |
                ((recent_matches['away_team_id'] == home_id) & (recent_matches['away_winner'] == True))
            ])
            df.at[idx, f'h2h_recent_home_wins_last{recent_matches_window}'] = recent_home_wins
            
            # Recent away wins
            recent_away_wins = len(recent_matches[
                ((recent_matches['home_team_id'] == away_id) & (recent_matches['home_winner'] == True)) |
                ((recent_matches['away_team_id'] == away_id) & (recent_matches['away_winner'] == True))
            ])
            df.at[idx, f'h2h_recent_away_wins_last{recent_matches_window}'] = recent_away_wins
            
            # Recent draws
            recent_draws = len(recent_matches[recent_matches['home_winner'].isna()])
            df.at[idx, f'h2h_recent_draws_last{recent_matches_window}'] = recent_draws
            
            # Recent avg goals
            df.at[idx, f'h2h_recent_avg_goals_last{recent_matches_window}'] = (
                recent_matches['home_goals'].sum() + recent_matches['away_goals'].sum()
            ) / len(recent_matches)
        
        # --- Streak Calculation ---
        # Check if home team is unbeaten in last N matches
        streak_matches = recent_matches.copy()
        if len(streak_matches) > 0:
            home_unbeaten = all(
                ((m['home_team_id'] == home_id) & (m['home_winner'] != False)) |
                ((m['away_team_id'] == home_id) & (m['away_winner'] != False))
                for _, m in streak_matches.iterrows()
            )
            if home_unbeaten:
                df.at[idx, 'h2h_streak'] = f"home_unbeaten_in_{len(streak_matches)}"
            else:
                away_unbeaten = all(
                    ((m['home_team_id'] == away_id) & (m['home_winner'] != False)) |
                    ((m['away_team_id'] == away_id) & (m['away_winner'] != False))
                    for _, m in streak_matches.iterrows()
                )
                if away_unbeaten:
                    df.at[idx, 'h2h_streak'] = f"away_unbeaten_in_{len(streak_matches)}"
    
    return df

# --- Usage Example ---
# df = pd.read_csv('your_dataset.csv')
# df_with_h2h = calculate_h2h_features(df, recent_matches_window=5)
# df_with_h2h.to_csv('enhanced_features_with_h2h.csv', index=False)