import pandas as pd
import os
from typing import Dict, Optional

def merge_all_data(
    fixtures_path: str,
    lineups_path: str,
    injuries_path: str,
    team_stats_path: str,
    player_stats_path: str,
    standings_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Merges all football data sources into a unified dataset.
    
    Args:
        fixtures_path: Path to fixtures CSV
        lineups_path: Path to lineups CSV
        injuries_path: Path to injuries CSV
        team_stats_path: Path to team stats CSV
        player_stats_path: Path to player stats CSV
        standings_path: Path to standings CSV (one row per team)
        output_path: Optional path to save merged data
    
    Returns:
        Merged DataFrame containing all data sources
    """

    # Load all datasets with error handling
    try:
        fixtures = pd.read_csv(fixtures_path)
        lineups = pd.read_csv(lineups_path)
        injuries = pd.read_csv(injuries_path)
        team_stats = pd.read_csv(team_stats_path)
        player_stats = pd.read_csv(player_stats_path)
        standings = pd.read_csv(standings_path)
    except Exception as e:
        raise ValueError(f"Error loading data files: {str(e)}")

    # Data Validation - Improved version
    def validate_columns(df, df_name, required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {df_name}: {missing}")

    # Validate each dataset
    validate_columns(fixtures, 'fixtures', ['fixture_id', 'date', 'home_team_id', 'away_team_id'])
    validate_columns(standings, 'standings', ['team_id', 'points'])
    
    # Convert dates to datetime
    fixtures['date'] = pd.to_datetime(fixtures['date'])

    # ======================
    # 1. MERGE CORE FIXTURES
    # ======================
    merged = fixtures.copy()

    # ======================
    # 2. MERGE STANDINGS DATA
    # ======================
    # Home team standings
    merged = pd.merge(
        merged,
        standings.add_prefix('home_standings_'),
        left_on='home_team_id',
        right_on='home_standings_team_id',
        how='left'
    )

    # Away team standings
    merged = pd.merge(
        merged,
        standings.add_prefix('away_standings_'),
        left_on='away_team_id',
        right_on='away_standings_team_id',
        how='left'
    )

    # ======================
    # 3. MERGE LINEUPS
    # ======================
    if not lineups.empty:
        # Home lineups
        home_lineups = lineups[lineups['team_id'].isin(merged['home_team_id'])]
        merged = pd.merge(
            merged,
            home_lineups.add_prefix('home_lineup_'),
            left_on=['fixture_id', 'home_team_id'],
            right_on=['home_lineup_fixture_id', 'home_lineup_team_id'],
            how='left'
        )

        # Away lineups
        away_lineups = lineups[lineups['team_id'].isin(merged['away_team_id'])]
        merged = pd.merge(
            merged,
            away_lineups.add_prefix('away_lineup_'),
            left_on=['fixture_id', 'away_team_id'],
            right_on=['away_lineup_fixture_id', 'away_lineup_team_id'],
            how='left'
        )

    # ======================
    # 4. MERGE TEAM STATS
    # ======================
    if not team_stats.empty:
        # Home team stats
        merged = pd.merge(
            merged,
            team_stats.add_prefix('home_teamstats_'),
            left_on=['fixture_id', 'home_team_id'],
            right_on=['home_teamstats_fixture_id', 'home_teamstats_team_id'],
            how='left'
        )

        # Away team stats
        merged = pd.merge(
            merged,
            team_stats.add_prefix('away_teamstats_'),
            left_on=['fixture_id', 'away_team_id'],
            right_on=['away_teamstats_fixture_id', 'away_teamstats_team_id'],
            how='left'
        )

    # ======================
    # 5. MERGE INJURIES
    # ======================
    if not injuries.empty:
        # Aggregate injuries by fixture and team
        injuries_agg = injuries.groupby(['fixture_id', 'team_id']).agg({
            'injury_count': 'sum',
            'total_severity': 'sum',
            'is_key_player': 'sum'
        }).reset_index()

        # Home injuries
        merged = pd.merge(
            merged,
            injuries_agg.add_prefix('home_injury_'),
            left_on=['fixture_id', 'home_team_id'],
            right_on=['home_injury_fixture_id', 'home_injury_team_id'],
            how='left'
        )

        # Away injuries
        merged = pd.merge(
            merged,
            injuries_agg.add_prefix('away_injury_'),
            left_on=['fixture_id', 'away_team_id'],
            right_on=['away_injury_fixture_id', 'away_injury_team_id'],
            how='left'
        )

    # ======================
    # 6. MERGE PLAYER STATS
    # ======================
    if not player_stats.empty:
        # Aggregate player stats to team level
        player_stats_agg = player_stats.groupby(['fixture_id', 'team_id']).agg({
            'games_rating': 'mean',
            'goals_total': 'sum',
            'passes_accuracy': 'mean',
            'tackles_total': 'sum'
        }).reset_index()

        # Home player stats
        merged = pd.merge(
            merged,
            player_stats_agg.add_prefix('home_player_'),
            left_on=['fixture_id', 'home_team_id'],
            right_on=['home_player_fixture_id', 'home_player_team_id'],
            how='left'
        )

        # Away player stats
        merged = pd.merge(
            merged,
            player_stats_agg.add_prefix('away_player_'),
            left_on=['fixture_id', 'away_team_id'],
            right_on=['away_player_fixture_id', 'away_player_team_id'],
            how='left'
        )

    # ======================
    # 7. FINAL PROCESSING
    # ======================
    # Clean up duplicate columns
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"✅ Successfully saved merged data to {output_path}")

    return merged

df = merge_all_data(
    fixtures_path = "data/processed/La Liga/2022/fixtures_enhanced.csv",
    lineups_path = "data/processed/La Liga/2022/lineups_enhanced.csv",
    injuries_path = "data/processed/La Liga/2022/injuries_enhanced.csv",
    team_stats_path = "data/processed/La Liga/2022/team_stats_enhanced.csv",
    player_stats_path = "data/processed/La Liga/2022/player_stats_enhanced.csv",
    standings_path = "data/processed/La Liga/2022/standings_enhanced.csv"
)

# Save final dataset
df.to_csv("data/final_merged_dataset.csv", index=False)