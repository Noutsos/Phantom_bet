import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

df = pd.read_csv('data/final_processed.csv')

serieA_2023 = df[(df['league_id'] == 135)
                 ]
chat_columns = [ 'fixture_id', 'round', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_points','away_points','home_goals_for','away_goals_for','home_goals_against','away_goals_against','home_played','away_played','home_wins','away_wins','home_draws','away_draws'
                ,'home_losses','away_losses','home_goals_diff','away_goals_diff','home_rank','away_rank']
#chat_df = serieA_2023[chat_columns]


chat_columns_2 = [ 'fixture_id','date',  'home_team', 'away_team','home_goals', 'away_goals', 'h2h_matches','h2h_home_wins','h2h_away_wins','h2h_draws','h2h_home_goals','h2h_away_goals','h2h_goal_diff','h2h_home_win_pct','h2h_away_win_pct','h2h_avg_goals',
                  'h2h_recent_home_wins_last5','h2h_recent_away_wins_last5','h2h_recent_draws_last5','h2h_recent_avg_goals_last5',
                 'h2h_streak','h2h_league_matches','h2h_cup_matches','h2h_cup_home_wins','h2h_cup_away_wins','h2h_same_country','h2h_win_streak','h2h_loss_streak']

#chat_df_2 = serieA_2023[chat_columns_2]
#chat_df_2.to_csv('data/serieA_2023_chat_2.csv', index=False)


chat_columns_3 = ['fixture_id', 'date', 'league_id', 'season','home_team','away_team', 'home_form_strength', 'away_form_strength', 'home_momentum', 'away_momentum']
chat_df_3 = serieA_2023[chat_columns_3]
chat_df_3.to_csv('data/serieA_2023_chat_3.csv', index=False)