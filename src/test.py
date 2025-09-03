import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils import LEAGUES  # Assuming LEAGUES is defined in utils.py




df = pd.read_csv('data/final_processed.csv')
print(f"There are {len(df[df['outcome']=='home_win'])} matches")
print(f"There are {len(df[df['outcome']=='away_win'])} matches")
print(f"There are {len(df[df['outcome']=='draw'])} matches")

print(f"There are odds data {df[df['odds_home_win']]}")