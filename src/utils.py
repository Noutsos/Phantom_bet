import os
import json
from datetime import datetime, timedelta
import pandas as pd
from sklearn.exceptions import NotFittedError
import logging

# League dictionary organized by country
LEAGUES = {
    # ==================== INTERNATIONAL COMPETITIONS ====================
    'Europe': {
        '2': {'name': 'Champions League', 'season_months': 9, 'category': 'european_elite'},
        '3': {'name': 'Europa League', 'season_months': 9, 'category': 'european_elite'},
        '848': {'name': 'Conference League', 'season_months': 9, 'category': 'european_elite'}
    },
    
    # ==================== TOP 5 EUROPEAN LEAGUES ====================
    'Top 5': {
        'Spain': {
            '140': {'name': 'La Liga', 'season_months': 10, 'category': 'top_tier'},
            '141': {'name': 'Segunda Division', 'season_months': 11, 'category': 'second_tier'},
            '143': {'name': 'Copa del Rey', 'season_months': 7, 'start_month': 10, 'category': 'domestic_cup'},
        },    
        'England': {
            '39': {'name': 'Premier League', 'season_months': 10, 'category': 'top_tier'},
            '40': {'name': 'Championship', 'season_months': 10, 'category': 'second_tier'},
            '41': {'name': 'League One', 'season_months': 10, 'category': 'third_tier'},
            '42': {'name': 'League Two', 'season_months': 10, 'category': 'fourth_tier'},
            '45': {'name': 'FA Cup', 'season_months': 7, 'start_month': 11, 'category': 'domestic_cup'},
            '46': {'name': 'EFL Trophy', 'season_months': 10, 'category': 'league_cup'},
            '47': {'name': 'FA Trophy', 'season_months': 10, 'category': 'lower_league_cup'},
        },
        # Italy
        'Italy': {
            '135': {'name': 'Serie A', 'season_months': 10, 'category': 'top_tier'},
            '136': {'name': 'Serie B', 'season_months': 10, 'category': 'second_tier'},
            '138': {'name': 'Serie C', 'season_months': 10, 'category': 'third_tier'},
            '137': {'name': 'Coppa Italia', 'season_months': 10, 'category': 'domestic_cup'},
        },
        # Germany
        'Germany': {
            '78': {'name': 'Bundesliga', 'season_months': 10, 'category': 'top_tier'},
            '79': {'name': '2. Bundesliga', 'season_months': 10, 'category': 'second_tier'},
            '80': {'name': '3. Liga', 'season_months': 10, 'category': 'third_tier'},
            '81': {'name': 'DFB Pokal', 'season_months': 10, 'category': 'domestic_cup'},
        },
        # France
        'France': {
            '61': {'name': 'Ligue 1', 'season_months': 10, 'category': 'top_tier'},
            '62': {'name': 'Ligue 2', 'season_months': 10, 'category': 'second_tier'},
            '63': {'name': 'National', 'season_months': 10, 'category': 'third_tier'},
            '66': {'name': 'Coupe de France', 'season_months': 12, 'start_month': 5, 'category': 'domestic_cup'}
        },
    },
    # ==================== OTHER WESTERN EUROPE ====================
    'Western Europe': {
        'Netherlands': {
            '88': {'name': 'Eredivisie', 'season_months': 10, 'category': 'top_tier'},
            '89': {'name': 'Eerste Divisie', 'season_months': 10, 'category': 'second_tier'},
            '90': {'name': 'KNVB Beker', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Portugal': {
            '94': {'name': 'Primeira Liga', 'season_months': 10, 'category': 'top_tier'},
            '95': {'name': 'Segunda Liga', 'season_months': 10, 'category': 'second_tier'},
            '97': {'name': 'Taca de Portugal', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Belgium': {
            '144': {'name': 'Jupiler Pro League', 'season_months': 11, 'start_month': 7, 'category': 'top_tier'},
            '145': {'name': 'Challenger Pro League', 'season_months': 10, 'category': 'second_tier'},
            '147': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Switzerland': {
            '207': {'name': 'Super League', 'season_months': 11, 'start_month': 7, 'category': 'top_tier'},
            '208': {'name': 'Challenge League', 'season_months': 10, 'category': 'second_tier'},
            '209': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Austria': {
            '218': {'name': 'Bundesliga', 'season_months': 10, 'category': 'top_tier'},
            '219': {'name': '2. Liga', 'season_months': 10, 'category': 'second_tier'},
            '220': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Scotland': {
            '179': {'name': 'Premiership', 'season_months': 10, 'category': 'top_tier'},
            '180': {'name': 'Championship', 'season_months': 10, 'category': 'second_tier'},
            '183': {'name': 'League One', 'season_months': 10, 'category': 'third_tier'},
            '184': {'name': 'League Two', 'season_months': 10, 'category': 'fourth_tier'},
            '181': {'name': 'FA Cup', 'season_months': 8, 'category': 'domestic_cup'},
            '182': {'name': 'Challenge Cup', 'season_months': 8, 'category': 'league_cup'},
            '185': {'name': 'League Cup', 'season_months': 8, 'category': 'league_cup'},
        },
        'Ireland': {
            '357': {'name': 'Premier Division', 'season_months': 9, 'start_month': 2, 'category': 'top_tier'},
            '358': {'name': 'First Division', 'season_months': 10, 'category': 'second_tier'},
            '359': {'name': 'FAI Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Wales': {
            '110': {'name': 'Premier League', 'season_months': 10, 'category': 'top_tier'},
            '111': {'name': 'FAW Championship', 'season_months': 10, 'category': 'second_tier'},
            '112': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
    },
    
    # ==================== EASTERN EUROPE ====================
    'Eastern Europe': {
        'Turkey': {
            '203': {'name': 'Super Lig', 'season_months': 10, 'start_month': 8, 'category': 'top_tier'},
            '204': {'name': '1. Lig', 'season_months': 10, 'category': 'second_tier'},
            '205': {'name': '2. Lig', 'season_months': 10, 'category': 'third_tier'},
            '206': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Russia': {
            '235': {'name': 'Premier League', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '236': {'name': 'First League', 'season_months': 10, 'category': 'second_tier'},
            '237': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Ukraine': {
            '333': {'name': 'Premier League', 'season_months': 10, 'start_month': 8,'category': 'top_tier'},
            '334': {'name': 'Persha Liga', 'season_months': 10, 'category': 'second_tier'},
            '335': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Poland': {
            '106': {'name': 'Ekstraklasa', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '107': {'name': 'I Liga', 'season_months': 10, 'category': 'second_tier'},
            '108': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Czech Republic': {
            '345': {'name': 'Czech Liga', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '346': {'name': 'FNL', 'season_months': 10, 'category': 'second_tier'},
            '347': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Romania': {
            '283': {'name': 'Liga I', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '284': {'name': 'Liga II', 'season_months': 10, 'category': 'second_tier'},
            '285': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Greece': {
            '197': {'name': 'Super League', 'season_months': 10, 'start_month': 8,'category': 'top_tier'},
            '198': {'name': 'Football League', 'season_months': 10, 'category': 'second_tier'},
            '199': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Hungary': {
            '271': {'name': 'NB I', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '272': {'name': 'NB II', 'season_months': 10, 'category': 'second_tier'},
            '273': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Serbia': {
            '286': {'name': 'Super Liga', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '287': {'name': 'Prva Liga', 'season_months': 10, 'category': 'second_tier'},
            '732': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Croatia': {
            '210': {'name': 'HNL', 'season_months': 10, 'start_month': 8,'category': 'top_tier'},
            '211': {'name': 'First NL', 'season_months': 10, 'category': 'second_tier'},
            '212': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Bulgaria': {
            '172': {'name': 'First League', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '173': {'name': 'Second League', 'season_months': 10, 'category': 'second_tier'},
            '174': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Slovakia': {
            '332': {'name': 'Fortuna Liga', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '506': {'name': '2. liga', 'season_months': 10, 'category': 'second_tier'},
            '680': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Slovenia': {
            '373': {'name': '1. SNL', 'season_months': 11, 'start_month': 7,'category': 'top_tier'},
            '374': {'name': '2. SNL', 'season_months': 10, 'category': 'second_tier'},
            '375': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
    },
    
    # ==================== BALTIC & NORDIC ====================
    'Baltic & Nordic': {
        'Sweden': {
        '113': {'name': 'Allsvenskan', 'season_months': 8, 'start_month': 3, 'category': 'top_tier'},
        '114': {'name': 'Superettan', 'season_months': 8, 'start_month': 3, 'category': 'second_tier'},
        '115': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Norway': {
        '103': {'name': 'Eliteserien', 'season_months': 8, 'start_month': 3, 'category': 'top_tier'},
        '104': {'name': '1. Divisjon', 'season_months': 8, 'start_month': 3, 'category': 'second_tier'},
        '105': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Denmark': {
        '119': {'name': 'Superliga', 'season_months': 8, 'start_month': 3, 'category': 'top_tier'},
        '120': {'name': '1. Division', 'season_months': 8, 'start_month': 3, 'category': 'second_tier'},
        '121': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Finland': {
        '244': {'name': 'Veikkausliiga', 'season_months': 8, 'start_month': 4, 'category': 'top_tier'},
        '245': {'name': 'Ykkonen', 'season_months': 8, 'start_month': 4, 'category': 'second_tier'},
        '246': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Estonia': {
        '328': {'name': 'Meistriliiga', 'season_months': 8, 'start_month': 4, 'category': 'top_tier'},
        '329': {'name': 'Esiliiga A', 'season_months': 8, 'start_month': 4, 'category': 'second_tier'},
        '657': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Latvia': {
        '364': {'name': 'Virsliga', 'season_months': 8, 'start_month': 4, 'category': 'top_tier'},
        '365': {'name': '1. Liga', 'season_months': 8, 'start_month': 4, 'category': 'second_tier'},
        '658': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Lithuania': {
        '361': {'name': 'A Lyga', 'season_months': 8, 'start_month': 3, 'category': 'top_tier'},
        '362': {'name': '1 Lyga', 'season_months': 8, 'start_month': 3, 'category': 'second_tier'},
        '661': {'name': 'Cup', 'season_months': 8, 'start_month': 2, 'category': 'domestic_cup'},
        },
        'Iceland': {
        '164': {'name': 'Úrvalsdeild', 'season_months': 8, 'start_month': 5, 'category': 'top_tier'},
        '165': {'name': '1. Deild', 'season_months': 8, 'start_month': 5, 'category': 'second_tier'},
        '166': {'name': '2. Deild', 'season_months': 8, 'start_month': 5, 'category': 'third_tier'},
        '167': {'name': 'Cup', 'season_months': 8, 'start_month': 4, 'category': 'domestic_cup'},
        '168': {'name': 'League Cup', 'season_months': 8, 'start_month': 4, 'category': 'league_cup'}
        },
    },
    
    # ==================== BALKANS & SMALLER EUROPEAN ====================
    'Balkans & Smaller Europe': {
        'Bosnia and Herzegovina': {
        '315': {'name': 'Premijer Liga', 'season_months': 11, 'start_month': 7, 'category': 'top_tier'},
        '316': {'name': '1st League - FBiH', 'season_months': 10, 'category': 'second_tier'},
        '317': {'name': '1st League - RS', 'season_months': 10, 'category': 'second_tier'},
        '314': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Albania': {
        '310': {'name': 'Superliga', 'season_months': 10, 'category': 'top_tier'},
        '311': {'name': '1st Division', 'season_months': 10, 'category': 'second_tier'},
        '707': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Skopja': {
        '371': {'name': 'First League', 'season_months': 10, 'category': 'top_tier'},
        '372': {'name': 'Second League', 'season_months': 10, 'category': 'second_tier'},
        '756': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Montenegro': {
        '355': {'name': 'First League', 'season_months': 10, 'category': 'top_tier'},
        '356': {'name': 'Second League', 'season_months': 10, 'category': 'second_tier'},
        '723': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Kosovo': {
        '664': {'name': 'Superliga', 'season_months': 10, 'category': 'top_tier'},
        '665': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Cyprus': {
        '318': {'name': 'First Division', 'season_months': 10, 'category': 'top_tier'},
        '319': {'name': 'Second Division', 'season_months': 10, 'category': 'second_tier'},
        '320': {'name': 'Third Division', 'season_months': 10, 'category': 'third_tier'},
        '321': {'na},me': 'Cup', 'season_months': 8, 'category': 'domestic_cup'},
        },
        'Luxembourg': {
            '261': {'name': 'National Division', 'season_months': 10, 'category': 'top_tier'},
            '721': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Andorra': {
            '312': {'name': 'Primera Divisio', 'season_months': 9,'start_month': 9, 'category': 'top_tier'},
            '313': {'name': 'Segona Divisio', 'season_months': 10, 'category': 'second_tier'},
            '655': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Gibraltar': {
            '758': {'name': 'Premier Division', 'season_months': 10, 'category': 'top_tier'},
            '837': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Georgia': {
            '327': {'name': 'Erovnuli Liga', 'season_months': 10, 'category': 'top_tier'},
            '326': {'name': 'Erovnuli Liga 2', 'season_months': 10, 'category': 'second_tier'},
            '672': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Armenia': {
            '342': {'name': 'Premier League', 'season_months': 10, 'category': 'top_tier'},
            '343': {'name': 'First League', 'season_months': 10, 'category': 'second_tier'},
            '709': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Azerbaijan': {
            '419': {'name': 'Premyer Liqa', 'season_months': 10, 'category': 'top_tier'},
            '418': {'name': 'Birinci Dasta', 'season_months': 10, 'category': 'second_tier'},
            '420': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        }
    },
    
    # ==================== ASIA ====================
    'East Asia': {
        'Japan': {
            '98': {'name': 'J1 League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '99': {'name': 'J2 League', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            '100': {'name': 'J3 League', 'season_months': 8, 'start_month': 2, 'category': 'third_tier'},
            '101': {'name': 'J-League Cup', 'season_months': 8, 'start_month': 5, 'category': 'league_cup'},
            '102': {'name': 'Emperor Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'South Korea': {
            '292': {'name': 'K League 1', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '293': {'name': 'K League 2', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            '294': {'name': 'FA Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'China': {
            '169': {'name': 'Super League', 'season_months': 8, 'start_month': 3, 'category': 'top_tier'},
            '170': {'name': 'League One', 'season_months': 8, 'start_month': 3, 'category': 'second_tier'},
            '171': {'name': 'FA Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Macau': {
            '589': {'name': 'Liga de Elite', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
        },
        'Mongolia': {
            '590': {'name': 'National Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
        },
        'Taiwan': {
            '591': {'name': 'Taiwan Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
        },
        'Hong Kong': {
            '592': {'name': 'Hong Kong Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
        }
    },
    'Middle East': {   
        'Saudi Arabia': {
            '307': {'name': 'Pro League', 'season_months': 8, 'start_month': 8, 'category': 'top_tier'},
            '308': {'name': 'First Division', 'season_months': 8, 'start_month': 8, 'category': 'second_tier'},
            '504': {'name': 'King Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Qatar': {
            '305': {'name': 'Stars League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
            '306': {'name': 'Second Division', 'season_months': 8, 'start_month': 9, 'category': 'second_tier'},
            '824': {'name': 'Emir Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'},
            '825': {'name': 'Qatar Cup', 'season_months': 8, 'start_month': 7, 'category': 'league_cup'}
        },
        'UAE': {
            '301': {'name': 'Pro League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
            '303': {'name': 'First Division', 'season_months': 8, 'start_month': 9, 'category': 'second_tier'},
            '302': {'name': 'League Cup', 'season_months': 8, 'start_month': 7, 'category': 'league_cup'},
            '560': {'name': 'Presidents Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Iran': {
            '290': {'name': 'Persian Gulf Pro League', 'season_months': 8, 'start_month': 8, 'category': 'top_tier'},
            '291': {'name': 'Azadegan League', 'season_months': 8, 'start_month': 8, 'category': 'second_tier'},
            '495': {'name': 'Hazfi Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Iraq': {
            '542': {'name': 'Iraqi Premier League', 'season_months': 8, 'start_month': 8, 'category': 'top_tier'},
        },
        'Kuwait': {
            '339': {'name': 'Kuwait Premier League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
            '331': {'name': 'Kuwait First Division', 'season_months': 8, 'start_month': 9, 'category': 'second_tier'},
            '720': {'name': 'Emir Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Oman': {
            '406': {'name': 'Oman Professional League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
            '726': {'name': 'Sultan Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Syria': {
            '425': {'name': 'Syrian Premier League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
        },
        'Jordan': {
            '387': {'name': 'Jordanian Pro League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
            '863': {'name': 'Jordan Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Egypt': {
            '233': {'name': 'Egyptian Premier League', 'season_months': 8, 'start_month': 9, 'category': 'top_tier'},
            '887': {'name': 'Egyptian Second Division', 'season_months': 8, 'start_month': 9, 'category': 'second_tier'},
            '714': {'name': 'Egyptian Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        }
    },
    'Southeast Asia': {   
        'Thailand': {
            '296': {'name': 'Thai League 1', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '297': {'name': 'Thai League 2', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            '298': {'name': 'FA Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Malaysia': {
            '278': {'name': 'Super League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '279': {'name': 'Premier League', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            '500': {'name': 'FA Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Indonesia': {
            '274': {'name': 'Liga 1', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '275': {'name': 'Liga 2', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            '924': {'name': 'Piala Indonesia Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Vietnam': {
            '276': {'name': 'V.League 1', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '277': {'name': 'V.League 2', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            '499': {'name': 'Vietnam National Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Philippines': {
            '765': {'name': 'Philippine Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '867': {'name': 'PFF National Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Singapore': {
            '368': {'name': 'Singapore Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '959': {'name': 'Singapore Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Cambodia': {
            '410': {'name': 'Cambodian League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '1174': {'name': 'Hun Sen Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Laos': {
            '647': {'name': 'Lao Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
        },
        'Myanmar': {
            '588': {'name': 'National League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
        },
    },
    'South Asia': {    
        'India': {
            '323': {'name': 'Indian Super League', 'season_months': 8, 'start_month': 10, 'category': 'top_tier'},
            '324': {'name': 'I-League', 'season_months': 8, 'start_month': 10, 'category': 'second_tier'},
            '325': {'name': 'Santosh Trophy', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Bangladesh': {
            '398': {'name': 'Bangladesh Premier League', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '811': {'name': 'Federation Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
    },
    'Central Asia': {
        'Kazakhstan': {
            '389': {'name': 'Premier League', 'season_months': 10, 'category': 'top_tier'},
            '388': {'name': '1. Division', 'season_months': 10, 'category': 'second_tier'},
            '498': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Uzbekistan': {
            '369': {'name': 'Super League', 'season_months': 10, 'category': 'top_tier'},
            '802': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Kyrgyzstan': {
            '396': {'name': 'Top League', 'season_months': 10, 'category': 'top_tier'},
            '397': {'name': 'First League', 'season_months': 10, 'category': 'second_tier'},
            '398': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
    },
    
    # ==================== OTHER REGIONS ====================
    'Other Regions': {
        'South Africa': {
            '288': {'name': 'Premier Soccer League', 'season_months': 8, 'start_month': 8, 'category': 'top_tier'},
            '289': {'name': 'First Division', 'season_months': 8, 'start_month': 8, 'category': 'second_tier'},
            '507': {'name': 'Cup', 'season_months': 8, 'start_month': 7, 'category': 'domestic_cup'}
        },
        'Belarus': {
            '116': {'name': 'Premier League', 'season_months': 10, 'category': 'top_tier'},
            '117': {'name': '1. Division', 'season_months': 10, 'category': 'second_tier'},
            '118': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },

        'Moldova': {
            '394': {'name': 'Super Liga', 'season_months': 10, 'category': 'top_tier'},
            '395': {'name': 'Liga 1', 'season_months': 10, 'category': 'second_tier'},
            '674': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        },
        'Israel': {
            '383': {'name': 'Ligat Haal', 'season_months': 10, 'category': 'top_tier'},
            '382': {'name': 'Liga Leumit', 'season_months': 10, 'category': 'second_tier'},
            '384': {'name': 'Cup', 'season_months': 8, 'category': 'domestic_cup'}
        }
    },
    'North America': {
        'USA': {
            '157': {'name': 'Major League Soccer', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            '158': {'name': 'USL Championship', 'season_months': 8, 'start_month': 3, 'category': 'second_tier'},
            '159': {'name': 'USL League One', 'season_months': 8, 'start_month': 4, 'category': 'third_tier'},
            '160': {'name': 'US Open Cup', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Canada': {
            '161': {'name': 'Canadian Premier League', 'season_months': 8, 'start_month': 4, 'category': 'top_tier'},
            '162': {'name': 'Canadian Championship', 'season_months': 8, 'start_month': 5, 'category': 'domestic_cup'}
        },
        'Mexico': {
            '155': {'name': 'Liga MX', 'season_months': 12, 'start_month': 7, 'category': 'top_tier'},
            '156': {'name': 'Ascenso MX', 'season_months': 12, 'start_month': 7, 'category': 'second_tier'},
            '561': {'name': 'Copa MX', 'season_months': 12, 'start_month': 7, 'category': 'domestic_cup'}
        },
    },
    'South America': {
        'Brazil': {
            #'137': {'name': 'Serie A', 'season_months': 8, 'start_month': 5, 'category': 'top_tier'},
            #'138': {'name': 'Serie B', 'season_months': 8, 'start_month': 5, 'category': 'second_tier'},
            #'139': {'name': 'Copa do Brasil', 'season_months': 8, 'start_month': 3, 'category': 'domestic_cup'}
        },
        'Argentina': {
           # '128': {'name': 'Primera Division', 'season_months': 8, 'start_month': 8, 'category': 'top_tier'},
            #'129': {'name': 'Primera Nacional', 'season_months': 8, 'start_month': 8, 'category': 'second_tier'},
            #'130': {'name': 'Copa Argentina', 'season_months': 8, 'start_month': 3, 'category': 'domestic_cup'}
        },
        'Chile': {
            #'143': {'name': 'Primera Division', 'season_months': 8, 'start_month': 2, 'category': 'top_tier'},
            #'144': {'name': 'Primera B', 'season_months': 8, 'start_month': 2, 'category': 'second_tier'},
            #'145': {'name': 'Copa Chile', 'season_months': 8, 'start_month': 3, 'category': 'domestic_cup'}
        },
        'Colombia': {
            #'134': {'name': 'Categoria Primera A', 'season_months': 8, 'start_month': 1, 'category': 'top_tier'},
            #'135': {'name': 'Categoria Primera B', 'season_months': 8, 'start_month': 1, 'category': 'second_tier'},
            #'136' : {'name' : '', season_months: , start_month: , category: ''} # No domestic cup
        },
        # Additional South American leagues can be added here
    },
}


# ==================== FEATURE NAMES EXTRACTION ====================
def get_feature_names_from_pipeline(model):
    """
    Extract feature names from sklearn pipeline
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained pipeline model
        
    Returns:
    --------
    list : Feature names
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Handle preprocessor
        if 'preprocessor' not in model.named_steps:
            logger.warning("No preprocessor found in pipeline")
            return []
        
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        
        # Numerical features
        if 'num' in preprocessor.named_transformers_:
            num_transformer = preprocessor.named_transformers_['num']
            if hasattr(num_transformer, 'get_feature_names_out'):
                num_features = num_transformer.get_feature_names_out()
                feature_names.extend(num_features)
            else:
                # Fallback: use original column names
                num_cols = preprocessor.transformers_[0][2]  # Get column indices for numerical features
                feature_names.extend(num_cols)
        
        # Categorical features
        if 'cat' in preprocessor.named_transformers_:
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                try:
                    cat_features = cat_transformer.get_feature_names_out()
                    feature_names.extend(cat_features)
                except NotFittedError:
                    logger.warning("OneHotEncoder not fitted yet")
            else:
                # Fallback: use original column names with prefix
                cat_cols = preprocessor.transformers_[1][2]  # Get column indices for categorical features
                for col in cat_cols:
                    # Get unique values to create dummy names
                    unique_vals = model.named_steps['preprocessor'].named_transformers_['cat'].categories_[
                        preprocessor.transformers_[1][2].index(col)
                    ]
                    for val in unique_vals:
                        feature_names.append(f"{col}_{val}")
        
        # Remainder features
        if hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if name == 'remainder' and trans == 'passthrough':
                    feature_names.extend(cols)
        
        # Handle variance threshold
        if 'variance_threshold' in model.named_steps:
            vt = model.named_steps['variance_threshold']
            if hasattr(vt, 'get_support'):
                mask = vt.get_support()
                if len(mask) == len(feature_names):
                    feature_names = [f for f, m in zip(feature_names, mask) if m]
        
        # Handle feature selection
        if 'feature_selection' in model.named_steps:
            selector = model.named_steps['feature_selection']
            
            if hasattr(selector, 'get_support'):
                # For SelectFromModel, RFE, SelectKBest
                selected_indices = selector.get_support()
                if len(selected_indices) == len(feature_names):
                    feature_names = [f for f, m in zip(feature_names, selected_indices) if m]
            elif hasattr(selector, 'components_'):
                # For PCA
                feature_names = [f'PC_{i+1}' for i in range(selector.n_components_)]
        
        logger.info(f"Extracted {len(feature_names)} feature names from pipeline")
        return feature_names
        
    except Exception as e:
        logger.error(f"Error extracting feature names: {str(e)}", exc_info=True)
        return []

""""
Collapse All	Ctrl + K then Ctrl + 0	
Expand All	Ctrl + K then Ctrl + J	
Toggle Fold (current section)	Ctrl + Shift + [	
Toggle Unfold (current section)	Ctrl + Shift + ]

python -m src.main --phase 1 --collection-phase 1 --filter-ids --season 2024
python -m src.main --phase 2 
python -m src.main --phase 1 --collection-phase 1 --filter-categories top 5 --filter-tiers second_tier --season 2018
python app.py
http://localhost:5000

            # Progress logging
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_matches:
                progress = (idx + 1) / total_matches * 100
                self.logger.info(f"Standings progress: {progress:.1f}% ({idx + 1}/{total_matches})")
"""