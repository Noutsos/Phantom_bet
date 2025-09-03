import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

class FootballDataValidator:
    def __init__(self, base_path: str = "data/extracted"):
        self.base_path = Path(base_path)
        self.results_path = Path("validation_results")
        self.results_path.mkdir(exist_ok=True)
        
    def analyze_leagues(self):
        """Main analysis function"""
        for league_dir in self.base_path.iterdir():
            if league_dir.is_dir():
                self.analyze_league(league_dir.name)

    def analyze_league(self, league: str):
        """Analyze a single league across all years"""
        league_path = self.base_path / league
        years = sorted([d.name for d in league_path.iterdir() if d.is_dir()])
        
        # Collect all data for this league
        null_reports = []
        column_tracker = defaultdict(dict)
        
        for year in years:
            year_path = league_path / year
            for csv_file in year_path.glob("*.csv"):
                file_type = csv_file.stem  # e.g., 'fixtures', 'stats'
                df = pd.read_csv(csv_file)
                
                # Null value analysis
                null_counts = df.isnull().sum()
                null_report = pd.DataFrame({
                    'league': league,
                    'year': year,
                    'file_type': file_type,
                    'column': null_counts.index,
                    'null_count': null_counts.values,
                    'null_percentage': (null_counts / len(df)) * 100
                })
                null_reports.append(null_report)
                
                # Track columns
                column_tracker[file_type][year] = set(df.columns)
        
        # Save null reports
        if null_reports:
            null_df = pd.concat(null_reports)
            null_df.to_csv(self.results_path / f"{league}_null_report.csv", index=False)
        
        # Generate column comparison
        self.compare_columns(league, column_tracker)

    def compare_columns(self, league: str, column_tracker: Dict[str, Dict[str, set]]):
        """Compare columns across years for each file type"""
        comparison_reports = []
        
        for file_type, year_data in column_tracker.items():
            all_years = sorted(year_data.keys())
            all_columns = set().union(*year_data.values())
            
            # Create presence matrix
            presence_df = pd.DataFrame(index=sorted(all_columns), columns=all_years)
            for year, columns in year_data.items():
                presence_df[year] = presence_df.index.isin(columns)
            
            # Identify changes
            for col in all_columns:
                present_years = [year for year in all_years if col in year_data[year]]
                missing_years = [year for year in all_years if year not in present_years]
                
                if missing_years:
                    comparison_reports.append({
                        'league': league,
                        'file_type': file_type,
                        'column': col,
                        'present_years': ', '.join(present_years),
                        'missing_years': ', '.join(missing_years),
                        'status': 'Added' if len(present_years) == 1 else 
                                  'Removed' if present_years and present_years[-1] != all_years[-1] else
                                  'Inconsistent'
                    })
        
        # Save comparison report
        if comparison_reports:
            comp_df = pd.DataFrame(comparison_reports)
            comp_df.to_csv(self.results_path / f"{league}_column_comparison.csv", index=False)

if __name__ == "__main__":
    validator = FootballDataValidator(base_path="data/extracted")
    validator.analyze_leagues()
    print("Analysis complete! Check the validation_results folder.")