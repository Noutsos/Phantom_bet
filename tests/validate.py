import pandas as pd
from typing import List, Dict, Set

def validate_fixture_rows_match(df1: pd.DataFrame, 
                               df2: pd.DataFrame, 
                               fixture_ids: List[int],
                               columns_to_check: List[str]) -> Dict:
    """
    Validate if specific rows with the same fixture IDs have matching values between two DataFrames.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        First DataFrame for comparison
    df2 : pd.DataFrame
        Second DataFrame for comparison  
    fixture_ids : List[int]
        List of fixture IDs to validate
    columns_to_check : List[str]
        List of column names to check for matching values
    
    Returns:
    --------
    Dict with validation results including:
    - matches: Boolean indicating if all specified fixture IDs match
    - details: Detailed information for each fixture ID
    - stats: Statistics about the comparison
    """
    
    results = {
        'all_match': False,
        'details': {},
        'stats': {
            'total_fixtures_checked': len(fixture_ids),
            'fixtures_found_in_both': 0,
            'fixtures_matching': 0,
            'fixtures_not_found_df1': [],
            'fixtures_not_found_df2': [],
            'columns_checked': columns_to_check
        },
        'errors': []
    }
    
    # Input validation
    if df1.empty or df2.empty:
        results['errors'].append("One or both DataFrames are empty")
        return results
    
    if 'fixture_id' not in df1.columns:
        results['errors'].append("'fixture_id' column not found in df1")
    if 'fixture_id' not in df2.columns:
        results['errors'].append("'fixture_id' column not found in df2")
    
    missing_cols_df1 = [col for col in columns_to_check if col not in df1.columns]
    missing_cols_df2 = [col for col in columns_to_check if col not in df2.columns]
    
    if missing_cols_df1:
        results['errors'].append(f"Columns missing in df1: {missing_cols_df1}")
    if missing_cols_df2:
        results['errors'].append(f"Columns missing in df2: {missing_cols_df2}")
    
    if results['errors']:
        return results
    
    # Check each fixture ID
    all_match = True
    
    for fixture_id in fixture_ids:
        # Get rows for this fixture ID from both DataFrames
        row_df1 = df1[df1['fixture_id'] == fixture_id]
        row_df2 = df2[df2['fixture_id'] == fixture_id]
        
        fixture_result = {
            'found_in_df1': not row_df1.empty,
            'found_in_df2': not row_df2.empty,
            'columns_match': {},
            'all_columns_match': False,
            'values_df1': {},
            'values_df2': {}
        }
        
        if not row_df1.empty:
            results['stats']['fixtures_found_in_both'] += 1
        else:
            results['stats']['fixtures_not_found_df1'].append(fixture_id)
            all_match = False
            fixture_result['error'] = f"Fixture ID {fixture_id} not found in df1"
            results['details'][fixture_id] = fixture_result
            continue
            
        if not row_df2.empty:
            results['stats']['fixtures_found_in_both'] += 1
        else:
            results['stats']['fixtures_not_found_df2'].append(fixture_id)
            all_match = False
            fixture_result['error'] = f"Fixture ID {fixture_id} not found in df2"
            results['details'][fixture_id] = fixture_result
            continue
        
        # Both DataFrames have this fixture ID, now check columns
        column_matches = True
        
        for col in columns_to_check:
            value_df1 = row_df1[col].iloc[0] if not row_df1.empty else None
            value_df2 = row_df2[col].iloc[0] if not row_df2.empty else None
            
            # Handle NaN values
            if pd.isna(value_df1) and pd.isna(value_df2):
                match = True
            elif pd.isna(value_df1) or pd.isna(value_df2):
                match = False
            else:
                match = value_df1 == value_df2
            
            fixture_result['columns_match'][col] = match
            fixture_result['values_df1'][col] = value_df1
            fixture_result['values_df2'][col] = value_df2
            
            if not match:
                column_matches = False
                all_match = False
        
        fixture_result['all_columns_match'] = column_matches
        
        if column_matches:
            results['stats']['fixtures_matching'] += 1
        
        results['details'][fixture_id] = fixture_result
    
    results['all_match'] = all_match
    results['stats']['match_percentage'] = (results['stats']['fixtures_matching'] / len(fixture_ids) * 100) if fixture_ids else 0
    
    return results

def print_fixture_validation_results(results: Dict):
    """Helper function to print validation results in a readable format"""
    print("=" * 60)
    print("FIXTURE VALIDATION RESULTS")
    print("=" * 60)
    
    if results['errors']:
        print("❌ ERRORS:")
        for error in results['errors']:
            print(f"   - {error}")
        print()
    
    print(f"📊 Overall Match: {'✅ YES' if results['all_match'] else '❌ NO'}")
    print(f"📈 Statistics:")
    stats = results['stats']
    print(f"   Total fixtures checked: {stats['total_fixtures_checked']}")
    print(f"   Fixtures matching: {stats['fixtures_matching']}")
    print(f"   Match percentage: {stats['match_percentage']:.1f}%")
    print(f"   Not found in df1: {len(stats['fixtures_not_found_df1'])}")
    print(f"   Not found in df2: {len(stats['fixtures_not_found_df2'])}")
    print(f"   Columns checked: {stats['columns_checked']}")
    print()
    
    print("🔍 Fixture Details:")
    for fixture_id, details in results['details'].items():
        status = "✅" if details.get('all_columns_match', False) else "❌"
        print(f"   {status} Fixture ID {fixture_id}:")
        
        if 'error' in details:
            print(f"     Error: {details['error']}")
        else:
            print(f"     Found in both: ✅")
            mismatched_cols = [col for col, match in details['columns_match'].items() if not match]
            
            if mismatched_cols:
                print(f"     Mismatched columns: {mismatched_cols}")
                for col in mismatched_cols:
                    print(f"       {col}: df1={details['values_df1'][col]}, df2={details['values_df2'][col]}")
            else:
                print(f"     All columns match: ✅")
        print()

# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.read_csv("data/final_processed_pipeline.csv")
    df1 = pd.read_csv("data/final_processed_new.csv")
    
    # Check specific fixture IDs
    fixture_ids_to_check = [78813, 636685, 872563, 1223678, 1342300]
    columns_to_check = ['away_defensive_pressure_rolling_5', 'home_defensive_efficiency_rolling_5', 'home_pressing_intensity_rolling_5', 'away_pass_accuracy_rolling_5', 'away_shot_quality_rolling_5']
    
    print("Checking fixture IDs:", fixture_ids_to_check)
    print("Checking columns:", columns_to_check)
    print()
    
    results = validate_fixture_rows_match(df, df1, fixture_ids_to_check, columns_to_check)
    print_fixture_validation_results(results)