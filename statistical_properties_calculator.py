import pandas as pd
import numpy as np
import json
from pathlib import Path

class StatisticalPropertiesCalculator:
    def __init__(self, matches_df):
        self.matches = matches_df.copy()
        self.matches['date'] = pd.to_datetime(self.matches['date'])
    
    def calculate_team_properties(self, team):
        home_matches = self.matches[self.matches['home_team'] == team]
        away_matches = self.matches[self.matches['away_team'] == team]
        
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return None
        
        home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
        home_draws = (home_matches['home_score'] == home_matches['away_score']).sum()
        home_losses = (home_matches['home_score'] < home_matches['away_score']).sum()
        
        away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
        away_draws = (away_matches['away_score'] == away_matches['home_score']).sum()
        away_losses = (away_matches['away_score'] < away_matches['home_score']).sum()
        
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses
        
        home_goals_scored = home_matches['home_score'].sum()
        away_goals_scored = away_matches['away_score'].sum()
        home_goals_conceded = home_matches['away_score'].sum()
        away_goals_conceded = away_matches['home_score'].sum()
        
        total_goals_scored = home_goals_scored + away_goals_scored
        total_goals_conceded = home_goals_conceded + away_goals_conceded
        
        win_margin = []
        loss_margin = []
        
        for _, match in home_matches.iterrows():
            diff = match['home_score'] - match['away_score']
            if diff > 0:
                win_margin.append(diff)
            elif diff < 0:
                loss_margin.append(abs(diff))
        
        for _, match in away_matches.iterrows():
            diff = match['away_score'] - match['home_score']
            if diff > 0:
                win_margin.append(diff)
            elif diff < 0:
                loss_margin.append(abs(diff))
        
        properties = {
            'team': team,
            'total_matches': total_matches,
            'win_prob': total_wins / total_matches,
            'draw_prob': total_draws / total_matches,
            'loss_prob': total_losses / total_matches,
            'home_win_prob': home_wins / len(home_matches) if len(home_matches) > 0 else 0,
            'away_win_prob': away_wins / len(away_matches) if len(away_matches) > 0 else 0,
            'home_advantage': (home_wins / len(home_matches) if len(home_matches) > 0 else 0) - 
                             (away_wins / len(away_matches) if len(away_matches) > 0 else 0),
            'avg_goals_scored': total_goals_scored / total_matches,
            'avg_goals_conceded': total_goals_conceded / total_matches,
            'goal_difference': (total_goals_scored - total_goals_conceded) / total_matches,
            'avg_win_margin': np.mean(win_margin) if win_margin else 0,
            'avg_loss_margin': np.mean(loss_margin) if loss_margin else 0
        }
        
        return properties
    
    def calculate_all_teams(self):
        teams = pd.concat([
            self.matches['home_team'],
            self.matches['away_team']
        ]).unique()
        
        all_properties = []
        
        for team in teams:
            props = self.calculate_team_properties(team)
            if props:
                all_properties.append(props)
        
        return pd.DataFrame(all_properties)
    
    def calculate_aggregate_properties(self):
        total_matches = len(self.matches)
        
        home_wins = (self.matches['home_score'] > self.matches['away_score']).sum()
        draws = (self.matches['home_score'] == self.matches['away_score']).sum()
        away_wins = (self.matches['home_score'] < self.matches['away_score']).sum()
        
        total_home_goals = self.matches['home_score'].sum()
        total_away_goals = self.matches['away_score'].sum()
        
        aggregate = {
            'total_matches': total_matches,
            'home_win_prob': home_wins / total_matches,
            'draw_prob': draws / total_matches,
            'away_win_prob': away_wins / total_matches,
            'home_advantage': (home_wins / total_matches) - (away_wins / total_matches),
            'avg_home_goals': total_home_goals / total_matches,
            'avg_away_goals': total_away_goals / total_matches,
            'avg_total_goals': (total_home_goals + total_away_goals) / total_matches,
            'home_goal_advantage': (total_home_goals / total_matches) - (total_away_goals / total_matches)
        }
        
        return aggregate
    
    def save_properties(self, output_name, output_dir='properties'):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        team_props = self.calculate_all_teams()
        team_props.to_csv(output_dir / f'{output_name}_team_properties.csv', index=False)
        
        aggregate_props = self.calculate_aggregate_properties()
        with open(output_dir / f'{output_name}_aggregate_properties.json', 'w') as f:
            json.dump(aggregate_props, f, indent=2)
        
        return {
            'team_properties': str(output_dir / f'{output_name}_team_properties.csv'),
            'aggregate_properties': str(output_dir / f'{output_name}_aggregate_properties.json')
        }


def process_all_scored_matches():
    scored_dir = Path('scored_matches')
    filled_dir = Path('filled_matches')
    properties_dir = Path('properties')
    properties_dir.mkdir(parents=True, exist_ok=True)
    
    # Process baseline per period from filled_matches
    print("Processing baseline data per period...")
    if filled_dir.exists():
        period_files = sorted(filled_dir.glob('period_*.csv'), reverse=True)
        print(f"Found {len(period_files)} baseline periods")
        
        for period_file in period_files:
            period_name = period_file.stem
            
            try:
                matches = pd.read_csv(period_file)
                # Keep only original matches (not synthetic)
                if 'is_synthetic' in matches.columns:
                    original_matches = matches[matches['is_synthetic'] == False].copy()
                else:
                    original_matches = matches.copy()
                
                calculator = StatisticalPropertiesCalculator(original_matches)
                calculator.save_properties(f'baseline_{period_name}', properties_dir)
                
                aggregate = calculator.calculate_aggregate_properties()
                print(f"  {period_name}: {aggregate['total_matches']} matches")
                
            except Exception as e:
                print(f"  Error processing {period_name}: {e}")
                continue
    else:
        print(f"Warning: {filled_dir} not found, skipping baseline per period")
    
    # Process overall baseline
    print("\nProcessing overall baseline...")
    original_data = pd.read_csv('original_data/og/results.csv')
    calculator = StatisticalPropertiesCalculator(original_data)
    calculator.save_properties('baseline', properties_dir)
    aggregate = calculator.calculate_aggregate_properties()
    print(f"  baseline: {aggregate['total_matches']} matches")
    
    # Process each model's scored matches
    if not scored_dir.exists():
        print(f"\nError: {scored_dir} does not exist")
        return
    
    models = ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']
    
    for model in models:
        model_dir = scored_dir / model
        if not model_dir.exists():
            print(f"\nSkipping {model}: directory not found")
            continue
        
        # Create model subdirectory
        model_props_dir = properties_dir / f'prop_{model}'
        model_props_dir.mkdir(parents=True, exist_ok=True)
        
        period_files = sorted(model_dir.glob('period_*.csv'), reverse=True)
        
        print(f"\nProcessing {model} ({len(period_files)} periods)")
        
        for period_file in period_files:
            period_name = period_file.stem
            
            try:
                matches = pd.read_csv(period_file)
                
                calculator = StatisticalPropertiesCalculator(matches)
                calculator.save_properties(f'{model}_{period_name}', model_props_dir)
                
                aggregate = calculator.calculate_aggregate_properties()
                print(f"  {period_name}: {aggregate['total_matches']} matches")
                
            except Exception as e:
                print(f"  Error processing {period_name}: {e}")
                continue


if __name__ == '__main__':
    print("Calculating statistical properties for all scored matches...")
    process_all_scored_matches()
    print("\nDone! Properties saved to properties/ directory")