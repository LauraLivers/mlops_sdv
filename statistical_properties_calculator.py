import pandas as pd
import numpy as np
import json
import os

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
    
    def save_properties(self, output_prefix, output_dir='properties'):
        os.makedirs(output_dir, exist_ok=True)
        
        team_props = self.calculate_all_teams()
        team_props.to_csv(f'{output_dir}/{output_prefix}_team_properties.csv', index=False)
        
        aggregate_props = self.calculate_aggregate_properties()
        with open(f'{output_dir}/{output_prefix}_aggregate_properties.json', 'w') as f:
            json.dump(aggregate_props, f, indent=2)
        
        return {
            'team_properties': f'{output_dir}/{output_prefix}_team_properties.csv',
            'aggregate_properties': f'{output_dir}/{output_prefix}_aggregate_properties.json'
        }

if __name__ == '__main__':
    matches = pd.read_csv('data/results.csv')
    matches['date'] = pd.to_datetime(matches['date'])
    
    baseline_matches = matches[
        (matches['date'].dt.year >= 2019) &
        (matches['date'].dt.year <= 2022)
    ]
    
    calculator = StatisticalPropertiesCalculator(baseline_matches)
    files = calculator.save_properties('baseline')
    
    aggregate = calculator.calculate_aggregate_properties()
    print("Baseline aggregate properties:")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nSaved to:")
    for key, path in files.items():
        print(f"  {path}")