import pandas as pd
import json
import os
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from baseline_period_finder import BaselinePeriodFinder

class HistoricalGapFiller:
    def __init__(self, results_path, target_year=2025):
        self.results = pd.read_csv(results_path)
        self.results = self.results[self.results['tournament'] != 'CONIFA World Cup']
        self.results['date'] = pd.to_datetime(self.results['date'])
        self.results['year'] = self.results['date'].dt.year
        
        finder = BaselinePeriodFinder(results_path, target_year)
        baseline_result = finder.find_optimal_baseline()
        self.baseline = baseline_result['baseline']
        
        self.baseline_data = self.results[
            (self.results['year'] >= self.baseline['start_year']) &
            (self.results['year'] <= self.baseline['end_year'])
        ].copy()
        
        self.team_match_counts = self._calculate_team_baseline()
    
    def _calculate_team_baseline(self):
        teams = pd.concat([
            self.baseline_data['home_team'],
            self.baseline_data['away_team']
        ]).unique()
        
        baseline_years = self.baseline['end_year'] - self.baseline['start_year'] + 1
        
        team_stats = {}
        for team in teams:
            home_matches = self.baseline_data[self.baseline_data['home_team'] == team]
            away_matches = self.baseline_data[self.baseline_data['away_team'] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            
            team_stats[team] = {
                'matches_per_year': total_matches / baseline_years,
                'home_ratio': len(home_matches) / total_matches if total_matches > 0 else 0.5
            }
        
        return team_stats
    
    def train_synthesizer(self):
        print("Training SDV synthesizer on baseline data...")
        print(f"Baseline period: {self.baseline['start_year']}-{self.baseline['end_year']}")
        print(f"Training on {len(self.baseline_data)} matches")
        
        metadata = Metadata.detect_from_dataframe(self.baseline_data)
        metadata.update_column(
            column_name='date',
            sdtype='datetime',
            datetime_format='%Y-%m-%d'
        )
        
        self.synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=True)
        self.synthesizer.fit(self.baseline_data)
        
        print("Training complete!")
    
    def calculate_period_gap(self, period_start, period_end):
        period_data = self.results[
            (self.results['year'] >= period_start) &
            (self.results['year'] <= period_end)
        ]
        
        period_years = period_end - period_start + 1
        
        teams_in_period = pd.concat([
            period_data['home_team'],
            period_data['away_team']
        ]).unique()
        
        gaps = {}
        for team in teams_in_period:
            if team in self.team_match_counts:
                home_matches = period_data[period_data['home_team'] == team]
                away_matches = period_data[period_data['away_team'] == team]
                
                actual_total = len(home_matches) + len(away_matches)
                expected_total = self.team_match_counts[team]['matches_per_year'] * period_years
                
                gaps[team] = max(0, int(expected_total - actual_total))
        
        return gaps, period_data
    
    def generate_synthetic_period(self, period_start, period_end):
        gaps, period_data = self.calculate_period_gap(period_start, period_end)
        
        baseline_total = self.baseline['total_matches']
        actual_matches = len(period_data)
        
        synthetic_needed = baseline_total - actual_matches
        
        if synthetic_needed <= 0:
            return period_data
        
        synthetic_matches = self.synthesizer.sample(num_rows=synthetic_needed)
        
        teams_in_period = set(pd.concat([period_data['home_team'], period_data['away_team']]).unique())
        teams_in_baseline = set(self.team_match_counts.keys())
        valid_teams = list(teams_in_period.intersection(teams_in_baseline))
        
        if len(valid_teams) < 2:
            return period_data
        
        team_match_count = {team: 0 for team in valid_teams}
        for _, row in period_data.iterrows():
            if row['home_team'] in team_match_count:
                team_match_count[row['home_team']] += 1
            if row['away_team'] in team_match_count:
                team_match_count[row['away_team']] += 1
        
        period_years = period_end - period_start + 1
        team_deficit = {}
        for team in valid_teams:
            expected = self.team_match_counts[team]['matches_per_year'] * period_years
            actual = team_match_count[team]
            team_deficit[team] = max(0, expected - actual)
        
        for idx in synthetic_matches.index:
            teams_sorted = sorted(team_deficit.items(), key=lambda x: x[1], reverse=True)
            teams_needing = [t for t, deficit in teams_sorted if deficit > 0]
            
            if len(teams_needing) < 2:
                teams_needing = valid_teams
            
            home_team = teams_needing[0]
            away_team = teams_needing[1] if len(teams_needing) > 1 else teams_needing[0]
            
            synthetic_matches.loc[idx, 'home_team'] = home_team
            synthetic_matches.loc[idx, 'away_team'] = away_team
            
            if home_team in team_deficit:
                team_deficit[home_team] = max(0, team_deficit[home_team] - 1)
            if away_team in team_deficit:
                team_deficit[away_team] = max(0, team_deficit[away_team] - 1)
            
            match_date = synthetic_matches.loc[idx, 'date']
            if pd.notna(match_date):
                year = period_start + (idx % period_years)
                try:
                    synthetic_matches.loc[idx, 'date'] = match_date.replace(year=year)
                except ValueError:
                    synthetic_matches.loc[idx, 'date'] = pd.Timestamp(year=year, month=match_date.month, day=1)
        
        augmented_period = pd.concat([period_data, synthetic_matches], ignore_index=True)
        
        return augmented_period
    
    def save_augmented_period(self, period_start, period_end, output_path):
        augmented_period = self.generate_synthetic_period(period_start, period_end)
        
        all_other_matches = self.results[
            (self.results['year'] < period_start) | 
            (self.results['year'] > period_end)
        ]
        
        complete_dataset = pd.concat([all_other_matches, augmented_period], ignore_index=True)
        complete_dataset = complete_dataset.sort_values('date').reset_index(drop=True)
        
        complete_dataset.to_csv(output_path, index=False)
        
        gaps, original = self.calculate_period_gap(period_start, period_end)
        
        return {
            'period': f'{period_start}-{period_end}',
            'original_matches': len(original),
            'augmented_matches': len(augmented_period),
            'synthetic_added': len(augmented_period) - len(original),
            'total_dataset_size': len(complete_dataset),
            'output_file': output_path
        }
    
    def process_all_historical_periods(self, output_dir='augmented_periods'):
        os.makedirs(output_dir, exist_ok=True)
        
        baseline_end = self.baseline['end_year']
        earliest_year = self.results['year'].min()
        
        all_results = []
        augmented_data = self.results.copy()
        
        period_to_fill_end = self.baseline['start_year'] - 1
        
        while period_to_fill_end >= earliest_year + 3:
            period_to_fill_start = period_to_fill_end - 3
            
            training_data = augmented_data[
                (augmented_data['year'] >= period_to_fill_end + 1) &
                (augmented_data['year'] <= baseline_end)
            ]
            
            if len(training_data) < 100:
                print(f"Skipping period {period_to_fill_start}-{period_to_fill_end}: insufficient training data")
                period_to_fill_end = period_to_fill_start - 1
                continue
            
            print(f"\nTraining on accumulated data from {period_to_fill_start + 1}-{baseline_end} ({len(training_data)} matches)")
            print(f"Filling period {period_to_fill_start}-{period_to_fill_end}")
            
            metadata = Metadata.detect_from_dataframe(training_data)
            metadata.update_column(column_name='date', sdtype='datetime', datetime_format='%Y-%m-%d')
            
            #synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer = CTGANSynthesizer(
                metadata, # required
                enforce_rounding=False,
                epochs=100,
                verbose=True
            )
            synthesizer.fit(training_data)
            
            period_data = augmented_data[
                (augmented_data['year'] >= period_to_fill_start) &
                (augmented_data['year'] <= period_to_fill_end)
            ]
            
            target_matches = self.baseline['total_matches']
            actual_matches = len(period_data)
            synthetic_needed = target_matches - actual_matches
            
            if synthetic_needed > 0:
                synthetic_matches = synthesizer.sample(num_rows=synthetic_needed)
                
                for idx in synthetic_matches.index:
                    date = synthetic_matches.loc[idx, 'date']
                    if pd.notna(date):
                        year = period_to_fill_start + (idx % 4)
                        try:
                            synthetic_matches.loc[idx, 'date'] = date.replace(year=year)
                        except ValueError:
                            synthetic_matches.loc[idx, 'date'] = pd.Timestamp(year=year, month=date.month, day=1)
                
                augmented_period = pd.concat([period_data, synthetic_matches], ignore_index=True)
                
                augmented_data = augmented_data[
                    (augmented_data['year'] < period_to_fill_start) |
                    (augmented_data['year'] > period_to_fill_end)
                ]
                augmented_data = pd.concat([augmented_data, augmented_period], ignore_index=True)
            else:
                augmented_period = period_data
            
            complete_dataset = augmented_data.sort_values('date').reset_index(drop=True)
            
            output_path = f"{output_dir}/period_{period_to_fill_start}_{period_to_fill_end}.csv"
            complete_dataset.to_csv(output_path, index=False)
            
            result = {
                'period_filled': f'{period_to_fill_start}-{period_to_fill_end}',
                'training_size': len(training_data),
                'original_matches': actual_matches,
                'augmented_matches': len(augmented_period),
                'synthetic_added': synthetic_needed if synthetic_needed > 0 else 0,
                'total_dataset_size': len(complete_dataset),
                'output_file': output_path
            }
            all_results.append(result)
            
            print(f"Period {period_to_fill_start}-{period_to_fill_end}: {actual_matches} → {len(augmented_period)} matches (+{result['synthetic_added']})")
            
            period_to_fill_end = period_to_fill_start - 1
        
        summary_path = f"{output_dir}/augmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results

if __name__ == '__main__':
    filler = HistoricalGapFiller('data/results.csv')
    
    results = filler.process_all_historical_periods()
    print(f"\nProcessed {len(results)} historical periods")