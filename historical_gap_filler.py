import pandas as pd
import json
import os
import numpy as np
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from baseline_period_finder import BaselinePeriodFinder
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class HistoricalGapFiller:
    def __init__(self, results_path, target_year=2025, synthesizer_type='ctgan'):
        self.results = pd.read_csv(results_path)
        self.results = self.results[self.results['tournament'] != 'CONIFA World Cup']
        self.results['date'] = pd.to_datetime(self.results['date'])
        
        # Load reference tables for constraints
        self.team_country = pd.read_csv('reference_tables/team_country.csv')
        self.tournament_locations = pd.read_csv('reference_tables/tournament_locations.csv')
        self.tournament_teams = pd.read_csv('reference_tables/tournament_teams.csv')
        self.city_country = pd.read_csv('reference_tables/city_country.csv')
        
        # Synthesizer configuration
        self.synthesizer_type = synthesizer_type.lower()
        if self.synthesizer_type not in ['ctgan', 'gaussian']:
            raise ValueError("synthesizer_type must be 'ctgan' or 'gaussian'")
        
        finder = BaselinePeriodFinder(results_path, target_year)
        baseline_result = finder.find_optimal_baseline()
        self.baseline = baseline_result['baseline']
        
        # Extract baseline data
        self.baseline_data = self.results[
            (self.results['date'].dt.year >= self.baseline['start_year']) &
            (self.results['date'].dt.year <= self.baseline['end_year'])
        ].copy()
        
        self.team_match_counts = self._calculate_team_baseline()
    
    def _calculate_team_baseline(self):
        """Calculate expected matches per team per year from baseline"""
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
    
    def _create_synthesizer(self, metadata):
        """Create synthesizer based on configuration"""
        if self.synthesizer_type == 'ctgan':
            return CTGANSynthesizer(
                metadata,
                enforce_rounding=False,
                epochs=100,
                verbose=True
            )
        else:  # gaussian
            return GaussianCopulaSynthesizer(metadata)
    
    def _fix_synthetic_constraints(self, synthetic_df):
        """Fix constraint violations in synthetic data using reference tables"""
        fixed_df = synthetic_df.copy()
        
        for idx in fixed_df.index:
            tournament = fixed_df.loc[idx, 'tournament']
            
            # Get all teams that have historically played in this tournament
            valid_teams = self.tournament_teams[
                self.tournament_teams['tournament'] == tournament
            ]['team'].unique()
            
            if len(valid_teams) >= 2:
                # Pick two different teams that have both played in this tournament
                home_team, away_team = np.random.choice(valid_teams, size=2, replace=False)
                fixed_df.loc[idx, 'home_team'] = home_team
                fixed_df.loc[idx, 'away_team'] = away_team
            
            # Get valid location for this tournament
            valid_locations = self.tournament_locations[
                self.tournament_locations['tournament'] == tournament
            ]
            
            if len(valid_locations) > 0:
                sampled_loc = valid_locations.sample(1).iloc[0]
                fixed_df.loc[idx, 'country'] = sampled_loc['country']
                fixed_df.loc[idx, 'city'] = sampled_loc['city']
                fixed_df.loc[idx, 'neutral'] = sampled_loc['neutral']
        
        return fixed_df
    
    def process_all_historical_periods(self, output_dir='augmented_periods'):
        """Fill historical gaps working backwards from baseline"""
        os.makedirs(output_dir, exist_ok=True)
        
        baseline_end = self.baseline['end_year']
        earliest_year = self.results['date'].dt.year.min()
        
        all_results = []
        augmented_data = self.results.copy()
        
        # Start from just before baseline and work backwards (sparsest gets most training data)
        period_to_fill_end = self.baseline['start_year'] - 1
        
        while period_to_fill_end >= earliest_year + 3:
            period_to_fill_start = period_to_fill_end - 3
            
            # Training data: everything from end of current period to baseline_end
            training_data = augmented_data[
                (augmented_data['date'].dt.year >= period_to_fill_end + 1) &
                (augmented_data['date'].dt.year <= baseline_end)
            ].copy()
            
            if len(training_data) < 100:
                print(f"Skipping period {period_to_fill_start}-{period_to_fill_end}: insufficient training data")
                period_to_fill_end = period_to_fill_start - 1
                continue
            
            print(f"\nUsing {self.synthesizer_type.upper()} synthesizer")
            print(f"Training on accumulated data from {period_to_fill_end + 1}-{baseline_end} ({len(training_data)} matches)")
            print(f"Filling period {period_to_fill_start}-{period_to_fill_end}")
            
            # Prepare metadata
            metadata = Metadata.detect_from_dataframe(training_data)
            metadata.update_column(column_name='date', sdtype='datetime', datetime_format='%Y-%m-%d')
            
            synthesizer = self._create_synthesizer(metadata)
            synthesizer.fit(training_data)
            
            # Get period data
            period_data = augmented_data[
                (augmented_data['date'].dt.year >= period_to_fill_start) &
                (augmented_data['date'].dt.year <= period_to_fill_end)
            ].copy()
            
            target_matches = self.baseline['total_matches']
            actual_matches = len(period_data)
            synthetic_needed = target_matches - actual_matches
            
            if synthetic_needed > 0:
                synthetic_matches = synthesizer.sample(num_rows=synthetic_needed)
                
                # Fix constraint violations using reference tables
                synthetic_matches = self._fix_synthetic_constraints(synthetic_matches)
                
                # Fix dates to be in the target period
                period_years = period_to_fill_end - period_to_fill_start + 1
                for idx in synthetic_matches.index:
                    date = synthetic_matches.loc[idx, 'date']
                    if pd.notna(date):
                        year = period_to_fill_start + (idx % period_years)
                        try:
                            synthetic_matches.loc[idx, 'date'] = date.replace(year=year)
                        except ValueError:
                            synthetic_matches.loc[idx, 'date'] = pd.Timestamp(year=year, month=date.month, day=1)
                
                augmented_period = pd.concat([period_data, synthetic_matches], ignore_index=True)
                
                # Replace period in augmented_data
                augmented_data = augmented_data[
                    (augmented_data['date'].dt.year < period_to_fill_start) |
                    (augmented_data['date'].dt.year > period_to_fill_end)
                ]
                augmented_data = pd.concat([augmented_data, augmented_period], ignore_index=True)
            else:
                augmented_period = period_data
            
            # Sort by date and save
            complete_dataset = augmented_data.sort_values('date').reset_index(drop=True)
            
            output_path = f"{output_dir}/period_{period_to_fill_start}_{period_to_fill_end}.csv"
            complete_dataset.to_csv(output_path, index=False)
            
            result = {
                'period_filled': f'{period_to_fill_start}-{period_to_fill_end}',
                'synthesizer_used': self.synthesizer_type,
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
        
        summary_path = f"{output_dir}/augmentation_summary_{self.synthesizer_type}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results

if __name__ == '__main__':
    # Test with CTGAN
    print("TESTING WITH CTGAN")
    filler_ctgan = HistoricalGapFiller('cleaned_data/results_standardized_qualifiers.csv', synthesizer_type='ctgan')
    results_ctgan = filler_ctgan.process_all_historical_periods(output_dir='augmented_periods_ctgan')
    print(f"\nCTGAN: Processed {len(results_ctgan)} historical periods")
    
    # Test with Gaussian Copula
    print("TESTING WITH GAUSSIAN COPULA")
    filler_gaussian = HistoricalGapFiller('cleaned_data/results_standardized_qualifiers.csv', synthesizer_type='gaussian')
    results_gaussian = filler_gaussian.process_all_historical_periods(output_dir='augmented_periods_gaussian')
    print(f"\nGaussian: Processed {len(results_gaussian)} historical periods")