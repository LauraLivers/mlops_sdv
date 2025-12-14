
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class HistoricalStageLabeler:
    def __init__(self, data_path='original_data/results.csv'):
        """Initialize the labeler with historical data."""
        self.data_path = data_path
        self.df = None
        
        # Define tournament structure rules (matches -> stage distribution)
        self.tournament_structures = {
            'FIFA World Cup': {
                64: {  # 2023 format: 48 group + 16 knockout
                    'final': 1,
                    'third_place': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'round_of_16': 8,
                    'group': 48
                },
                52: {  # 2015-2019 format
                    'final': 1,
                    'third_place': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'round_of_16': 8,
                    'group': 36
                },
                32: {  # Older format
                    'final': 1,
                    'third_place': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 24
                }
            },
            'UEFA Euro': {
                31: {  # Modern Euro format
                    'final': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'round_of_16': 8,
                    'group': 16
                },
                25: {  # Older format
                    'final': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 18
                },
                15: {  # Small tournament
                    'final': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 8
                }
            },
            'Copa América': {
                26: {  # Standard Copa format
                    'final': 1,
                    'third_place': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 18
                },
                20: {  # Smaller format
                    'final': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 13
                }
            },
            'Olympic Games': {
                26: {  # Standard Olympic format
                    'final': 1,
                    'third_place': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 18
                },
                20: {
                    'final': 1,
                    'semi_final': 2,
                    'quarter_final': 4,
                    'group': 13
                }
            }
        }
        
        # Tournaments where we can infer stages
        self.structured_tournaments = [
            'FIFA World Cup',
            'UEFA Euro',
            'Copa América', 
            'Olympic Games',
            'African Cup of Nations',
            'AFC Asian Cup',
            'CONCACAF Championship',
            'Gold Cup'
        ]
    
    def load_data(self):
        """Load historical data."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        if 'stage' not in self.df.columns:
            self.df['stage'] = None
        
        print(f"Loaded {len(self.df):,} historical matches")
        return self
    
    def _infer_knockout_structure(self, num_matches):

        if num_matches <= 3:
            return {'final': 1, 'group': num_matches - 1}
        
        structure = {}
        remaining = num_matches
        
        # Final (always 1)
        structure['final'] = 1
        remaining -= 1
        
        # 3rd place match (for tournaments with >16 matches)
        if num_matches >= 16:
            structure['third_place'] = 1
            remaining -= 1
        
        # Semi-finals (always 2 if enough matches)
        if remaining >= 2:
            structure['semi_final'] = 2
            remaining -= 2
        
        # Quarter-finals (4 if enough matches)
        if remaining >= 4:
            structure['quarter_final'] = 4
            remaining -= 4
        
        # Round of 16 (8 if enough matches)
        if remaining >= 8:
            structure['round_of_16'] = 8
            remaining -= 8
        
        # Everything else is group stage
        if remaining > 0:
            structure['group'] = remaining
        
        return structure
    
    def _get_tournament_structure(self, tournament_name, num_matches):
        """Get the stage structure for a tournament based on match count."""
        if tournament_name in self.tournament_structures:
            # Check for exact match
            if num_matches in self.tournament_structures[tournament_name]:
                return self.tournament_structures[tournament_name][num_matches]
            
            # Find closest match count
            available_counts = list(self.tournament_structures[tournament_name].keys())
            closest = min(available_counts, key=lambda x: abs(x - num_matches))
            
            # If within 20% of a known structure, use it
            if abs(closest - num_matches) / closest < 0.2:
                return self.tournament_structures[tournament_name][closest]
        
        return self._infer_knockout_structure(num_matches)
    
    def label_tournament_instance(self, matches_df, tournament_name):

        num_matches = len(matches_df)
        structure = self._get_tournament_structure(tournament_name, num_matches)
        
        stages = ['group'] * num_matches  # Default everything to group

        idx = num_matches - 1
        
        # 1. Last match = FINAL
        if 'final' in structure and idx >= 0:
            stages[idx] = 'final'
            idx -= 1
        
        # 2. Third place (usually same day or day before final)
        if 'third_place' in structure and idx >= 0:
            stages[idx] = 'third_place'
            idx -= 1
        
        # 3. Semi-finals (2 matches before final/third place)
        if 'semi_final' in structure:
            for _ in range(structure['semi_final']):
                if idx >= 0:
                    stages[idx] = 'semi_final'
                    idx -= 1
        
        # 4. Quarter-finals
        if 'quarter_final' in structure:
            for _ in range(structure['quarter_final']):
                if idx >= 0:
                    stages[idx] = 'quarter_final'
                    idx -= 1
        
        # 5. Round of 16
        if 'round_of_16' in structure:
            for _ in range(structure['round_of_16']):
                if idx >= 0:
                    stages[idx] = 'round_of_16'
                    idx -= 1
        
        return stages
    
    def label_all_tournaments(self):
        """Label stages for all structured tournaments in the dataset."""
        print("\nLabeling tournament stages...")
        
        labeled_count = 0
        
        for tournament in self.structured_tournaments:
            tournament_matches = self.df[self.df['tournament'] == tournament].copy()
            
            if len(tournament_matches) == 0:
                continue
            
            print(f"\nProcessing {tournament}...")
            
            tournament_matches['year'] = tournament_matches['date'].dt.year
            
            for year in tournament_matches['year'].unique():
                year_matches = tournament_matches[tournament_matches['year'] == year].copy()
                year_matches = year_matches.sort_values('date')
                
                stages = self.label_tournament_instance(year_matches, tournament)
                indices = year_matches.index
                self.df.loc[indices, 'stage'] = stages
                
                labeled_count += len(stages)
                
                print(f"  {year}: {len(year_matches)} matches - "
                      f"Final: {stages.count('final')}, "
                      f"SF: {stages.count('semi_final')}, "
                      f"QF: {stages.count('quarter_final')}, "
                      f"Group: {stages.count('group')}")
        
        self._label_remaining_matches()
        return self
    
    def _label_remaining_matches(self):
        """Label remaining matches based on tournament type."""
        qualifier_tournaments = self.df[
            (self.df['stage'].isna()) & 
            (self.df['tournament'].str.contains('qualif', case=False, na=False))
        ]
        self.df.loc[qualifier_tournaments.index, 'stage'] = 'qualifier'
        
        friendly_tournaments = self.df[
            (self.df['stage'].isna()) & 
            (self.df['tournament'].str.contains('friendly', case=False, na=False))
        ]
        self.df.loc[friendly_tournaments.index, 'stage'] = 'friendly'
        
        remaining = self.df[self.df['stage'].isna()].copy()
        if len(remaining) > 0:
            remaining['year'] = remaining['date'].dt.year
            
            for tournament in remaining['tournament'].unique():
                t_matches = remaining[remaining['tournament'] == tournament]
                
                for year in t_matches['year'].unique():
                    year_matches = t_matches[t_matches['year'] == year]
                    
                    if len(year_matches) == 1:
                        # Single match = final
                        self.df.loc[year_matches.index, 'stage'] = 'final'
                    elif len(year_matches) == 2:
                        # Two matches = likely final + 3rd place or 2 semis
                        indices = year_matches.sort_values('date').index
                        self.df.loc[indices[0], 'stage'] = 'semi_final'
                        self.df.loc[indices[1], 'stage'] = 'final'
                    else:
                        # Multiple matches - last is final, rest are group
                        year_matches_sorted = year_matches.sort_values('date')
                        indices = year_matches_sorted.index
                        self.df.loc[indices[-1], 'stage'] = 'final'
                        self.df.loc[indices[:-1], 'stage'] = 'group'
        
        self.df.loc[self.df['stage'].isna(), 'stage'] = 'group'
    
    def save_labeled_data(self, output_path=None):
        """Save the labeled data."""
        if output_path is None:
            output_path = self.data_path.replace('.csv', '_with_stages.csv')
        
        self.df.to_csv(output_path, index=False)
        
        # Also save to original location for consistency
        if output_path != self.data_path:
            self.df.to_csv(self.data_path, index=False)
        
        return self
    
    def print_summary(self):

        stage_counts = self.df['stage'].value_counts()
        for stage, count in stage_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {stage:15s}: {count:6,} matches ({pct:5.1f}%)")
        
        for tournament in self.structured_tournaments:
            count = len(self.df[self.df['tournament'] == tournament])
            if count > 0:
                finals = len(self.df[
                    (self.df['tournament'] == tournament) & 
                    (self.df['stage'] == 'final')
                ])
                print(f"  {tournament:30s}: {count:4,} matches, {finals} finals")
        


def main():
    """Main execution function."""
    labeler = HistoricalStageLabeler()
    
    labeler.load_data()
    labeler.label_all_tournaments()
    labeler.save_labeled_data()
    labeler.print_summary()
    
if __name__ == "__main__":
    main()
