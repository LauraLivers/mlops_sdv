import pandas as pd

class BaselinePeriodFinder:
    def __init__(self, results_path, target_year=2025):
        self.results = pd.read_csv(results_path)
        self.results = self.results[self.results['tournament'] != 'CONIFA World Cup']
        self.results['date'] = pd.to_datetime(self.results['date'])
        self.results['year'] = self.results['date'].dt.year
        self.target_year = target_year
        
        euro_tournaments = ['UEFA Euro', 'UEFA Euro qualification', 'Euro', 'European Championship']
        self.target_matches = self.results[
            (self.results['tournament'].isin(euro_tournaments)) &
            (self.results['year'] == target_year)
        ]

    
    def find_optimal_baseline(self):
        baseline_start = self.target_year - 4
        baseline_end = self.target_year - 1
        
        period_mask = (self.results['year'] >= baseline_start) & (self.results['year'] <= baseline_end)
        period_data = self.results[period_mask]
        
        total_matches = len(period_data)
        nations = pd.concat([period_data['home_team'], period_data['away_team']]).nunique()
        
        return {
            'baseline': {
                'start_year': baseline_start,
                'end_year': baseline_end,
                'total_matches': total_matches,
                'nations': nations
            },
            'target_year': self.target_year,
            'target_matches': len(self.target_matches)
        }
    
    def print_analysis(self):
        result = self.find_optimal_baseline()
        baseline = result['baseline']
        
        print(f"TARGET: UEFA Euro {self.target_year}")
        print(f"  Target matches found: {len(self.target_matches)}")
        print()
        print("BASELINE PERIOD:")
        print(f"  Period: {baseline['start_year']}-{baseline['end_year']}")
        print(f"  Total matches: {baseline['total_matches']}")
        print(f"  Nations: {baseline['nations']}")
        
        return result

if __name__ == '__main__':
    finder = BaselinePeriodFinder('data/results.csv')
    result = finder.print_analysis()