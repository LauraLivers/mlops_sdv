import pandas as pd
from datetime import timedelta

class BaselinePeriodFinder:
    def __init__(self, results_path, target_year=2025):
        self.results = pd.read_csv(results_path)
        self.results = self.results[self.results['tournament'] != 'CONIFA World Cup']
        self.results['date'] = pd.to_datetime(self.results['date'])
        self.results['year'] = self.results['date'].dt.year
        self.target_year = target_year
        
        self.euro_tournaments = ['UEFA Euro', 'Euro', 'European Championship']
        self.euro_qualifiers = ['UEFA Euro qualification']
        
        self.target_matches = self.results[
            (self.results['tournament'].isin(self.euro_tournaments + self.euro_qualifiers)) &
            (self.results['year'] == target_year)
        ]

    def extract_tournament_periods(self):
        """Extract dense tournament periods from historical Euro tournaments."""
        baseline_start = self.target_year - 4
        baseline_end = self.target_year - 1
        
        tournament_periods = []
        
        for year in range(baseline_start, baseline_end + 1):
            euro_matches = self.results[
                (self.results['tournament'].isin(self.euro_tournaments)) &
                (self.results['year'] == year)
            ]
            
            if len(euro_matches) > 0:
                start_date = euro_matches['date'].min()
                end_date = euro_matches['date'].max()
                duration_days = (end_date - start_date).days
                
                tournament_periods.append({
                    'year': year,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': duration_days,
                    'total_matches': len(euro_matches),
                    'start_month': start_date.month,
                    'start_day': start_date.day,
                    'end_month': end_date.month,
                    'end_day': end_date.day
                })
        
        return tournament_periods
    
    def get_typical_tournament_window(self):
        """Determine typical tournament timing from historical data."""
        periods = self.extract_tournament_periods()
        
        if not periods:
            return {
                'typical_start_month': 6,
                'typical_start_day': 15,
                'typical_end_month': 7,
                'typical_end_day': 15,
                'typical_duration_days': 30,
                'confidence': 'default'
            }
        
        df_periods = pd.DataFrame(periods)
        
        return {
            'typical_start_month': int(df_periods['start_month'].mode()[0]),
            'typical_start_day': int(df_periods['start_day'].median()),
            'typical_end_month': int(df_periods['end_month'].mode()[0]),
            'typical_end_day': int(df_periods['end_day'].median()),
            'typical_duration_days': int(df_periods['duration_days'].median()),
            'avg_matches': int(df_periods['total_matches'].mean()),
            'confidence': 'historical',
            'sample_size': len(periods)
        }
    
    def extract_qualifier_patterns(self):
        """Analyze when qualifier matches typically occur."""
        baseline_start = self.target_year - 4
        baseline_end = self.target_year - 1
        
        qualifier_matches = self.results[
            (self.results['tournament'].isin(self.euro_qualifiers)) &
            (self.results['year'] >= baseline_start) &
            (self.results['year'] <= baseline_end)
        ].copy() 
        
        if len(qualifier_matches) == 0:
            return None
        
        qualifier_matches['month'] = qualifier_matches['date'].dt.month
        monthly_distribution = qualifier_matches.groupby('month').size().to_dict()
        
        return {
            'total_qualifier_matches': len(qualifier_matches),
            'years_covered': qualifier_matches['year'].nunique(),
            'monthly_distribution': monthly_distribution,
            'peak_months': sorted(monthly_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def find_optimal_baseline(self):
        baseline_start = self.target_year - 4
        baseline_end = self.target_year - 1
        
        period_mask = (self.results['year'] >= baseline_start) & (self.results['year'] <= baseline_end)
        period_data = self.results[period_mask]
        
        total_matches = len(period_data)
        nations = pd.concat([period_data['home_team'], period_data['away_team']]).nunique()
        
        tournament_matches = period_data[period_data['tournament'].isin(self.euro_tournaments)]
        qualifier_matches = period_data[period_data['tournament'].isin(self.euro_qualifiers)]
        
        return {
            'baseline': {
                'start_year': baseline_start,
                'end_year': baseline_end,
                'total_matches': total_matches,
                'nations': nations,
                'tournament_matches': len(tournament_matches),
                'qualifier_matches': len(qualifier_matches)
            },
            'target_year': self.target_year,
            'target_matches': len(self.target_matches),
            'tournament_periods': self.extract_tournament_periods(),
            'typical_tournament_window': self.get_typical_tournament_window(),
            'qualifier_patterns': self.extract_qualifier_patterns()
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
        print(f"  Tournament matches: {baseline['tournament_matches']}")
        print(f"  Qualifier matches: {baseline['qualifier_matches']}")
        print(f"  Nations: {baseline['nations']}")
        print()
        
        # Print tournament periods
        print("HISTORICAL TOURNAMENT PERIODS:")
        for period in result['tournament_periods']:
            print(f"  {period['year']}: {period['start_date'].strftime('%Y-%m-%d')} to "
                  f"{period['end_date'].strftime('%Y-%m-%d')} ({period['duration_days']} days, "
                  f"{period['total_matches']} matches)")
        print()
        
        
        if result['qualifier_patterns']:
            qual = result['qualifier_patterns']
            print("QUALIFIER PATTERNS:")
            print(f"  Total matches: {qual['total_qualifier_matches']}")
            print(f"  Peak months: {', '.join([f'Month {m} ({c} matches)' for m, c in qual['peak_months']])}")
        
        return result

if __name__ == '__main__':
    finder = BaselinePeriodFinder('original_data/results.csv')
    result = finder.print_analysis()