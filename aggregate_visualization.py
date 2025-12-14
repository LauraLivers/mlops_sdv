import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AggregateVisualizer:
    def __init__(self):
        self.properties_dir = Path('properties')
        self.models = ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']
        self.periods = [
            '1957_1960', '1961_1964', '1965_1968', '1969_1972',
            '1973_1976', '1977_1980', '1981_1984', '1985_1988',
            '1989_1992', '1993_1996', '1997_2000', '2001_2004',
            '2005_2008', '2009_2012', '2013_2016', '2017_2020'
        ]
        
        self.period_colors = {
            '1957_1960': '#2C3E50', '1961_1964': '#34495E', '1965_1968': '#7F8C8D',
            '1969_1972': '#95A5A6', '1973_1976': '#BDC3C7', '1977_1980': '#ECF0F1',
            '1981_1984': '#3498DB', '1985_1988': '#5DADE2', '1989_1992': '#85C1E9',
            '1993_1996': '#AED6F1', '1997_2000': '#D6EAF8', '2001_2004': '#E8F8F5',
            '2005_2008': '#A3E4D7', '2009_2012': '#76D7C4', '2013_2016': '#48C9B0',
            '2017_2020': '#1ABC9C'
        }
    
    def visualize_single_model(self, model_name='gaussian_copula', display_name='GAUSSIAN_COPULA'):
        output_dir = Path('visualizations/aggregate_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scored_dir = Path('scored_matches') / model_name
        if not scored_dir.exists():
            print(f"No scored matches found for {model_name}")
            return
        
        print(f"Processing {model_name}...")
        
        baseline_df = pd.read_csv('original_data/og/results.csv')
        baseline_df['date'] = pd.to_datetime(baseline_df['date'])
        baseline_df['year'] = baseline_df['date'].dt.year
        
        baseline_by_year = baseline_df.groupby('year', as_index=False).apply(
            lambda x: pd.Series({
                'year': x.name,
                'home_win_prob': (x['home_score'] > x['away_score']).mean()
            }), include_groups=False
        )
        
        fig, ax = plt.subplots(figsize=(24, 10))
        
        prev_end_year = None
        prev_end_value = None
        
        for idx, period in enumerate(self.periods):
            period_file = scored_dir / f'period_{period}.csv'
            if not period_file.exists():
                continue
                
            start_year, end_year = map(int, period.split('_'))
            
            df = pd.read_csv(period_file)
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            
            period_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
            
            if len(period_data) == 0:
                continue
            
            yearly_stats = period_data.groupby('year', as_index=False).apply(
                lambda x: pd.Series({
                    'year': x.name,
                    'home_win_prob': (x['home_score'] > x['away_score']).mean()
                }), include_groups=False
            )
            
            years = yearly_stats['year'].values
            values = yearly_stats['home_win_prob'].values
            
            if prev_end_year is not None and prev_end_value is not None:
                boundary = start_year - 0.5
                ax.fill_between([prev_end_year, boundary], [0, 0], 
                               [prev_end_value, prev_end_value],
                               color=self.period_colors[self.periods[idx-1]], alpha=0.6)
                
                years = np.concatenate([[boundary, start_year], years])
                values = np.concatenate([[prev_end_value, values[0]], values])
            
            ax.fill_between(years, 0, values,
                           color=self.period_colors[period], alpha=0.6, label=f'{period}')
            
            ax.plot(years, values,
                   color=self.period_colors[period], linewidth=2, alpha=0.9)
            
            prev_end_year = end_year
            prev_end_value = values[-1]
            
            if idx < len(self.periods) - 1:
                ax.axvline(x=end_year + 0.5, color='black', linewidth=1.5, 
                          linestyle='-', alpha=0.4, zorder=10)
        
        ax.plot(baseline_by_year['year'], baseline_by_year['home_win_prob'],
               color='red', linewidth=3, linestyle='-', label='Baseline',
               alpha=1.0, zorder=20)
        
        ax.set_title(f'{display_name}: Home Win Probability by Period',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Home Win Probability', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_dir}/{model_name}_temporal_evolution.png")


def main():
    print("Creating aggregate visualizations for all models...\n")
    visualizer = AggregateVisualizer()
    
    model_configs = [
        ('poisson', 'POISSON'),
        ('ctgan', 'CTGAN'),
        ('gaussian_copula', 'GAUSSIAN COPULA'),
        ('copulagan', 'COPULAGAN'),
        ('tvae', 'TVAE')
    ]
    
    for model_name, display_name in model_configs:
        visualizer.visualize_single_model(model_name, display_name)
    
    print("\nAll visualizations complete!")


if __name__ == '__main__':
    main()
