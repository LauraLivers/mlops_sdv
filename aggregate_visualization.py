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
        
        self.model_colors = {
            'poisson': '#11fa1d',
            'ctgan': '#fa11e5',
            'gaussian_copula': '#2404e1',
            'copulagan': '#edc903',
            'tvae': '#03e8ed'
        }
    
    def get_model_data(self, model_name):
        scored_dir = Path('scored_matches') / model_name
        if not scored_dir.exists():
            return None, None
        
        all_years = []
        all_values = []
        
        for period in self.periods:
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
            
            all_years.extend(yearly_stats['year'].values)
            all_values.extend(yearly_stats['home_win_prob'].values)
        
        if all_years:
            return all_years, all_values
        return None, None
    
    def visualize_progressive_models(self):
        output_dir = Path('visualizations/aggregate_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_df = pd.read_csv('original_data/og/results.csv')
        baseline_df['date'] = pd.to_datetime(baseline_df['date'])
        baseline_df['year'] = baseline_df['date'].dt.year
        
        baseline_by_year = baseline_df.groupby('year', as_index=False).apply(
            lambda x: pd.Series({
                'year': x.name,
                'home_win_prob': (x['home_score'] > x['away_score']).mean()
            }), include_groups=False
        )
        
        model_order = ['gaussian_copula', 'poisson', 'ctgan', 'tvae']
        
        progressive_sets = [
            (['baseline'], 'baseline'),
            (['baseline', 'gaussian_copula'], 'baseline_gc'),
            (['baseline', 'gaussian_copula', 'poisson'], 'baseline_gc_poisson'),
            (['baseline', 'gaussian_copula', 'poisson', 'ctgan'], 'baseline_gc_poisson_ctgan'),
            (['baseline', 'gaussian_copula', 'poisson', 'ctgan', 'tvae'], 'baseline_gc_poisson_ctgan_tvae')
        ]
        
        for models_to_plot, filename in progressive_sets:
            print(f"Creating plot: {filename}...")
            
            fig, ax = plt.subplots(figsize=(24, 10))
            
            ax.plot(baseline_by_year['year'], baseline_by_year['home_win_prob'],
                   color='black', linewidth=4, linestyle='--', label='BASELINE',
                   alpha=1.0, zorder=-1)
            
            for model_name in models_to_plot:
                if model_name == 'baseline':
                    continue
                
                years, values = self.get_model_data(model_name)
                if years is not None:
                    ax.plot(years, values,
                           color=self.model_colors[model_name], linewidth=2.5, 
                           label=model_name.upper().replace('_', ' '), alpha=0.8)
            
            ax.set_title('Home Win Probability Comparison',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=14, fontweight='bold')
            ax.set_ylabel('Home Win Probability', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.2, linestyle=':')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_dir}/{filename}.png")
    
    def visualize_tvae_highlight(self):
        output_dir = Path('visualizations/aggregate_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating plot: tvae_highlight...")
        
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
        
        ax.plot(baseline_by_year['year'], baseline_by_year['home_win_prob'],
               color='black', linewidth=4, linestyle='--', label='BASELINE',
               alpha=1.0, zorder=-1)
        
        models_to_plot = ['gaussian_copula', 'poisson', 'ctgan', 'tvae']
        
        for model_name in models_to_plot:
            years, values = self.get_model_data(model_name)
            if years is not None:
                if model_name == 'tvae':
                    ax.plot(years, values,
                           color=self.model_colors[model_name], linewidth=5, 
                           label=model_name.upper().replace('_', ' '), alpha=1.0, zorder=10)
                else:
                    ax.plot(years, values,
                           color=self.model_colors[model_name], linewidth=2.5, 
                           label=model_name.upper().replace('_', ' '), alpha=0.8)
        
        ax.set_title('Home Win Probability Comparison - TVAE Best Performer',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Home Win Probability', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tvae_highlight.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_dir}/tvae_highlight.png")


def main():
    print("Creating progressive aggregate visualizations...\n")
    visualizer = AggregateVisualizer()
    visualizer.visualize_progressive_models()
    visualizer.visualize_tvae_highlight()
    print("\nAll visualizations complete!")


if __name__ == '__main__':
    main()
