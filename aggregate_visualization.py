import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AggregateVisualizer:
    def __init__(self):
        self.properties_dir = Path('properties')
        self.models = ['baseline', 'poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']
        self.periods = [
            '1957_1960', '1961_1964', '1965_1968', '1969_1972',
            '1973_1976', '1977_1980', '1981_1984', '1985_1988',
            '1989_1992', '1993_1996', '1997_2000', '2001_2004',
            '2005_2008', '2009_2012', '2013_2016', '2017_2020'
        ]
        self.colors = {
            'baseline': 'black',
            'poisson': '#1f77b4',
            'ctgan': '#ff7f0e',
            'gaussian_copula': '#2ca02c',
            'copulagan': '#d62728',
            'tvae': '#9467bd'
        }
    
    def _load_data_for_metric(self, metric):
        """Load all data and organize it as: rows=models, cols=periods"""
        data = []
        
        # Load baseline per period from ORIGINAL DATA filtered by period dates
        original_df = pd.read_csv('original_data/og/results.csv')
        original_df['date'] = pd.to_datetime(original_df['date'])
        
        period_ranges = {
            '1957_1960': ('1957-01-01', '1960-12-31'),
            '1961_1964': ('1961-01-01', '1964-12-31'),
            '1965_1968': ('1965-01-01', '1968-12-31'),
            '1969_1972': ('1969-01-01', '1972-12-31'),
            '1973_1976': ('1973-01-01', '1976-12-31'),
            '1977_1980': ('1977-01-01', '1980-12-31'),
            '1981_1984': ('1981-01-01', '1984-12-31'),
            '1985_1988': ('1985-01-01', '1988-12-31'),
            '1989_1992': ('1989-01-01', '1992-12-31'),
            '1993_1996': ('1993-01-01', '1996-12-31'),
            '1997_2000': ('1997-01-01', '2000-12-31'),
            '2001_2004': ('2001-01-01', '2004-12-31'),
            '2005_2008': ('2005-01-01', '2008-12-31'),
            '2009_2012': ('2009-01-01', '2012-12-31'),
            '2013_2016': ('2013-01-01', '2016-12-31'),
            '2017_2020': ('2017-01-01', '2020-12-31'),
        }
        
        for period, (start_date, end_date) in period_ranges.items():
            period_df = original_df[
                (original_df['date'] >= start_date) & 
                (original_df['date'] <= end_date)
            ]
            
            if len(period_df) > 0:
                # Calculate the metric for this period
                if metric == 'home_win_prob':
                    value = (period_df['home_score'] > period_df['away_score']).mean()
                elif metric == 'draw_prob':
                    value = (period_df['home_score'] == period_df['away_score']).mean()
                elif metric == 'away_win_prob':
                    value = (period_df['home_score'] < period_df['away_score']).mean()
                elif metric == 'avg_total_goals':
                    value = (period_df['home_score'] + period_df['away_score']).mean()
                elif metric == 'avg_home_goals':
                    value = period_df['home_score'].mean()
                elif metric == 'avg_away_goals':
                    value = period_df['away_score'].mean()
                elif metric == 'home_advantage':
                    home_win = (period_df['home_score'] > period_df['away_score']).mean()
                    away_win = (period_df['home_score'] < period_df['away_score']).mean()
                    value = home_win - away_win
                else:
                    value = np.nan
                
                data.append({
                    'model': 'baseline',
                    'period': period,
                    'value': value
                })
        
        # Load model data - ACTUAL paths that exist
        model_paths = {
            'poisson': 'prop_gaussian',  # poisson is in prop_gaussian!
            'gaussian_copula': 'prop_gaussian',  # gaussian_copula is in prop_gaussian!
            'ctgan': 'prop_ctgan',
            'copulagan': 'prop_copulagan',
            'tvae': 'prop_tvae'
        }
        
        for model, dir_name in model_paths.items():
            for period in self.periods:
                json_file = self.properties_dir / dir_name / f'{model}_period_{period}_aggregate_properties.json'
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        props = json.load(f)
                        data.append({
                            'model': model,
                            'period': period,
                            'value': props.get(metric, np.nan)
                        })
        
        df = pd.DataFrame(data)
        
        # Pivot to get: rows=models, columns=periods
        if len(df) > 0:
            pivot = df.pivot(index='model', columns='period', values='value')
            # Reorder to match our model and period order
            pivot = pivot.reindex(self.models)
            pivot = pivot[self.periods]
            return pivot
        
        return pd.DataFrame()
    
    def visualize_single_model(self, model_name='gaussian_copula', display_name='GAUSSIAN_COPULA'):
        """Visualize how ONE model's properties change over time across all 16 period files"""
        
        output_dir = Path('visualizations/aggregate_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map model to directory
        model_dirs = {
            'poisson': 'prop_gaussian',
            'gaussian_copula': 'prop_gaussian',
            'ctgan': 'prop_ctgan',
            'copulagan': 'prop_copulagan',
            'tvae': 'prop_tvae'
        }
        
        scored_dir = Path('scored_matches') / model_name
        if not scored_dir.exists():
            print(f"No scored matches found for {model_name}")
            return
        
        # Load baseline - calculate per year
        print("Loading baseline data...")
        baseline_df = pd.read_csv('original_data/og/results.csv')
        baseline_df['date'] = pd.to_datetime(baseline_df['date'])
        baseline_df['year'] = baseline_df['date'].dt.year
        
        baseline_by_year = baseline_df.groupby('year', as_index=False).apply(
            lambda x: pd.Series({
                'year': x.name,
                'home_win_prob': (x['home_score'] > x['away_score']).mean()
            }), include_groups=False
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # For each period file, calculate home_win_prob by year
        print(f"Processing {model_name} period files...")
        
        # DARK, VISIBLE colors for white background
        period_colors = {
            '1957_1960': '#8B0000',  # DarkRed
            '1961_1964': '#A52A2A',  # Brown
            '1965_1968': '#DC143C',  # Crimson
            '1969_1972': '#C71585',  # MediumVioletRed
            '1973_1976': '#D2691E',  # Chocolate
            '1977_1980': '#FF4500',  # OrangeRed
            '1981_1984': '#FF6347',  # Tomato
            '1985_1988': '#CD5C5C',  # IndianRed
            '1989_1992': '#228B22',  # ForestGreen
            '1993_1996': '#2E8B57',  # SeaGreen
            '1997_2000': '#008080',  # Teal
            '2001_2004': '#4B0082',  # Indigo
            '2005_2008': '#483D8B',  # DarkSlateBlue
            '2009_2012': '#6A5ACD',  # SlateBlue
            '2013_2016': '#4169E1',  # RoyalBlue
            '2017_2020': '#00008B'   # DarkBlue
        }
        
        for idx, period in enumerate(self.periods):
            period_file = scored_dir / f'period_{period}.csv'
            if period_file.exists():
                df = pd.read_csv(period_file)
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year
                
                # Calculate home_win_prob by year for this period file (ALL data including synthetic)
                yearly_stats = df.groupby('year', as_index=False).apply(
                    lambda x: pd.Series({
                        'year': x.name,
                        'home_win_prob': (x['home_score'] > x['away_score']).mean()
                    }), include_groups=False
                )
                
                # Plot this curve with clear label using DARK color
                ax.plot(yearly_stats['year'], yearly_stats['home_win_prob'],
                       color=period_colors[period], linewidth=2, alpha=0.85,
                       label=f'{period}', zorder=2)
        
        # Plot baseline - thin gray so it doesn't obscure the data
        ax.plot(baseline_by_year['year'], baseline_by_year['home_win_prob'],
               color='gray', linewidth=1, linestyle='--', label='Baseline (historical)',
               alpha=0.6, zorder=1)
        
        ax.set_title(f'{display_name}: Home Win Probability Over Time (16 Period Files)\n' +
                    'Solid lines = model with synthetic data | Dashed black = baseline',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Home Win Probability', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_temporal_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_dir}/{model_name}_temporal_evolution.png")
    
    def visualize_single_metric(self, metric='home_win_prob', metric_name='Home Win Probability'):
        """Create line plot showing how metric changes as synthetic data accumulates"""
        data_matrix = self._load_data_for_metric(metric)
        
        if data_matrix.empty:
            print(f"No data found for {metric}")
            return
        
        output_dir = Path('visualizations/aggregate_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create line plot
        fig, ax = plt.subplots(figsize=(18, 10))
        
        x = range(len(self.periods))
        
        # Plot baseline (historical values per period)
        if 'baseline' in data_matrix.index:
            baseline_values = data_matrix.loc['baseline'].values
            ax.plot(x, baseline_values, 
                   color=self.colors['baseline'], marker='o', 
                   linewidth=3, markersize=8, label='Baseline (historical)', 
                   alpha=0.9, zorder=10)
        
        # Plot each model (cumulative: original + synthetic)
        for model in ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']:
            if model in data_matrix.index:
                model_values = data_matrix.loc[model].values
                ax.plot(x, model_values,
                       color=self.colors[model], marker='o', 
                       linewidth=2.5, markersize=6, label=model, alpha=0.8)
        
        ax.set_title(f'{metric_name}: How Synthetic Data Affects Total Dataset Properties\n' +
                    'Left = most synthetic data added | Right = least synthetic data added',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Period (accumulating synthetic data)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.periods, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {output_dir}/")
        print(f"  - {metric}_comparison.png")


def main():
    print("Creating aggregate property visualizations...")
    visualizer = AggregateVisualizer()
    
    # Visualize one model at a time to see how synthetic data amount affects properties
    print("\nGenerating visualizations for CTGAN model...")
    visualizer.visualize_single_model('poisson', 'POISSON')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
