import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ModelComparisonScatter:
    def __init__(self):
        self.models = ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']
        self.periods = [
            '1957_1960', '1961_1964', '1965_1968', '1969_1972',
            '1973_1976', '1977_1980', '1981_1984', '1985_1988',
            '1989_1992', '1993_1996', '1997_2000', '2001_2004',
            '2005_2008', '2009_2012', '2013_2016', '2017_2020'
        ]
        
        self.metrics = {
            'avg_total_goals': 'Average Total Goals per Match',
            'avg_home_goals': 'Average Home Goals per Match',
            'avg_away_goals': 'Average Away Goals per Match',
            'home_advantage': 'Home Advantage (Home Win % - Away Win %)'
        }
        
        self.model_colors = {
            'poisson': '#2C3E50',
            'ctgan': '#E74C3C',
            'gaussian_copula': '#3498DB',
            'copulagan': '#F39C12',
            'tvae': '#9B59B6'
        }
    
    def calculate_baseline_entire_file(self):
        baseline_df = pd.read_csv('original_data/og/results.csv')
        
        return {
            'avg_total_goals': (baseline_df['home_score'] + baseline_df['away_score']).mean(),
            'avg_home_goals': baseline_df['home_score'].mean(),
            'avg_away_goals': baseline_df['away_score'].mean(),
            'home_advantage': (
                (baseline_df['home_score'] > baseline_df['away_score']).mean() -
                (baseline_df['home_score'] < baseline_df['away_score']).mean()
            )
        }
    
    def calculate_model_entire_file(self, model_name, period):
        scored_file = Path('scored_matches') / model_name / f'period_{period}.csv'
        
        if not scored_file.exists():
            return None
        
        df = pd.read_csv(scored_file)
        
        return {
            'avg_total_goals': (df['home_score'] + df['away_score']).mean(),
            'avg_home_goals': df['home_score'].mean(),
            'avg_away_goals': df['away_score'].mean(),
            'home_advantage': (
                (df['home_score'] > df['away_score']).mean() -
                (df['home_score'] < df['away_score']).mean()
            )
        }
    
    def create_scatter_comparison(self):
        output_dir = Path('visualizations/model_comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_entire = self.calculate_baseline_entire_file()
        
        for metric_key, metric_name in self.metrics.items():
            print(f"Creating scatter plot for {metric_name}...")
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            baseline_value = baseline_entire[metric_key]
            
            all_values = []
            y_offset = 0
            
            for model in self.models:
                model_values = []
                
                for period_idx, period in enumerate(self.periods):
                    model_vals = self.calculate_model_entire_file(model, period)
                    
                    if model_vals is not None:
                        model_values.append(model_vals[metric_key])
                
                y_positions = [y_offset + i for i in range(len(model_values))]
                
                for i, (val, y_pos) in enumerate(zip(model_values, y_positions)):
                    alpha = 0.3 + (i / len(model_values)) * 0.7
                    
                    ax.scatter([val], [y_pos], 
                              color=self.model_colors[model], s=200, alpha=alpha,
                              edgecolors='black', linewidth=2)
                
                if len(model_values) > 0:
                    ax.scatter([], [], color=self.model_colors[model], s=200, alpha=0.7,
                              edgecolors='black', linewidth=2, label=model.upper())
                
                all_values.extend(model_values)
                y_offset += len(model_values) + 2
            
            all_values.append(baseline_value)
            
            ax.axvline(x=baseline_value, color='red', linewidth=3, 
                      linestyle=':', label='BASELINE', zorder=5, alpha=0.9)
            
            if len(all_values) > 0:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.05
                ax.set_xlim(min_val - margin, max_val + margin)
            
            ax.set_ylim(-1, y_offset)
            
            ax.set_xlabel(metric_name, fontsize=14, fontweight='bold')
            ax.set_ylabel('Files', fontsize=14, fontweight='bold')
            ax.set_title(f'{metric_name}', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':', axis='x')
            ax.legend(loc='best', fontsize=12)
            ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric_key}_scatter.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {output_dir}/{metric_key}_scatter.png")


def main():
    print("Creating model comparison scatter plots...\n")
    visualizer = ModelComparisonScatter()
    visualizer.create_scatter_comparison()
    print("\nAll scatter plots complete!")


if __name__ == '__main__':
    main()
