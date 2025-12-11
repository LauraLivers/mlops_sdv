import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from baseline_period_finder import BaselinePeriodFinder
from pathlib import Path

class MatchDensityVisualizer:
    def __init__(self, results_path):
        self.results = pd.read_csv(results_path)
        self.results['date'] = pd.to_datetime(self.results['date'])
        self.results['year'] = self.results['date'].dt.year
        
        finder = BaselinePeriodFinder(results_path)
        baseline_result = finder.find_optimal_baseline()
        self.baseline = baseline_result['baseline']
        print(self.baseline)
        self.baseline_total = self.baseline['total_matches']
        
    def get_matches_per_year(self):
        matches_by_year = self.results['year'].value_counts().sort_index()
        return matches_by_year
    
    def create_density_grid(self):
        years = sorted(self.results['year'].unique(), reverse=True)
        months = range(1, 13)
        
        grid_data = []
        
        for year in years:
            year_data = self.results[self.results['year'] == year]
            row = []
            
            for month in months:
                matches_this_month = len(year_data[year_data['date'].dt.month == month])
                row.append(matches_this_month)
            
            grid_data.append(row)
        
        return np.array(grid_data), years, months
    
    def create_visualization(self):
        grid_data, years, months = self.create_density_grid()
        
        n_years = len(years)
        n_months = len(months)
        
        figwidth = 20
        figheight = (n_years / n_months) * figwidth + 2
        
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
        
        sns.heatmap(
            grid_data,
            cmap='viridis',
            cbar=False,
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            yticklabels=False,
            square=False
        )
        
        baseline_start_idx = years.index(self.baseline['end_year'])
        baseline_end_idx = years.index(self.baseline['start_year'])
        
        ax.add_patch(plt.Rectangle(
            (0, baseline_start_idx),
            12,
            baseline_end_idx - baseline_start_idx + 1,
            fill=False,
            edgecolor='red',
            linewidth=6
        ))
        
        ax.set_xlabel('Month', fontsize=18, labelpad=15)
        ax.set_ylabel('Year (newest to oldest)', fontsize=18, labelpad=15)
        ax.set_title(
            f"Women's Football Match Density Over Time\nBaseline: {self.baseline['start_year'] + 1}-{self.baseline['end_year']-1}",
            fontsize=68,
            pad=60,
            fontweight='bold'
        )
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, output_filename='match_density.png'):
        fig = self.create_visualization()
        fig.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_filename

if __name__ == '__main__':
    augmented_dir = Path('augmented_periods')
    csv_files = list(augmented_dir.glob('*.csv'))
    
    for idx, csv_path in enumerate(csv_files, start=1):
        visualizer = MatchDensityVisualizer(str(csv_path))
        output_file = f'match_density_{idx}.png'
        visualizer.save_visualization(output_file)
        print(f"Visualization {idx} saved to: {output_file}")