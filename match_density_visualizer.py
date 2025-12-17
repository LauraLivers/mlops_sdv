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
        
        figwidth = 23
        cell_height = 0.5  # Very short cells
        figheight = n_years * cell_height
        
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
        
        ax.set_aspect('auto')
        
        from matplotlib.colors import LinearSegmentedColormap
        single_color = LinearSegmentedColormap.from_list('single_pink', ['white', '#E91E63'])
        
        sns.heatmap(
            grid_data,
            cmap=single_color,
            cbar=False,
            linewidths=0.3,
            linecolor='lightgray',
            ax=ax,
            xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            yticklabels=years,
            square=False,
            vmin=0
        )
        
        # # Find baseline indices - years are sorted newest to oldest (descending)
        # baseline_start_idx = years.index(self.baseline['start_year'] - 1)
        # baseline_end_idx = years.index(self.baseline['end_year'] + 1)
        
        # # Rectangle: (x, y, width, height) where y is row index
        # ax.add_patch(plt.Rectangle(
        #     (0, baseline_start_idx),
        #     12,
        #     baseline_end_idx - baseline_start_idx + 1,
        #     fill=False,
        #     edgecolor='red',
        #     linewidth=6
        # ))
        
        ax.set_xlabel('Month', fontsize=18, labelpad=15)
        ax.set_ylabel('Year (newest to oldest)', fontsize=18, labelpad=15)
        ax.set_title(
            f"Women's Football Match Density Over Time",
            fontsize=58,
            pad=60,
            fontweight='bold'
        )
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, output_filename='match_density.png'):
        Path('visualizations/match_density').mkdir(parents=True, exist_ok=True)
        output_path = Path('visualizations/match_density') / output_filename
        fig = self.create_visualization()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(output_path)

if __name__ == '__main__':
    augmented_dir = Path('original_data')
    csv_files = sorted(list(augmented_dir.glob('*.csv')), reverse=True)
    
    for idx, csv_path in enumerate(csv_files, start=1):
        visualizer = MatchDensityVisualizer(str(csv_path))
        output_file = f'match_density_{csv_path.stem}.png'
        saved_path = visualizer.save_visualization(output_file)
        print(f"Visualization {idx} saved to: {saved_path}")