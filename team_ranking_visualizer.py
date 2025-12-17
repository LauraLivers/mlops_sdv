import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class TeamRankingVisualizer:
    def __init__(self):
        self.models = ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']
        self.european_countries = {
            'Germany', 'England', 'France', 'Italy', 'Spain', 'Netherlands', 'Portugal',
            'Belgium', 'Russia', 'Ukraine', 'Poland', 'Romania', 'Czech Republic',
            'Greece', 'Sweden', 'Austria', 'Switzerland', 'Denmark', 'Norway',
            'Finland', 'Scotland', 'Ireland', 'Wales', 'Northern Ireland', 'Iceland',
            'Croatia', 'Serbia', 'Slovenia', 'Slovakia', 'Hungary', 'Bulgaria',
            'Albania', 'North Macedonia', 'Bosnia-Herzegovina', 'Montenegro',
            'Belarus', 'Estonia', 'Latvia', 'Lithuania', 'Moldova', 'Armenia',
            'Georgia', 'Azerbaijan', 'Kazakhstan', 'Turkey', 'Cyprus', 'Malta',
            'Luxembourg', 'Liechtenstein', 'Andorra', 'Monaco', 'San Marino',
            'Faroe Islands', 'Gibraltar', 'Kosovo', 'FR Yugoslawia', 'West-Germany', 
            'Czechoslovakia', 
        }
        self.country_colors = self._assign_country_colors()
    
    def _assign_country_colors(self):
        countries = sorted(self.european_countries)
        color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
        color_palette_2 = plt.cm.tab20b(np.linspace(0, 1, 20))
        color_palette_3 = plt.cm.tab20c(np.linspace(0, 1, 20))
        all_colors = list(color_palette) + list(color_palette_2) + list(color_palette_3)
        
        country_colors = {}
        for i, country in enumerate(countries):
            if country == 'Switzerland':
                country_colors[country] = 'red'
            else:
                country_colors[country] = all_colors[i % len(all_colors)]
        return country_colors
    
    def calculate_team_rankings(self, file_path):
        df = pd.read_csv(file_path)
        total_games = len(df)
        
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        team_stats = {}
        
        for team in teams:
            points = 0
            goals_for = 0
            goals_against = 0
            
            home_matches = df[df['home_team'] == team]
            for _, match in home_matches.iterrows():
                home_score = match['home_score']
                away_score = match['away_score']
                
                if pd.notna(home_score) and pd.notna(away_score):
                    goals_for += home_score
                    goals_against += away_score
                    
                    if home_score > away_score:
                        points += 3
                    elif home_score == away_score:
                        points += 1
            
            away_matches = df[df['away_team'] == team]
            for _, match in away_matches.iterrows():
                home_score = match['home_score']
                away_score = match['away_score']
                
                if pd.notna(home_score) and pd.notna(away_score):
                    goals_for += away_score
                    goals_against += home_score
                    
                    if away_score > home_score:
                        points += 3
                    elif home_score == away_score:
                        points += 1
            
            goal_diff = goals_for - goals_against
            team_stats[team] = {
                'points': points,
                'goal_diff': goal_diff,
                'goals_for': goals_for,
                'goals_against': goals_against
            }
        
        rankings = sorted(team_stats.items(), 
                         key=lambda x: (x[1]['points'], x[1]['goal_diff']), 
                         reverse=True)
        
        european_rankings = [(team, stats) for team, stats in rankings 
                            if team in self.european_countries]
        
        return european_rankings, total_games
    
    def create_ranking_visualization(self, rankings, model, period, total_games, top_n=30):
        top_teams = rankings[:top_n]
        
        teams = [team for team, _ in top_teams]
        points = [stats['points'] for _, stats in top_teams]
        goal_diffs = [stats['goal_diff'] for _, stats in top_teams]
        
        fig, ax = plt.subplots(figsize=(16, max(12, len(top_teams) * 0.45)))
        
        y_positions = np.arange(len(teams))
        
        colors = [self.country_colors[team] for team in teams]
        alphas = [1.0 if team == 'Switzerland' else 0.5 for team in teams]
        
        for i, (y_pos, point, color, alpha) in enumerate(zip(y_positions, points, colors, alphas)):
            ax.barh(y_pos, point, color=color, alpha=alpha, edgecolor='black', linewidth=1.5)
        
        max_points = max(points) if points else 1
        for i, (team, stats) in enumerate(top_teams):
            ax.text(stats['points'] - max_points * 0.02, i, 
                   f"{int(stats['points'])} pts (GD: {int(stats['goal_diff']):+d})  ",
                   va='center', ha='right', fontsize=10, fontweight='bold', color='white')
        
        for i, team in enumerate(teams):
            ax.text(-max_points * 0.01, i, team,
                   va='center', ha='right', fontsize=11, fontweight='bold', color='black')
        
        ax.set_yticks([])
        ax.set_xticks([])
        ax.invert_yaxis()
        
        ax.set_title(f'Team Rankings: {model.upper()} - {period}',
                    fontsize=15, fontweight='bold', pad=20)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.set_xlim(0, max_points * 1.05)
        ax.set_ylim(len(teams) - 0.5, -0.5)
        
        ax.text(max_points * 0.97, len(teams) - 1, f'{total_games:,}',
               fontsize=160, fontweight='bold',
               ha='right', va='bottom', color='black', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_all_files(self):
        output_dir = Path('visualizations/team_rankings')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nProcessing baseline...")
        baseline_file = Path('original_data/og/results.csv')
        if baseline_file.exists():
            rankings, total_games = self.calculate_team_rankings(baseline_file)
            if rankings:
                fig = self.create_ranking_visualization(rankings, 'baseline', 'overall', total_games)
                output_file = output_dir / f'baseline_overall_rankings.png'
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: baseline overall ranking")
        
        for model in self.models:
            model_dir = Path('scored_matches') / model
            if not model_dir.exists():
                print(f"Skipping {model} - no scored matches found")
                continue
            
            print(f"\nProcessing {model}...")
            
            period_files = list(model_dir.glob('period_*.csv'))
            
            for period_file in period_files:
                period_name = period_file.stem.replace('period_', '')
                rankings, total_games = self.calculate_team_rankings(period_file)
                fig = self.create_ranking_visualization(rankings, model, period_name, total_games)
                output_file = output_dir / f'{model}_{period_name}_rankings.png'
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {period_name}")
        
        print(f"\nAll rankings saved to {output_dir}/")


def main():
    print("Creating team ranking visualizations...\n")
    visualizer = TeamRankingVisualizer()
    visualizer.visualize_all_files()
    print("\nDone!")


if __name__ == '__main__':
    main()
