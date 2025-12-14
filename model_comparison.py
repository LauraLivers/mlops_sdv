import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ModelComparison:
    def __init__(self):
        self.historical = pd.read_csv('original_data/results.csv')
        self.models = ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']
        self.periods = [
            '2017_2020', '2013_2016', '2009_2012', '2005_2008',
            '2001_2004', '1997_2000', '1993_1996', '1989_1992',
            '1985_1988', '1981_1984', '1977_1980', '1973_1976',
            '1969_1972', '1965_1968', '1961_1964', '1957_1960'
        ]
        self.baseline_team_stats = self._calculate_team_stats(self.historical)
    
    def _calculate_team_stats(self, df):
        """Calculate per-team statistics from a dataframe"""
        team_stats = {}
        
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for team in all_teams:
            home_matches = df[df['home_team'] == team]
            away_matches = df[df['away_team'] == team]
            
            if len(home_matches) == 0 and len(away_matches) == 0:
                continue
            
            home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
            away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
            home_draws = (home_matches['home_score'] == home_matches['away_score']).sum()
            away_draws = (away_matches['away_score'] == away_matches['home_score']).sum()
            
            total_matches = len(home_matches) + len(away_matches)
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            
            goals_scored = home_matches['home_score'].sum() + away_matches['away_score'].sum()
            goals_conceded = home_matches['away_score'].sum() + away_matches['home_score'].sum()
            
            team_stats[team] = {
                'matches': total_matches,
                'wins': total_wins,
                'draws': total_draws,
                'losses': total_matches - total_wins - total_draws,
                'win_rate': total_wins / total_matches if total_matches > 0 else 0,
                'draw_rate': total_draws / total_matches if total_matches > 0 else 0,
                'goals_per_match': goals_scored / total_matches if total_matches > 0 else 0,
                'conceded_per_match': goals_conceded / total_matches if total_matches > 0 else 0,
                'goal_difference': (goals_scored - goals_conceded) / total_matches if total_matches > 0 else 0
            }
        
        return team_stats
    
    def compare_all_models(self):
        """Compare all models across all periods with team-level granularity"""
        
        overall_results = []
        team_comparisons = []
        
        for period in tqdm(self.periods, desc="Processing periods"):
            for model in self.models:
                file_path = f'scored_matches/{model}/period_{period}.csv'
                
                if not Path(file_path).exists():
                    continue
                
                df = pd.read_csv(file_path)
                synthetic = df[df['is_synthetic'] == True].copy()
                
                if len(synthetic) == 0:
                    continue
                
                synthetic = synthetic.dropna(subset=['home_score', 'away_score'])
                
                if len(synthetic) == 0:
                    overall_results.append({
                        'period': period,
                        'model': model,
                        'synthetic_count': len(df[df['is_synthetic'] == True]),
                        'valid_scores': 0,
                        'missing_rate': 1.0
                    })
                    continue
                
                # Calculate overall statistics
                total_synthetic = len(df[df['is_synthetic'] == True])
                missing_rate = 1.0 - (len(synthetic) / total_synthetic)
                
                overall_stats = {
                    'period': period,
                    'model': model,
                    'synthetic_count': total_synthetic,
                    'valid_scores': len(synthetic),
                    'missing_rate': missing_rate,
                    'home_win_rate': (synthetic['home_score'] > synthetic['away_score']).mean(),
                    'draw_rate': (synthetic['home_score'] == synthetic['away_score']).mean(),
                    'avg_total_goals': (synthetic['home_score'] + synthetic['away_score']).mean(),
                }
                overall_results.append(overall_stats)
                
                # Calculate team-level statistics
                model_team_stats = self._calculate_team_stats(synthetic)
                
                for team in model_team_stats:
                    if team not in self.baseline_team_stats:
                        continue
                    
                    baseline = self.baseline_team_stats[team]
                    model_stat = model_team_stats[team]
                    
                    team_comparisons.append({
                        'period': period,
                        'model': model,
                        'team': team,
                        'baseline_matches': baseline['matches'],
                        'synthetic_matches': model_stat['matches'],
                        'baseline_win_rate': baseline['win_rate'],
                        'model_win_rate': model_stat['win_rate'],
                        'win_rate_error': abs(baseline['win_rate'] - model_stat['win_rate']),
                        'baseline_goals_per_match': baseline['goals_per_match'],
                        'model_goals_per_match': model_stat['goals_per_match'],
                        'goals_error': abs(baseline['goals_per_match'] - model_stat['goals_per_match']),
                        'baseline_goal_diff': baseline['goal_difference'],
                        'model_goal_diff': model_stat['goal_difference'],
                        'goal_diff_error': abs(baseline['goal_difference'] - model_stat['goal_difference']),
                    })
        
        return pd.DataFrame(overall_results), pd.DataFrame(team_comparisons)
    
    def save_comparison(self, overall_df, team_df):
        """Save detailed comparison results"""
        
        overall_df.to_csv('scored_matches/overall_comparison.csv', index=False)
        team_df.to_csv('scored_matches/team_level_comparison.csv', index=False)
        
        # Summary statistics
        summary = {
            'overall': {},
            'by_model': {},
            'by_team': {},
            'top_performing_models_per_team': {}
        }
        
        # Overall summary
        for model in self.models:
            model_data = overall_df[overall_df['model'] == model]
            if len(model_data) == 0:
                continue
            
            total_synthetic = model_data['synthetic_count'].sum()
            total_valid = model_data['valid_scores'].sum()
            
            summary['by_model'][model] = {
                'total_synthetic': int(total_synthetic),
                'total_valid': int(total_valid),
                'success_rate': float(total_valid / total_synthetic) if total_synthetic > 0 else 0,
                'avg_home_win_rate': float(model_data['home_win_rate'].mean()),
                'avg_draw_rate': float(model_data['draw_rate'].mean()),
                'avg_total_goals': float(model_data['avg_total_goals'].mean())
            }
        
        # Team-level summary
        if len(team_df) > 0:
            for model in self.models:
                model_team_data = team_df[team_df['model'] == model]
                if len(model_team_data) == 0:
                    continue
                
                summary['by_model'][model].update({
                    'avg_win_rate_error': float(model_team_data['win_rate_error'].mean()),
                    'avg_goals_error': float(model_team_data['goals_error'].mean()),
                    'avg_goal_diff_error': float(model_team_data['goal_diff_error'].mean()),
                    'total_team_error': float(model_team_data[['win_rate_error', 'goals_error', 'goal_diff_error']].sum(axis=1).mean())
                })
            
            # Find best model for each team
            teams = team_df['team'].unique()
            for team in teams:
                team_data = team_df[team_df['team'] == team]
                team_data = team_data.copy()
                team_data['total_error'] = team_data[['win_rate_error', 'goals_error', 'goal_diff_error']].sum(axis=1)
                
                best_model = team_data.groupby('model')['total_error'].mean().idxmin()
                worst_model = team_data.groupby('model')['total_error'].mean().idxmax()
                
                summary['by_team'][team] = {
                    'best_model': best_model,
                    'best_error': float(team_data[team_data['model'] == best_model]['total_error'].mean()),
                    'worst_model': worst_model,
                    'worst_error': float(team_data[team_data['model'] == worst_model]['total_error'].mean()),
                    'baseline_matches': int(team_data['baseline_matches'].iloc[0])
                }
        
        with open('scored_matches/comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def visualize_comparison(self, overall_df, team_df):
        """Create visualizations comparing models"""
        
        Path('visualizations/model_comparison').mkdir(parents=True, exist_ok=True)
        
        # 1. Model Success Rates
        fig, ax = plt.subplots(figsize=(12, 6))
        model_summary = []
        for model in self.models:
            model_data = overall_df[overall_df['model'] == model]
            if len(model_data) == 0:
                continue
            total = model_data['synthetic_count'].sum()
            valid = model_data['valid_scores'].sum()
            model_summary.append({
                'model': model,
                'success_rate': valid / total if total > 0 else 0
            })
        
        if model_summary:
            summary_df = pd.DataFrame(model_summary)
            bars = ax.bar(summary_df['model'], summary_df['success_rate'] * 100)
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Model Success Rate: Valid Scores Generated')
            ax.set_ylim([0, 105])
            plt.xticks(rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('visualizations/model_comparison/success_rates.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Team-level error comparison
        if len(team_df) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for model in self.models:
                model_data = team_df[team_df['model'] == model]
                if len(model_data) == 0:
                    continue
                
                axes[0].plot(range(len(self.periods)), 
                           [model_data[model_data['period'] == p]['win_rate_error'].mean() 
                            for p in self.periods], 
                           label=model, marker='o')
                
                axes[1].plot(range(len(self.periods)), 
                           [model_data[model_data['period'] == p]['goals_error'].mean() 
                            for p in self.periods], 
                           label=model, marker='o')
                
                axes[2].plot(range(len(self.periods)), 
                           [model_data[model_data['period'] == p]['goal_diff_error'].mean() 
                            for p in self.periods], 
                           label=model, marker='o')
            
            axes[0].set_title('Win Rate Error Over Time')
            axes[0].set_xlabel('Period (newest → oldest)')
            axes[0].set_ylabel('Avg Absolute Error')
            axes[0].legend()
            
            axes[1].set_title('Goals Per Match Error Over Time')
            axes[1].set_xlabel('Period (newest → oldest)')
            axes[1].set_ylabel('Avg Absolute Error')
            axes[1].legend()
            
            axes[2].set_title('Goal Difference Error Over Time')
            axes[2].set_xlabel('Period (newest → oldest)')
            axes[2].set_ylabel('Avg Absolute Error')
            axes[2].legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/model_comparison/team_level_errors.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Top 20 teams comparison
            top_teams = team_df.groupby('team')['baseline_matches'].first().nlargest(20).index
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            team_errors = []
            for team in top_teams:
                team_data = team_df[team_df['team'] == team]
                for model in self.models:
                    model_data = team_data[team_data['model'] == model]
                    if len(model_data) > 0:
                        total_error = model_data[['win_rate_error', 'goals_error', 'goal_diff_error']].sum(axis=1).mean()
                        team_errors.append({
                            'team': team,
                            'model': model,
                            'error': total_error
                        })
            
            if team_errors:
                error_df = pd.DataFrame(team_errors)
                pivot = error_df.pivot(index='team', columns='model', values='error')
                pivot.plot(kind='bar', ax=ax)
                ax.set_ylabel('Total Error (lower is better)')
                ax.set_title('Top 20 Teams: Model Accuracy Comparison')
                ax.legend(title='Model')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('visualizations/model_comparison/top_teams_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()


def main():
    comparison = ModelComparison()
    
    print("Comparing all models...")
    overall_df, team_df = comparison.compare_all_models()
    
    print("\nSaving results...")
    comparison.save_comparison(overall_df, team_df)
    
    print("\nCreating visualizations...")
    comparison.visualize_comparison(overall_df, team_df)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print("\nModel Success Rates:")
    for model in ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']:
        model_data = overall_df[overall_df['model'] == model]
        if len(model_data) == 0:
            print(f"  {model:20s}: No data")
            continue
        total = model_data['synthetic_count'].sum()
        valid = model_data['valid_scores'].sum()
        success_rate = valid / total if total > 0 else 0
        print(f"  {model:20s}: {valid:6d}/{total:6d} ({success_rate*100:5.1f}%)")
    
    if len(team_df) > 0:
        print("\nAverage Team-Level Errors (across all teams):")
        for model in ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']:
            model_data = team_df[team_df['model'] == model]
            if len(model_data) == 0:
                print(f"  {model:20s}: No data")
                continue
            
            win_err = model_data['win_rate_error'].mean()
            goals_err = model_data['goals_error'].mean()
            gd_err = model_data['goal_diff_error'].mean()
            total_err = win_err + goals_err + gd_err
            
            print(f"  {model:20s}: {total_err:.4f} (win: {win_err:.4f}, goals: {goals_err:.4f}, gd: {gd_err:.4f})")
        
        # Show which teams each model performs best on
        print("\nBest Model Per Team (top 10 most common):")
        teams = team_df['team'].unique()
        best_models = {}
        for team in teams:
            team_data = team_df[team_df['team'] == team].copy()
            team_data['total_error'] = team_data[['win_rate_error', 'goals_error', 'goal_diff_error']].sum(axis=1)
            best_model = team_data.groupby('model')['total_error'].mean().idxmin()
            best_models[team] = best_model
        
        from collections import Counter
        model_counts = Counter(best_models.values())
        for model, count in model_counts.most_common():
            print(f"  {model:20s}: best for {count:3d} teams")
    
    print("\nResults saved to:")
    print("  - scored_matches/overall_comparison.csv")
    print("  - scored_matches/team_level_comparison.csv")
    print("  - scored_matches/comparison_summary.json")
    print("  - visualizations/model_comparison/")


if __name__ == '__main__':
    main()
