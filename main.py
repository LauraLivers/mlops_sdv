import pandas as pd
from football_data_loader import FootballDataLoader
from match_predictor import MatchPredictor
from sdv_contaminator import SDVContaminator
from stress_test_runner import StressTestRunner
from visualizer import StressTestVisualizer

CONFIG = {
    'results_path': 'results.csv',
    'goalscorers_path': 'goalscorers.csv',
    'shootouts_path': 'shootouts.csv',
    'test_season_start': '2024-07-01',
    'test_season_end': '2025-06-30',
    'contamination_levels': [0, 25, 50, 75, 100],
    'contamination_types': ['matches', 'season', 'teams', 'league'],
    'synthesizers': ['ctgan', 'gaussian'],
    'random_state': 42,
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'skew_targets': [
        {'property': 'home_win_pct', 'value': 60.0},
        {'property': 'avg_goals', 'value': 3.5}
    ],
    'output_path': 'stress_test_results.json',
    'output_dir': 'stress_test_output'
}

if __name__ == '__main__':
    loader = FootballDataLoader(
        CONFIG['results_path'],
        CONFIG['goalscorers_path'],
        CONFIG['shootouts_path']
    )
    
    train_data, test_data = loader.split_by_season(
        CONFIG['test_season_start'],
        CONFIG['test_season_end']
    )
    
    runner = StressTestRunner(train_data, test_data, CONFIG)
    runner.run_preservation_test()
    runner.run_skewing_test()
    runner.save_results(CONFIG['output_path'])
    
    visualizer = StressTestVisualizer(CONFIG['output_path'])
    tables = visualizer.generate_all_tables()