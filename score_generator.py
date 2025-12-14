import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

if torch.backends.mps.is_available():
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    use_cuda = False
elif torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


class ScoreGenerator:
    def __init__(self, historical_data_path='original_data/results.csv'):
        df = pd.read_csv(historical_data_path)
        self.team_stats = self._calculate_team_stats(df)
        self.stage_stats = self._calculate_stage_stats(df)
        self.training_data = self._prepare_training_data(df)
        self.trained_models = {}
    
    def _calculate_team_stats(self, df):
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        stats = []
        
        for team in all_teams:
            home = df[df['home_team'] == team]
            away = df[df['away_team'] == team]
            total = len(home) + len(away)
            
            if total == 0:
                continue
            
            gf = home['home_score'].sum() + away['away_score'].sum()
            ga = home['away_score'].sum() + away['home_score'].sum()
            
            stats.append({'team': team, 'strength': (gf - ga) / total})
        
        return pd.DataFrame(stats)
    
    def _calculate_stage_stats(self, df):
        if 'stage' not in df.columns:
            return {}
        
        stats = {}
        for stage in df['stage'].dropna().unique():
            stage_df = df[df['stage'] == stage]
            stats[stage] = {
                'avg_goals': (stage_df['home_score'] + stage_df['away_score']).mean(),
                'draw_rate': (stage_df['home_score'] == stage_df['away_score']).mean()
            }
        return stats
    
    def _prepare_training_data(self, df):
        team_dict = self.team_stats.set_index('team')['strength'].to_dict()
        
        df['home_strength'] = df['home_team'].map(lambda x: team_dict.get(x, 0.0))
        df['away_strength'] = df['away_team'].map(lambda x: team_dict.get(x, 0.0))
        df['is_neutral'] = df['neutral'].astype(int)
        
        if 'stage' not in df.columns:
            df['stage'] = 'group'
        
        return df[['home_score', 'away_score', 'is_neutral', 'stage', 'home_strength', 'away_strength']].copy()
    
    def _create_metadata(self):
        metadata = Metadata.detect_from_dataframe(self.training_data)
        return metadata
    
    def _generate_scores(self, matches_df, stage, model_name):
        team_dict = self.team_stats.set_index('team')['strength'].to_dict()
        
        matches_df['home_strength'] = matches_df['home_team'].map(lambda x: team_dict.get(x, 0.0))
        matches_df['away_strength'] = matches_df['away_team'].map(lambda x: team_dict.get(x, 0.0))
        
        if model_name == 'poisson':
            return self._generate_scores_fallback(matches_df, stage)
        
        matches_df['is_neutral'] = matches_df['neutral'].astype(int)
        
        if model_name not in self.trained_models:
            metadata = self._create_metadata()
            
            if model_name == 'ctgan':
                model = CTGANSynthesizer(metadata, epochs=100, verbose=False, cuda=use_cuda)
            elif model_name == 'gaussian_copula':
                model = GaussianCopulaSynthesizer(metadata)
            elif model_name == 'copulagan':
                model = CopulaGANSynthesizer(metadata, epochs=100, verbose=False, cuda=use_cuda)
            elif model_name == 'tvae':
                model = TVAESynthesizer(metadata, epochs=100, verbose=False, cuda=use_cuda)
            
            model.fit(self.training_data)
            self.trained_models[model_name] = model
        
        model = self.trained_models[model_name]
        generated = model.sample(len(matches_df))
        
        matches_df['home_score'] = generated['home_score'].clip(lower=0).round().astype(int).clip(upper=10).values
        matches_df['away_score'] = generated['away_score'].clip(lower=0).round().astype(int).clip(upper=10).values
        
        if stage in ['final', 'semi_final', 'quarter_final', 'round_of_16', 'third_place']:
            for idx in matches_df.index:
                if matches_df.loc[idx, 'home_score'] == matches_df.loc[idx, 'away_score']:
                    home_str = matches_df.loc[idx, 'home_strength']
                    away_str = matches_df.loc[idx, 'away_strength']
                    if home_str >= away_str:
                        matches_df.loc[idx, 'home_score'] += 1
                    else:
                        matches_df.loc[idx, 'away_score'] += 1
        
        matches_df = matches_df.drop(columns=['home_strength', 'away_strength', 'is_neutral'])
        return matches_df
    
    def _generate_scores_fallback(self, matches_df, stage):
        if stage in self.stage_stats:
            base_goals = self.stage_stats[stage]['avg_goals'] / 2
        else:
            base_goals = 1.3 if stage in ['final', 'semi_final'] else 1.5
        
        for idx in matches_df.index:
            home_str = matches_df.loc[idx, 'home_strength']
            away_str = matches_df.loc[idx, 'away_strength']
            is_neutral = matches_df.loc[idx, 'neutral']
            
            home_lambda = max(0.3, base_goals + home_str * 0.4 - away_str * 0.2)
            away_lambda = max(0.3, base_goals + away_str * 0.4 - home_str * 0.2)
            
            if not is_neutral:
                home_lambda *= 1.15
                away_lambda *= 0.92
            
            matches_df.loc[idx, 'home_score'] = min(10, np.random.poisson(home_lambda))
            matches_df.loc[idx, 'away_score'] = min(10, np.random.poisson(away_lambda))
            
            if stage in ['final', 'semi_final', 'quarter_final', 'round_of_16', 'third_place']:
                if matches_df.loc[idx, 'home_score'] == matches_df.loc[idx, 'away_score']:
                    if home_str >= away_str:
                        matches_df.loc[idx, 'home_score'] += 1
                    else:
                        matches_df.loc[idx, 'away_score'] += 1
        
        matches_df = matches_df.drop(columns=['home_strength', 'away_strength'])
        return matches_df
    
    def _process_tournament(self, tournament_df, model_name):
        tournament_df = tournament_df.sort_values('date', ascending=False).copy()
        
        stages_order = ['final', 'third_place', 'semi_final', 'quarter_final', 'round_of_16', 'group']
        
        for stage in stages_order:
            stage_matches = tournament_df[tournament_df['stage'] == stage].copy()
            if len(stage_matches) == 0:
                continue
            
            stage_matches = self._generate_scores(stage_matches, stage, model_name)
            
            for idx in stage_matches.index:
                tournament_df.loc[idx, 'home_score'] = stage_matches.loc[idx, 'home_score']
                tournament_df.loc[idx, 'away_score'] = stage_matches.loc[idx, 'away_score']
        
        return tournament_df
    
    def generate_scores(self, df, model_name):
        synthetic_mask = (df['is_synthetic'] == True) & df['home_score'].isna()
        needs_scoring = df[synthetic_mask].copy()
        
        if len(needs_scoring) == 0:
            return df
        
        tournaments = needs_scoring.groupby(['tournament', needs_scoring['date'].apply(lambda x: pd.to_datetime(x).year)])
        
        for (tournament, year), tournament_df in tqdm(tournaments, desc="    ", leave=False):
            scored = self._process_tournament(tournament_df, model_name)
            for idx in scored.index:
                df.loc[idx, 'home_score'] = scored.loc[idx, 'home_score']
                df.loc[idx, 'away_score'] = scored.loc[idx, 'away_score']
        
        return df
    
    def process_all_periods(self):
        periods = [
            '2017_2020', '2013_2016', '2009_2012', '2005_2008',
            '2001_2004', '1997_2000', '1993_1996', '1989_1992',
            '1985_1988', '1981_1984', '1977_1980', '1973_1976',
            '1969_1972', '1965_1968', '1961_1964', '1957_1960'
        ]
        
        for model in ['poisson', 'ctgan', 'gaussian_copula', 'copulagan', 'tvae']:
            output_dir = Path(f'scored_matches/{model}')
            
            if output_dir.exists() and len(list(output_dir.glob('*.csv'))) == len(periods):
                continue
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for period in tqdm(periods, desc=f"{model:20s}"):
                self.trained_models = {}
                input_file = f'filled_matches/period_{period}.csv'
                df = pd.read_csv(input_file)
                df['date'] = pd.to_datetime(df['date'])
                
                scored_df = self.generate_scores(df, model)
                
                output_file = output_dir / f'period_{period}.csv'
                scored_df.to_csv(output_file, index=False)


def main():
    generator = ScoreGenerator()
    generator.process_all_periods()


if __name__ == "__main__":
    main()
