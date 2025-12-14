import pandas as pd
import json
import os
import numpy as np
from datetime import timedelta
from baseline_period_finder import BaselinePeriodFinder

def generate_tournament_years(start_year, interval, first_known, last_year=2024):
    years = list(range(first_known, last_year + 1, interval))
    backward_years = list(range(first_known - interval, start_year - 1, -interval))
    return sorted(backward_years + years)

TOURNAMENT_SCHEDULE = {
    'FIFA World Cup': generate_tournament_years(1956, 4, 1991),
    'Olympic Games': generate_tournament_years(1956, 4, 1996),
    'UEFA Euro': generate_tournament_years(1956, 4, 1984),
    'Copa América': generate_tournament_years(1956, 4, 1991),
    'AFC Asian Cup': generate_tournament_years(1956, 4, 1991),
    'African Cup of Nations': generate_tournament_years(1956, 2, 1991),
    'CONCACAF Championship': generate_tournament_years(1956, 4, 1991),
    'CONCACAF Gold Cup': generate_tournament_years(1956, 2, 1991),
    'OFC Nations Cup': generate_tournament_years(1956, 4, 1991),
    'COSAFA Championship': list(range(1997, 2025)),
    'AFF Championship': generate_tournament_years(1956, 2, 1996),
    'SAFF Championship': generate_tournament_years(1956, 2, 1991),
    'Algarve Cup': list(range(1994, 2025)),
    'SheBelieves Cup': list(range(2016, 2025)),
    'Cyprus Cup': list(range(2008, 2023)),
    'Arnold Clark Cup': list(range(2022, 2025)),
    'Tournament of Nations': [2017, 2018],
    'Tournoi de France': [1997, 1998, 2019, 2023]
}

class HistoricalMatchFiller:
    def __init__(self, results_path, target_year=2025):
        self.results = pd.read_csv(results_path)
        self.results = self.results[self.results['tournament'] != 'CONIFA World Cup']
        self.results['date'] = pd.to_datetime(self.results['date'])
        
        # Add stage labels to historical data if not present
        if 'stage' not in self.results.columns or self.results['stage'].isna().any():
            print("Adding stage labels to historical data...")
            from historical_stage_labeler import HistoricalStageLabeler
            labeler = HistoricalStageLabeler(results_path)
            labeler.load_data()
            labeler.label_all_tournaments()
            labeler.save_labeled_data()
            # Reload with stages
            self.results = pd.read_csv(results_path)
            self.results = self.results[self.results['tournament'] != 'CONIFA World Cup']
            self.results['date'] = pd.to_datetime(self.results['date'])
            print(f"✓ Stage labels added to {self.results['stage'].notna().sum():,} matches")
        
        self.team_country = pd.read_csv('reference_tables/team_country.csv', header=None, names=['team', 'country'])
        self.team_tournaments = pd.read_csv('reference_tables/team_tournaments.csv', header=None, names=['team', 'tournament'])
        self.city_country = pd.read_csv('reference_tables/city_country.csv', header=None, names=['city', 'country'])
        
        finder = BaselinePeriodFinder(results_path, target_year)
        baseline_result = finder.find_optimal_baseline()    
        self.baseline = baseline_result['baseline']
        self.tournament_periods = baseline_result['tournament_periods']
        self.typical_tournament_window = baseline_result['typical_tournament_window']
        self.qualifier_patterns = baseline_result['qualifier_patterns']
        
        self.baseline_data = self.results[
            (self.results['date'].dt.year >= self.baseline['start_year']) &
            (self.results['date'].dt.year <= self.baseline['end_year'])
        ].copy()
    
    def _get_tournament_years(self, tournament, period_start, period_end):
        if tournament in TOURNAMENT_SCHEDULE:
            return [year for year in TOURNAMENT_SCHEDULE[tournament] 
                    if period_start <= year <= period_end]
        elif 'qualification' in tournament.lower() or 'qualifying' in tournament.lower():
            base_tournament = tournament.replace(' qualification', '').replace(' qualifying', '').replace(' qualifyication', '')
            if base_tournament in TOURNAMENT_SCHEDULE:
                tournament_years = [year for year in TOURNAMENT_SCHEDULE[base_tournament] 
                                  if year > period_end]
                if tournament_years:
                    next_tournament = min(tournament_years)
                    return list(range(max(period_start, next_tournament - 3), min(period_end + 1, next_tournament)))
            return list(range(period_start, period_end + 1))
        else:
            return list(range(period_start, period_end + 1))
    
    def _analyze_patterns(self, training_data):
        training_years = training_data['date'].dt.year.max() - training_data['date'].dt.year.min() + 1
        
        tournament_info = {}
        for tournament in training_data['tournament'].unique():
            tournament_matches = training_data[training_data['tournament'] == tournament]
            years_active = tournament_matches['date'].dt.year.nunique()
            
            # Get ALL teams that EVER participated in this tournament in training data
            teams_in_tournament = pd.concat([
                tournament_matches['home_team'], 
                tournament_matches['away_team']
            ]).unique().tolist()
            
            # Analyze WHEN this specific tournament typically occurs
            typical_months = tournament_matches['date'].dt.month.mode().tolist()
            if len(typical_months) == 0:
                typical_months = [6]  # Default to June if no data
            
            # Get typical start and duration for THIS tournament
            if len(tournament_matches) > 0:
                dates_sorted = tournament_matches['date'].sort_values()
                typical_start_month = int(dates_sorted.iloc[0].month)
                typical_start_day = int(dates_sorted.iloc[0].day)
                if len(dates_sorted) > 1:
                    typical_duration = (dates_sorted.iloc[-1] - dates_sorted.iloc[0]).days
                else:
                    typical_duration = 30
            else:
                typical_start_month = typical_months[0]
                typical_start_day = 15
                typical_duration = 30
            
            # Analyze tournament structure if it's a dense tournament
            has_structure = False
            if 'qualification' not in tournament.lower() and 'qualifying' not in tournament.lower() and 'friendly' not in tournament.lower():
                # Estimate tournament structure based on number of teams and matches
                num_teams = len(teams_in_tournament)
                avg_matches = len(tournament_matches) // max(years_active, 1)
                
                # Rough heuristic: tournaments with many teams relative to matches have knockout stages
                if num_teams >= 8 and avg_matches >= 15:
                    has_structure = True
            
            tournament_info[tournament] = {
                'total_matches': len(tournament_matches),
                'matches_per_occurrence': len(tournament_matches) // max(years_active, 1),
                'teams': teams_in_tournament,  # All eligible teams
                'locations': tournament_matches[['city', 'country']].drop_duplicates().to_dict('records'),
                'is_dense': 'qualification' not in tournament.lower() and 'qualifying' not in tournament.lower() and 'friendly' not in tournament.lower(),
                'has_structure': has_structure,
                'typical_months': typical_months,
                'typical_start_month': typical_start_month,
                'typical_start_day': min(28, typical_start_day),  # Cap at 28 to avoid month-end issues
                'typical_duration': min(60, typical_duration)  # Cap at 60 days
            }
        
        team_stats = {}
        teams = pd.concat([training_data['home_team'], training_data['away_team']]).unique()
        for team in teams:
            team_matches = training_data[(training_data['home_team'] == team) | (training_data['away_team'] == team)]
            team_stats[team] = {
                'matches_per_year': len(team_matches) / training_years,
                'tournaments': team_matches['tournament'].unique().tolist(),
                'total_matches': len(team_matches)
            }
        
        return tournament_info, team_stats

    def _get_dense_tournament_dates(self, year, tournament, tournament_info):
        """Generate dates for dense tournaments based on THIS tournament's historical patterns."""
        # Use THIS tournament's specific timing, not global window
        if tournament in tournament_info:
            info = tournament_info[tournament]
            start_month = info['typical_start_month']
            start_day = info['typical_start_day']
            duration = info['typical_duration']
        else:
            # Fallback: use typical tournament window
            window = self.typical_tournament_window
            start_month = window['typical_start_month']
            start_day = window['typical_start_day']
            duration = window['typical_duration_days']
        
        start_date = pd.Timestamp(year=year, month=start_month, day=start_day)
        end_date = start_date + timedelta(days=duration)
        
        # Generate dates with typical match frequency (every 2-4 days for group stage, daily for knockouts)
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            # Tournaments often have multiple matches per day
            current_date += timedelta(days=int(np.random.choice([2, 3, 4])))
        
        # Add more dates for larger tournaments (multiple matches same day)
        extended_dates = []
        for date in dates:
            extended_dates.append(date)
            # Add same day for parallel matches
            extended_dates.append(date)
            extended_dates.append(date)
        
        return extended_dates
    
    def _get_qualifier_dates(self, year, num_matches):
        """Generate scattered dates for qualifiers/friendlies based on historical patterns."""
        if self.qualifier_patterns and 'monthly_distribution' in self.qualifier_patterns:
            # Use historical monthly distribution
            monthly_dist = self.qualifier_patterns['monthly_distribution']
            months = list(monthly_dist.keys())
            weights = [monthly_dist[m] for m in months]
            weights = np.array(weights) / sum(weights)
        else:
            # Default: spread across the year, avoiding only peak summer
            # Women's football friendlies happen year-round
            months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
            weights = [0.08, 0.09, 0.10, 0.11, 0.10, 0.09, 0.08, 0.11, 0.10, 0.09, 0.05]
        
        dates = []
        # Generate more dates than needed
        for _ in range(num_matches * 2):
            month = np.random.choice(months, p=weights)
            day = np.random.randint(1, 29)
            date = pd.Timestamp(year=year, month=int(month), day=day)
            if date not in dates:
                dates.append(date)
        
        return sorted(dates)
    
    def _can_team_play(self, team, date, team_schedule):
        """Check if team can play on this date (not same day or day after)."""
        if team not in team_schedule:
            return True
        
        for scheduled_date in team_schedule[team]:
            if abs((date - scheduled_date).days) <= 1:
                return False
        
        return True
    
    def _is_home_advantage(self, home_team, away_team, location_country):
        """
        Determine if a match should be considered neutral or home advantage.
        """
        
        if home_team == location_country or away_team == location_country:
            return False  # Not neutral - one team is playing in their home country

        country_variations = {
            'United States': ['USA', 'United States'],
            'England': ['England', 'United Kingdom'],
            'Wales': ['Wales', 'United Kingdom'],
            'Scotland': ['Scotland', 'United Kingdom'],
            'Northern Ireland': ['Northern Ireland', 'United Kingdom'],
            'China PR': ['China', 'China PR'],
            'South Korea': ['Korea Republic', 'South Korea'],
            'North Korea': ['Korea DPR', 'North Korea'],
        }
        
        for country, variations in country_variations.items():
            if location_country in variations:
                if home_team in variations or away_team in variations:
                    return False
        
        return True  # Neutral - neither team is from the host country
    
    def _rank_teams_by_strength(self, teams, team_stats):
        """Rank teams by historical performance (win rate and goal difference)."""
        team_rankings = []
        for team in teams:
            if team in team_stats:
                stats = team_stats[team]
                # Combine win probability and matches played as strength indicator
                strength = stats['matches_per_year'] * 0.3 + stats['total_matches'] * 0.001
            else:
                strength = 0.0
            team_rankings.append((team, strength))
        
        # Sort by strength descending
        team_rankings.sort(key=lambda x: x[1], reverse=True)
        return [team for team, _ in team_rankings]
    
    def _generate_structured_tournament(self, year, tournament, teams, location, team_stats, base_date, duration_days):
        """
        Generate a structured tournament with group stage and knockout rounds.
        Uses team strength to create realistic progression.
        Returns list of match dictionaries with 'stage' and 'match_order' metadata.
        """
        matches = []
        num_teams = len(teams)
        
        if num_teams < 4:
            # Too small for structure, just round-robin
            return None
        
        # Rank teams by strength
        ranked_teams = self._rank_teams_by_strength(teams, team_stats)
        
        # Determine tournament format based on number of teams
        if num_teams >= 16:
            # Large tournament: group stage + knockouts
            group_stage_matches, advancing_teams = self._generate_group_stage(ranked_teams, 4, base_date)
            knockout_matches = self._generate_knockout_rounds(advancing_teams, base_date + timedelta(days=int(duration_days * 0.6)))
            matches = group_stage_matches + knockout_matches
        elif num_teams >= 8:
            # Medium tournament: simple groups + knockouts
            group_stage_matches, advancing_teams = self._generate_group_stage(ranked_teams, 2, base_date)
            knockout_matches = self._generate_knockout_rounds(advancing_teams, base_date + timedelta(days=int(duration_days * 0.5)))
            matches = group_stage_matches + knockout_matches
        else:
            # Small tournament: direct knockout
            matches = self._generate_knockout_rounds(ranked_teams, base_date)
        
        # Add tournament info to all matches
        for match in matches:
            match['tournament'] = tournament
            match['city'] = location.get('city', 'Unknown')
            match['country'] = location.get('country', 'Unknown')
            match['is_synthetic'] = True
        
        return matches
    
    def _generate_group_stage(self, ranked_teams, num_groups, start_date):
        """Generate group stage matches. Returns matches and teams that 'advance'."""
        matches = []
        teams_per_group = len(ranked_teams) // num_groups
        
        # Split into groups (distribute strong teams across groups)
        groups = [[] for _ in range(num_groups)]
        for i, team in enumerate(ranked_teams):
            groups[i % num_groups].append(team)
        
        current_date = start_date
        match_order = 0
        
        # Generate round-robin within each group
        for group_idx, group in enumerate(groups):
            # Each team plays each other once
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    matches.append({
                        'date': current_date,
                        'home_team': group[i],
                        'away_team': group[j],
                        'stage': 'group',
                        'group_id': group_idx,
                        'match_order': match_order,
                        'qualifier': 0
                    })
                    match_order += 1
                    # Stagger dates
                    if match_order % 3 == 0:
                        current_date += timedelta(days=2)
        
        # "Advancing" teams: top teams from each group (simplified - take top 2 from each)
        advancing_teams = []
        for group in groups:
            advancing_teams.extend(group[:2])  # Top 2 from each group
        
        return matches, advancing_teams
    
    def _generate_knockout_rounds(self, teams, start_date):
        """Generate knockout rounds (QF, SF, Final) with proper elimination."""
        matches = []
        current_teams = teams.copy()
        current_date = start_date
        match_order = 1000  # Start from 1000 to separate from group stage
        
        round_names = []
        num_teams = len(current_teams)
        
        # Determine round structure
        if num_teams >= 16:
            round_names = ['round_of_16', 'quarter_final', 'semi_final', 'final']
        elif num_teams >= 8:
            round_names = ['quarter_final', 'semi_final', 'final']
        elif num_teams >= 4:
            round_names = ['semi_final', 'final']
        else:
            round_names = ['final']
        
        for round_name in round_names:
            round_matches = []
            num_matches = len(current_teams) // 2
            
            # Pair teams (1st vs last, 2nd vs 2nd-last, etc.)
            for i in range(num_matches):
                match = {
                    'date': current_date,
                    'home_team': current_teams[i],
                    'away_team': current_teams[-(i+1)],
                    'stage': round_name,
                    'match_order': match_order,
                    'qualifier': 0
                }
                round_matches.append(match)
                match_order += 1
            
            matches.extend(round_matches)
            
            # "Winners" advance (take first team from each pairing - stronger teams)
            current_teams = [current_teams[i] for i in range(num_matches)]
            
            # Move to next round date
            current_date += timedelta(days=3)
        
        return matches
    
    def _assign_teams_to_match_slot(self, date, teams, team_schedule, team_match_count, team_stats, period_years, max_multiplier=5.0):
        """Try to assign two teams to a match slot respecting constraints."""
        available_teams = [t for t in teams if self._can_team_play(t, date, team_schedule)]
        
        if len(available_teams) < 2:
            return None
        
        # Shuffle for randomness
        np.random.shuffle(available_teams)
        
        # Try to find a valid pair - be very lenient with match counts
        for i in range(len(available_teams)):
            for j in range(i + 1, len(available_teams)):
                home_team = available_teams[i]
                away_team = available_teams[j]
                
                # Very lenient check - allow up to 5x expected matches
                if home_team in team_stats:
                    expected = team_stats[home_team]['matches_per_year'] * period_years
                    if team_match_count.get(home_team, 0) >= expected * max_multiplier:
                        continue
                
                if away_team in team_stats:
                    expected = team_stats[away_team]['matches_per_year'] * period_years
                    if team_match_count.get(away_team, 0) >= expected * max_multiplier:
                        continue
                
                # Valid pair found
                return (home_team, away_team)
        
        # If still no match, try WITHOUT match count constraint
        if len(available_teams) >= 2:
            return (available_teams[0], available_teams[1])
        
        return None
    
    def _generate_matches_for_period(self, period_start, period_end, training_data):
        period_years = period_end - period_start + 1
        tournament_info, team_stats = self._analyze_patterns(training_data)
        
        matches = []
        team_match_count = {}
        team_schedule = {}
                
        # Categorize tournaments by priority
        small_tournaments = []
        regional_tournaments = []
        major_tournaments = []
        friendlies_qualifiers = []
        
        for tournament, info in tournament_info.items():
            tournament_lower = tournament.lower()
            num_teams = len(info['teams'])
            
            if 'friendly' in tournament_lower:
                friendlies_qualifiers.append((tournament, info))
            elif 'qualification' in tournament_lower or 'qualifying' in tournament_lower:
                friendlies_qualifiers.append((tournament, info))
            elif num_teams < 10:
                small_tournaments.append((tournament, info))
            elif num_teams <= 30:
                regional_tournaments.append((tournament, info))
            else:
                major_tournaments.append((tournament, info))
        
        # Sort each category by number of matches (smallest first for constrained, largest first for flexible)
        small_tournaments.sort(key=lambda x: x[1]['matches_per_occurrence'])
        regional_tournaments.sort(key=lambda x: x[1]['matches_per_occurrence'])
        major_tournaments.sort(key=lambda x: x[1]['matches_per_occurrence'])
        friendlies_qualifiers.sort(key=lambda x: x[1]['matches_per_occurrence'], reverse=True)
        
        ordered_tournaments = small_tournaments + regional_tournaments + major_tournaments + friendlies_qualifiers
        
        for tournament, info in ordered_tournaments:
            valid_years = self._get_tournament_years(tournament, period_start, period_end)
            
            if len(valid_years) == 0:
                continue
            
            is_dense = info['is_dense']
            matches_per_occurrence = info['matches_per_occurrence']
            teams = info['teams']
            locations = info['locations']
            
            # Skip if critical data is missing
            if len(teams) < 2 or len(locations) == 0:
                continue
            
            is_qualifier_or_friendly = 'friendly' in tournament.lower() or 'qualification' in tournament.lower() or 'qualifying' in tournament.lower()
            
            for year in valid_years:
                # Check if we should use structured tournament generation
                use_structure = info.get('has_structure', False) and is_dense and len(teams) >= 8
                
                if use_structure:
                    # Generate structured tournament
                    tournament_location = locations[np.random.randint(0, len(locations))]
                    base_date = pd.Timestamp(year=year, month=info['typical_start_month'], day=info['typical_start_day'])
                    duration = info['typical_duration']
                    
                    structured_matches = self._generate_structured_tournament(
                        year, tournament, teams, tournament_location, 
                        team_stats, base_date, duration
                    )
                    
                    if structured_matches:
                        # Add neutral flag and qualifier to structured matches
                        for match in structured_matches:
                            location_country = match['country']
                            is_neutral = self._is_home_advantage(match['home_team'], match['away_team'], location_country)
                            match['neutral'] = is_neutral
                        
                        # Add to matches and update counters
                        for match in structured_matches:
                            home_team = match['home_team']
                            away_team = match['away_team']
                            
                            if home_team not in team_match_count:
                                team_match_count[home_team] = 0
                                team_schedule[home_team] = []
                            if away_team not in team_match_count:
                                team_match_count[away_team] = 0
                                team_schedule[away_team] = []
                            
                            team_match_count[home_team] += 1
                            team_match_count[away_team] += 1
                            team_schedule[home_team].append(match['date'])
                            team_schedule[away_team].append(match['date'])
                        
                        matches.extend(structured_matches)
                        generated = len(structured_matches)
                        print(f"  Generated {generated} structured matches for {tournament} {year}")
                        continue  # Skip to next tournament/year
                
                # Use original random generation for non-structured tournaments
                if is_dense:
                    date_slots = self._get_dense_tournament_dates(year, tournament, tournament_info)
                else:
                    date_slots = self._get_qualifier_dates(year, matches_per_occurrence * 3)
                
                if is_dense:
                    tournament_location = locations[np.random.randint(0, len(locations))]
                    is_neutral = True
                
                generated = 0
                target_matches = matches_per_occurrence
                year_matches = []  # Collect matches for this tournament occurrence
                
                if is_qualifier_or_friendly:
                    max_multipliers = [50.0, 100.0, 1000.0]
                elif len(teams) < 10:
                    max_multipliers = [5.0, 10.0, 20.0]
                else:
                    max_multipliers = [10.0, 20.0, 50.0]
                
                for max_mult in max_multipliers:
                    if generated >= target_matches:
                        break
                    
                    for date in date_slots:
                        if generated >= target_matches:
                            break
                        
                        if not is_dense:
                            match_location = locations[np.random.randint(0, len(locations))]
                            is_neutral = False
                        else:
                            match_location = tournament_location
                            is_neutral = True
                        
                        team_pair = self._assign_teams_to_match_slot(
                            date, teams, team_schedule, team_match_count, team_stats, period_years, max_mult
                        )
                        
                        if team_pair is None:
                            continue
                        
                        home_team, away_team = team_pair
                        
                        if home_team not in team_match_count:
                            team_match_count[home_team] = 0
                            team_schedule[home_team] = []
                        if away_team not in team_match_count:
                            team_match_count[away_team] = 0
                            team_schedule[away_team] = []
                        
                        # Determine if neutral based on whether a team is playing in their home country
                        location_country = match_location.get('country', 'Unknown')
                        if is_dense:
                            # For tournaments, check if either team is from the host country
                            is_neutral = self._is_home_advantage(home_team, away_team, location_country)
                        else:
                            # For qualifiers/friendlies, already set correctly above
                            is_neutral = False
                        
                        qualifier = 1 if is_qualifier_or_friendly else 0
                        
                        match_dict = {
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'tournament': tournament,
                            'city': match_location.get('city', 'Unknown'),
                            'country': match_location.get('country', 'Unknown'),
                            'neutral': is_neutral,
                            'qualifier': qualifier,
                            'is_synthetic': True,
                            'stage': None  # Will be set later
                        }
                        
                        year_matches.append(match_dict)
                        
                        team_match_count[home_team] += 1
                        team_match_count[away_team] += 1
                        team_schedule[home_team].append(date)
                        team_schedule[away_team].append(date)
                        
                        generated += 1
                
                # Assign stages to matches for this tournament occurrence
                if is_dense and len(year_matches) > 0:
                    # Sort by date to identify progression
                    year_matches_sorted = sorted(year_matches, key=lambda x: x['date'])
                    
                    # Mark stages based on position
                    total = len(year_matches_sorted)
                    if total == 1:
                        year_matches_sorted[0]['stage'] = 'final'
                    elif total <= 3:
                        # Very small tournament: mark last as final
                        for i in range(total - 1):
                            year_matches_sorted[i]['stage'] = 'group'
                        year_matches_sorted[-1]['stage'] = 'final'
                    else:
                        # Larger tournament: estimate stages
                        # Last match is final
                        year_matches_sorted[-1]['stage'] = 'final'
                        
                        # Last ~10% before final are semi-finals
                        sf_start = max(1, total - 3)
                        for i in range(sf_start, total - 1):
                            year_matches_sorted[i]['stage'] = 'semi_final'
                        
                        # Everything else is group stage
                        for i in range(sf_start):
                            year_matches_sorted[i]['stage'] = 'group'
                else:
                    # Qualifiers/friendlies: mark as 'qualifier' or 'friendly'
                    for match in year_matches:
                        if 'qualification' in tournament.lower() or 'qualifying' in tournament.lower():
                            match['stage'] = 'qualifier'
                        elif 'friendly' in tournament.lower():
                            match['stage'] = 'friendly'
                        else:
                            match['stage'] = 'group'  # Default
                
                matches.extend(year_matches)
                
                if generated < target_matches * 0.5 and not is_qualifier_or_friendly:
                    print(f"  Warning: Only generated {generated}/{target_matches} matches for {tournament} in {year}")
        
        return pd.DataFrame(matches)
    
    def fill_historical_periods(self, output_dir='filled_matches'):
        os.makedirs(output_dir, exist_ok=True)
        
        baseline_end = self.baseline['end_year']
        earliest_year = self.results['date'].dt.year.min()
        
        all_results = []
        augmented_data = self.results.copy()
        augmented_data['is_synthetic'] = False
        
        period_to_fill_end = self.baseline['start_year'] - 1
        
        while period_to_fill_end >= earliest_year + 3:
            period_to_fill_start = period_to_fill_end - 3
            
            training_data = augmented_data[
                (augmented_data['date'].dt.year >= period_to_fill_end + 1) &
                (augmented_data['date'].dt.year <= baseline_end)
            ].copy()
            
            if len(training_data) < 100:
                print(f"Skipping period {period_to_fill_start}-{period_to_fill_end}: insufficient training data")
                period_to_fill_end = period_to_fill_start - 1
                continue
            
            period_data = augmented_data[
                (augmented_data['date'].dt.year >= period_to_fill_start) &
                (augmented_data['date'].dt.year <= period_to_fill_end)
            ].copy()
            
            print(f"\nFilling period {period_to_fill_start}-{period_to_fill_end}")
            print(f"Training data: {len(training_data)} matches from {period_to_fill_end + 1}-{baseline_end}")
            print(f"Existing matches in period: {len(period_data)}")
            
            target_matches = self.baseline['total_matches']
            actual_matches = len(period_data)
            needed = target_matches - actual_matches
            
            if needed > 0:
                synthetic_matches = self._generate_matches_for_period(
                    period_to_fill_start, 
                    period_to_fill_end, 
                    training_data
                )
                synthetic_matches = synthetic_matches.head(needed)
                
                augmented_period = pd.concat([period_data, synthetic_matches], ignore_index=True)
                
                augmented_data = augmented_data[
                    (augmented_data['date'].dt.year < period_to_fill_start) |
                    (augmented_data['date'].dt.year > period_to_fill_end)
                ]
                augmented_data = pd.concat([augmented_data, augmented_period], ignore_index=True)
            else:
                augmented_period = period_data
            
            complete_dataset = augmented_data.sort_values('date').reset_index(drop=True)
            
            output_path = f"{output_dir}/period_{period_to_fill_start}_{period_to_fill_end}.csv"
            complete_dataset.to_csv(output_path, index=False)
            
            result = {
                'period_filled': f'{period_to_fill_start}-{period_to_fill_end}',
                'training_data_size': len(training_data),
                'original_matches': actual_matches,
                'synthetic_added': len(synthetic_matches) if needed > 0 else 0,
                'total_dataset_size': len(complete_dataset),
                'output_file': output_path
            }
            all_results.append(result)
            
            print(f"Added {result['synthetic_added']} synthetic matches. Total dataset: {result['total_dataset_size']}")
            
            period_to_fill_end = period_to_fill_start - 1
        
        with open(f"{output_dir}/fill_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results

if __name__ == '__main__':
    filler = HistoricalMatchFiller('cleaned_data/results_standardized_qualifiers.csv')
    results = filler.fill_historical_periods()
    print(f"\nFilled {len(results)} historical periods")