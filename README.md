## 0. Data Cleaning CLI
download [original dataset](https://www.kaggle.com/datasets/martj42/womens-international-football-results/data) from kaggle

- standardizing spelling
```
sed -e 's/,Euro,/,UEFA Euro,/g' \
    -e 's/,European Championship,/,UEFA Euro,/g' \
    -e 's/,World Cup,/,FIFA World Cup,/g' \
    -e 's/,Olympic qualifyication,/,Olympic qualification,/g' \
    -e 's/,EAFF East Asian Cup,/,EAFF E-1 Championship,/g' \
    -e 's/,EAFF Championship,/,EAFF E-1 Championship,/g' \
    -e 's/,UEFA Olympic Qualifying play-off,/,UEFA Olympic Qualifying Tournament,/g' \
    -e 's/,Africa Cup of Nations qualification,/,African Cup of Nations qualification,/g' \
    -e 's/,OFC Championship,/,OFC Nations Cup,/g' \
    original_data/results.csv > cleaned_data/results_standardized.csv
```
- new col qualifiers
```
awk -F',' 'NR==1 {print $0",qualifier"; next} 
           tolower($6) ~ /qualification|qualifying|qualifyication/ {print $0",1"; next} 
           {print $0",0"}' cleaned_data/results_standardized.csv > cleaned_data/results_standardized_qualifiers.csv
```
- clean up cities with comma
```
sed -i '' 's/"Washington, D.C."/Washington D.C./g; s/"Athens, Georgia"/Athens Georgia/g' cleaned_data/results_standardized_qualifiers.csv
```

## 1. Data Engineering
### reference tables used to connect teams and tournaments
```
tail -n +2 cleaned_data/results_standardized_qualifiers.csv | cut -d',' -f7,8 | sort | uniq > reference_tables/city_country.csv

tail -n +2 cleaned_data/results_standardized_qualifiers.csv | awk -F',' '$6 !~ /FIFA World Cup/ {print $6","$8}' | sort | uniq > reference_tables/tournament_country.csv

tail -n +2 cleaned_data/results_standardized_qualifiers.csv | awk -F',' '{print $2","$8; print $3","$8}' | sort -u > reference_tables/team_country.csv
```
## 2. filling the gaps and scoring the games
1. Tournament stage labelling `historical_stage_labeller.py`
2. fill the gaps so each 4-year period is as dense as the baseline period with `historical_gap_filler.py` 
3. Score each synthetic match with SVD `score_generator.py` going in reverse. This will produce 16 `.csv` files with gradually more synthetic data. For the full set use the earliest period. 

## 3. Visualisation
-  model comparison for specific statistical property for overall data set `aggregate_visualization.py`
- model comparison for specific statistical property for overall data set `model_comparison_scatter.py`
- team ranking Europe comparison for CTGAN `team_ranking_visualizer.py`

