## 0. Data Cleaning
standardizing spelling
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
new col qualifiers
```
awk -F',' 'NR==1 {print $0",qualifier"; next} 
           tolower($6) ~ /qualification|qualifying|qualifyication/ {print $0",1"; next} 
           {print $0",0"}' cleaned_data/results_standardized.csv > cleaned_data/results_standardized_qualifiers.csv
```
clean up cities with commata
```
sed -i '' 's/"Washington, D.C."/Washington D.C./g; s/"Athens, Georgia"/Athens Georgia/g' cleaned_data/results_standardized_qualifiers.csv
```


#### reference tables
```
tail -n +2 cleaned_data/results_standardized_qualifiers.csv | cut -d',' -f7,8 | sort | uniq > reference_tables/city_country.csv

tail -n +2 cleaned_data/results_standardized_qualifiers.csv | awk -F',' '$6 !~ /FIFA World Cup/ {print $6","$8}' | sort | uniq > reference_tables/tournament_country.csv

tail -n +2 cleaned_data/results_standardized_qualifiers.csv | awk -F',' '{print $2","$8; print $3","$8}' | sort -u > reference_tables/team_country.csv
```


## 1. OG Dataset Analysis
- properties
- data cleaning: exclude homelesscup : done
- handle disappearing countries (USSR, Yugoslawia, etc.) : 
## 2. Verify ML Pipeline - Baseline

## 3. Test SVD
### 3.1 fill in the gaps - match density
baseline: 2021 - 2024 (includes world cup, continental cups, olympics)  
fill in the gaps in reverse order (the sparser the original period, the more data available to fill in)  
- refinement 1: connect tournament, city, country for more realistic data
- refinement 2: consider dates for more realistic data
- refinement 3: connect team - country - neutral
$\rightarrow$ verify properties are still in tact

### constraints in SDV
```python3 -c "from sdv import constraints; print(dir(constraints))"
['Constraint', 'FixedCombinations', 'FixedIncrements', 'Inequality', 'Negative', 'OneHotEncoding', 'Positive', 'Range', 'ScalarInequality', 'ScalarRange', 'Unique', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'base', 'create_custom_constraint_class', 'errors', 'tabular', 'utils']```

### 3.2 fictional teams
### 3.3 parallel Universe
## 4. Bonus - league where Switzerland wins World Cup

