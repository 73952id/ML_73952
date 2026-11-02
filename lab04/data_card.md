# Adult data card

- Source: Becker, B. and Kohavi, R. (1996), Adult, UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20. Dictionary and license: https://archive.ics.uci.edu/dataset/2/adult (accessed 2026-10-26).
- Distribution used: OpenML `fetch_openml(name='adult', version=2, as_frame=True)`, dataset ID 1590. UCI source license: CC BY 4.0; retain attribution when sharing adaptations.
- Origin: selected records from the 1994 US Census database; this is a historical, filtered sample, not a description of today's population.
- Raw size: 48,842 rows, 14 inputs and one income label. Six numeric inputs and eight categorical inputs. Label is annual income >$50,000 versus <=$50,000.
- Missingness: workclass 2,799; occupation 2,809; native-country 857. Original '?' markers are already represented as missing values by OpenML. Missing values are accepted and imputed from train only.
- Split: stratified 70/30, seed 42. Train 34,189 before / 34,165 after removing 24 exact duplicate rows. Test 14,653, with no rows removed. Processed table contains 48,818 rows, 16 inputs plus row_id, split and income (19 columns).
- Transformations: train median/mode imputation; numeric clipping to train 1st/99th percentiles; capital_net and hours_per_age from clipped inputs; scaling and train-fitted one-hot encoding for Logistic Regression. `fnlwgt` is an input, not a sample weight.
- Quality: zero violations of prescribed types, category domains and ranges; 52 raw excess exact duplicate rows; five identical-feature groups have conflicting labels (10 records). Conflicts are reported and retained. The 50-row review table is in results/review_50.csv: all rows are marked pass, with no impossible values found; three rows contain allowed missing values.
- Biases: historical income and employment patterns, unequal demographic representation, binary recorded sex categories, selection criteria and missingness may encode social and economic disparities. Predictive accuracy does not establish fairness or causation. No subgroup fairness audit is claimed.
- Privacy: public demographic microdata contain combinations of quasi-identifiers and a sensitive income label. No direct names are present, but public availability does not establish anonymity; avoid linking records to people.
- Intended use: reproducible educational comparison and data-quality practice. Performance applies only to this split and period. Income labels cannot be verified from demographics alone.
- Artifacts: results/config.yaml records processing choices, versions and SHA256; results/processed.csv is the exact hashed cleaned table before sparse encoding. It contains demographic data; share with appropriate attribution and care.
