import pandas as pd
import math
import json
from pathlib import Path
from collections import defaultdict

METRICS = Path(
    "/data/scratch/projects/punim1703/nn_tb/REVISION/results/metrics")

tb = pd.read_csv(METRICS / 'naiive_perf_metrics.csv', skiprows=1)
who = pd.read_csv(METRICS / 'who_by_drug_metrics.csv')
cr_who = pd.read_csv(METRICS / 'who_by_drug_metrics.csv')
er = pd.read_csv(METRICS / 'expert_rules_by_drug_metrics.csv')
cr_er = pd.read_csv(METRICS / 'expert_rules_CR_by_drug_metrics.csv')
filt = pd.read_csv(METRICS / 'filtered_expert_rules_by_drug_metrics.csv')
cr_filt = pd.read_csv(METRICS / 'filtered_expert_rules_CR_by_drug_metrics.csv')


assert tb['Drug'].equals(who['drug'])

cr_drugs = ['bedaquiline', 'clofazimine', 'ethionamide',
            'isoniazid', 'levofloxacin', 'moxifloxacin']

tb['Sensitivity'] = tb['Sensitivity'].str.split('/').str[0]

sens_comp_df = pd.concat(
    [
        tb['Drug'],
        tb['Sensitivity'],
        who['sensitivity'],
        cr_who['sensitivity'],
        er['sensitivity'],
        cr_er['sensitivity'],
        filt['sensitivity'],
        cr_filt['sensitivity']
    ],
    axis=1
)

sens_comp_df.columns = ['drug', 'TB', 'WHO', 'WHO CR',
                        'ER', 'ER CR', 'FILT', 'FILT CR']

sens_comp_df = sens_comp_df[sens_comp_df['drug'].isin(cr_drugs)]

print('Sensitivity')
print(sens_comp_df)

tb['Specificity'] = tb['Specificity'].str.split('/').str[0]

spec_comp_df = pd.concat(
    [
        tb['Drug'],
        tb['Specificity'],
        who['specificity'],
        cr_who['specificity'],
        er['specificity'],
        cr_er['specificity'],
        filt['specificity'],
        cr_filt['specificity']
    ],
    axis=1
)

spec_comp_df.columns = ['drug', 'TB', 'WHO', 'WHO CR',
                        'ER', 'ER CR', 'FILT', 'FILT CR']

spec_comp_df = spec_comp_df[spec_comp_df['drug'].isin(cr_drugs)]

print('\n')

print('Specificity')

print(spec_comp_df)
