import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from datetime import date




df = pd.read_csv('/home/gawshalini/Documents/customer profiling/test example python/customer_dummy_data/ASSCOCIATION_RULE_MINING_TESTING_DATA.csv')
# print(store_data)
print(df.dtypes)
print(df.columns.tolist())

df_new=df[['CUSTOMER_TITLE','CITY_NAME','CUSTOMER_GENDER']]
print(df_new)
print(df_new.shape)
records = []
for i in range(0, 2):
    records.append([str(df_new.values[i,j]) for j in range(0, 2)])

association_rules = apriori(records, min_support=0.3, min_confidence=0.5, min_lift=0.5, min_length=3)
association_results = list(association_rules)
print(association_results)


def apriori_show_mining_results(association_results):
    ap = []
    for association_result in association_results:
        converted_record = association_result._replace(ordered_statistics=[x._asdict() for x in association_result.ordered_statistics])
        ap.append(converted_record._asdict())
    print("Rules:\n------")
    for ptn in ap:
        for rule in ptn["ordered_statistics"]:
            head = rule["items_base"]
            tail = rule["items_add"]
            if len(head) == 0 or len(tail) == 0:
                  continue
            confidence = rule["confidence"]
            lift = rule["lift"]
            support = ptn["support"]
            print('({}) ==> ({})  confidence = {} support= {} lift={}'.format(', '.join(head), ', '.join(tail), round(confidence, 3),support,lift))

apriori_show_mining_results(association_results)
