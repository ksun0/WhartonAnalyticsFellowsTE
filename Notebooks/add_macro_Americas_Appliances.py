import pandas as pd

from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

####################################################################################################
#########################           READ DATA SETS            ######################################
####################################################################################################
# Read original files
brent = pd.read_csv('data/macro/Brent_Oil-2.csv',index_col=0)
gold = pd.read_csv('data/macro/Gold_plot-2.csv',index_col=0)
palladium = pd.read_csv('data/macro/Palladium_plot-2.csv',index_col=0)
pmi = pd.read_csv('data/macro/PMI_plot-2.csv',index_col=0)
semi = pd.read_csv('data/macro/Semi_plot.csv',index_col=0)
usauto = pd.read_csv('data/macro/USAutoSale-2.csv',index_col=0)
sp500auto = pd.read_csv('data/macro/SP500AutoIndex_plot-2.csv',index_col=0)
cny = pd.read_csv('data/macro/cny.csv',index_col=0)
cny.rename(columns={"PX_LAST":"CNY"},inplace=True)

inv_level = pd.read_csv('data/internal/inventory_level.csv',index_col=0)
inv_level.rename(columns={'tyco_electronics_corp_part_nbr':'te_corporate_part_number',
                          'tyco_year_id':'fiscal_year_historical',
                          'Quarter':'fiscal_quarter_historical'},inplace=True)


# add paths
# internal data
dist_sales_path = "data/internal/distributor_sales.csv"
cma_path = "data/internal/cma.csv"
direct_sales_path = "data/internal/direct_sales.csv"
part_path = "data/internal/part.csv"
dist_inv_path = "data/internal/distributor_inventory.csv"
customer_path = "data/internal/new_customer.csv"
tariff_path = "data/internal/tariff.csv"

#  read individual datasets
dist_sales = pd.read_csv(dist_sales_path,index_col=0)
cma = pd.read_csv(cma_path,index_col=0)
direct_sales = pd.read_csv(direct_sales_path,index_col=0)
part = pd.read_csv(part_path,index_col=0)
dist_inv = pd.read_csv(dist_inv_path,index_col=0)
customer = pd.read_csv(customer_path,index_col=0)

external = pd.read_csv('data/External factor analysis v2.csv',index_col=0)

externel_keep_ls = [
  'fiscal_month_historical',
  'PMI_USA', 
  'PMI_China',
  'FX_DXY', 
  'FX_CNY',
  'Brent_spot',
  'Gold'
]

externel_df = external[externel_keep_ls]
externel_df = externel_df.reset_index()

# look at tariff
tariff = pd.read_csv(tariff_path,index_col=0)

# try to match tariff to part master
part_tariff = part.merge(tariff,how="inner",left_on="te_corporate_part_number",right_on="Part Number")

# join parts and dist sales
dist_sales_part_joined_df = dist_sales.merge(
    part, how='inner', on=['te_corporate_part_number'])

# join customer
distributor_sales_part_customer_df_joined = dist_sales_part_joined_df.merge(
    customer, how='inner',  left_on=['distributor_customer_id'], right_on=['customer_id'])

# select Americas and Appliances as 1st focus
temp_master = distributor_sales_part_customer_df_joined[(distributor_sales_part_customer_df_joined['product_owning_business_unit_name'] == 'Appliances') & 
                    (distributor_sales_part_customer_df_joined['customer_region'] == 'Americas') &
                                                  (distributor_sales_part_customer_df_joined['customer_distributor_indicator'] == 'Yes')]


keep_list = ['fiscal_year_historical',
       'fiscal_quarter_historical',
       'fiscal_month_historical',
       'sales_quantity',
       'sales_amount',
       'distributor_customer_id',
       'part_sales_status_name',
       'part_marketing_brand_name',
       'part_promote_indicator',
       'product_classification_name',
       'product_structure_label_level_1',
       'product_family_label_level_1',
       'product_owning_segment_name',
       'customer_country_code',
       'customer_industry_business_code_label_level_1',
       'customer_industry_label_level_1',
       'customer_subcontractor_indicator',
       'tier2_distributor_classification_name',
       'inventory_level']

temp_master = temp_master[keep_list].set_index('distributor_customer_id')


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dummy_columns = []  # array for multiple value columns

for column in ['part_sales_status_name',
               'part_marketing_brand_name',
               'product_classification_name',
               'product_structure_label_level_1',
               'product_family_label_level_1',
               'product_owning_segment_name',
               'customer_country_code',
               'customer_industry_business_code_label_level_1',
               'customer_industry_label_level_1',
               'customer_subcontractor_indicator'
               ]:
    if temp_master[column].dtype == object:
        if temp_master[column].nunique() == 2:
            # apply Label Encoder for binary ones
            temp_master[column] = le.fit_transform(temp_master[column])
        else:
            dummy_columns.append(column)
            

temp_master = pd.get_dummies(data=temp_master, columns=dummy_columns)

class_mapping_adp = {
  'High Service':5,
  'Gold':4,
  'Broadliner':3,
  'Silver':2,
  'Affiliate':1
}
# apply mapping method
temp_master['tier2_distributor_classification_name'] = temp_master['tier2_distributor_classification_name'].map(class_mapping_adp)

#########################################################################################################
######################################### Train/Test Data Split ######################################### 
########################################################################################################

# train test split; 2015-2018 Q1/Q2 for training, 2018 Q3/4 + 2019 for testing
# training
temp_master_train_2015 = temp_master[temp_master['fiscal_year_historical'] == 2015]
temp_master_train_2016 = temp_master[temp_master['fiscal_year_historical'] == 2016]
temp_master_train_2017 = temp_master[temp_master['fiscal_year_historical'] == 2017]
temp_master_train_2018 = temp_master[temp_master['fiscal_year_historical'] == 2018]
temp_master_train_2019 = temp_master[temp_master['fiscal_year_historical'] == 2019]
#us_apl_train_2018 = us_apl[(us_apl['fiscal_year_historical'] == 2018) & 
 #                   (us_apl['fiscal_quarter_historical'].isin([1,2]))]

# test data
temp_master_test_2019 = temp_master[temp_master['fiscal_year_historical'] == 2019]
temp_master_test_2018 = temp_master[(temp_master['fiscal_year_historical'] == 2018) & 
                    (temp_master['fiscal_quarter_historical'].isin([4]))]

#us_apl_test_2015 = us_apl[(us_apl['fiscal_year_historical'] == 2017) & 
  #                  (us_apl['fiscal_quarter_historical'].isin([2,3,4]))]

temp_master_train = pd.concat([temp_master_train_2015,
                          temp_master_train_2016, 
                          temp_master_train_2017,
                          temp_master_train_2018,
                          temp_master_train_2019],ignore_index=True)

temp_master_test = pd.concat([temp_master_test_2018,
                         temp_master_test_2019],ignore_index=True)


#us_apl_real_test = pd.concat([us_apl_test_2018,
  #                       us_apl_test_2019],ignore_index=True)

######################################################################################################
#############################################  Modeling  #############################################
######################################################################################################

y_train = temp_master_train['sales_amount']
X_train = temp_master_train.drop('sales_amount',axis=1)

clf_xg = xgb.XGBRegressor(random_state=42)

clf_rf = RandomForestRegressor(random_state=42)

clf_xg.fit(X_train,y_train)
clf_rf.fit(X_train,y_train)

y_test = temp_master_test['sales_amount']
X_test = temp_master_test.drop('sales_amount',axis=1)

clf_xg_hat = clf_xg.predict(X_test).tolist()
clf_rf_hat = clf_rf.predict(X_test).tolist()


#new df 
X_test['sales_amount'] = y_test
X_test['predicted_sales'] = clf_xg_hat


results = X_test[['fiscal_year_historical','fiscal_quarter_historical',
                'sales_amount','predicted_sales']]

results = results.sort_values(['fiscal_year_historical', 'fiscal_quarter_historical'], ascending=True).reset_index()

results1 = results[results['fiscal_year_historical'] == 2018]
results1.groupby(['fiscal_quarter_historical']).agg({'predicted_sales': 'sum'})

results2 = results[results['fiscal_year_historical'] == 2019]
results2.groupby(['fiscal_quarter_historical']).agg({'predicted_sales': 'sum'})

#new df 
X_test['sales_amount'] = y_test
X_test['predicted_sales'] = clf_rf_hat

results = X_test[['fiscal_year_historical','fiscal_quarter_historical',
                'sales_amount','predicted_sales']]

results = results.sort_values(['fiscal_year_historical', 'fiscal_quarter_historical'], ascending=True).reset_index()

results1 = results[results['fiscal_year_historical'] == 2018]
results1.groupby(['fiscal_quarter_historical']).agg({'predicted_sales': 'sum'})

results2 = results[results['fiscal_year_historical'] == 2019]
results2.groupby(['fiscal_quarter_historical']).agg({'predicted_sales': 'sum'})



from xgboost import plot_importance
plot_importance(clf_xg)
plt.show()
