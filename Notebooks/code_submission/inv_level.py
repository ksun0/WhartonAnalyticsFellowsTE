
# add inventory level

ListOfParts = dist_inv["tyco_electronics_corp_part_nbr"].unique().tolist()

PartnbrYearMonthInventory = dist_inv.drop(["disty_ww_acct","region_name", "fisdate", "invy_at_avg_sell_price"],axis=1)

firstpartnbr = PartnbrYearMonthInventory[PartnbrYearMonthInventory["tyco_electronics_corp_part_nbr"] == "027608-000"]
firstpartnbr.head()

event_dictionary = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
PartnbrYearMonthInventory['Quarter'] = PartnbrYearMonthInventory['tyco_month_of_year_id'].map(event_dictionary)

PartnbrYearMonthInventory = PartnbrYearMonthInventory.drop(["tyco_month_of_year_id"],axis=1)

inventory_quarter = PartnbrYearMonthInventory.groupby(
   ['tyco_electronics_corp_part_nbr', 'tyco_year_id', 'Quarter']).agg({'inventory_qty': 'sum'})

inventory_quarter_groups = inventory_quarter.groupby(
   ['tyco_electronics_corp_part_nbr']).groups

inventory_quarter_groups_size = inventory_quarter.groupby(
  ['tyco_electronics_corp_part_nbr']).size()

inventory_quarter_groups_size.head()

inventory_quarter_groups_size = inventory_quarter_groups_size.tolist()

inventory_total = PartnbrYearMonthInventory.groupby(
   ['tyco_electronics_corp_part_nbr']).agg({'inventory_qty': 'sum'})

inventory_total['quarters_with_data'] = inventory_quarter_groups_size

inventory_total['average'] = inventory_total['inventory_qty'] / inventory_total['quarters_with_data']

inventory_quarter['average'] = inventory_total['average']

inventory_quarter = pd.merge (inventory_quarter, inventory_total, how = "inner", on = "tyco_electronics_corp_part_nbr")

inventory_quarter = inventory_quarter.drop (columns = ["average_x", "inventory_qty_y"])

inventory_quarter_groups_std = inventory_quarter.groupby(
  ['tyco_electronics_corp_part_nbr']).std()

inventory_quarter_groups_std = inventory_quarter_groups_std.drop (columns = ["quarters_with_data", "average_y"])

inventory_quarter.rename(columns={'tyco_electronics_corp_part_nbr':'tyco_electronics_corp_part_nbr',
                          'inventory_qty_x':'quarter_inventory',
                          'quarters_with_data':'quarters_with_data',
                          'average_y':'average_partnbr'}, 
                 inplace=True)

inventory_quarter = pd.merge (inventory_quarter, inventory_quarter_groups_std, how = "inner", on = "tyco_electronics_corp_part_nbr")

inventory_quarter.rename(columns={'tyco_electronics_corp_part_nbr':'tyco_electronics_corp_part_nbr',
                          'quarter_inventory':'quarter_inventory',
                          'quarters_with_data':'quarters_with_data',
                          'average_partnbr':'average_partnbr',
                          'inventory_qty_x': 'std_partnbr'}, 
                 inplace=True)

inventory_quarter ["average_plus_std"] = inventory_quarter["average_partnbr"] + inventory_quarter["std_partnbr"]

inventory_quarter ["average_minus_std"] = inventory_quarter["average_partnbr"] - inventory_quarter["std_partnbr"]

def getValue(row):
    if float(row['average_plus_std']) < float(row['quarter_inventory']) :
        return "2"
    elif float(row['average_minus_std']) > float(row['quarter_inventory']) :
        return "0"
    else:
        return "1"
      
inventory_quarter['value'] = inventory_quarter.apply(getValue, axis=1)
inventory_quarter = inventory_quarter.drop(['quarter_inventory','quarters_with_data',
                                            'average_partnbr','std_partnbr','average_plus_std',
                                            'average_minus_std'],axis=1)

inventory_quarter_new = PartnbrYearMonthInventory.groupby(
   ['tyco_electronics_corp_part_nbr', 'tyco_year_id', 'Quarter']).agg({'inventory_qty': 'sum'})

inventory_quarter_new = inventory_quarter_new.sort_values('fiscal_month')

inventory_quarter_groups.tolist()

inventory_quarter.head()
inventory_quarter.shape
inventory_quarter.columns

inventory_level = inventory_quarter.reset_index()
inventory_level = inventory_level.drop([])

