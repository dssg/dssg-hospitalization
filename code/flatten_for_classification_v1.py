import pandas as pd
import numpy as np
from collections import Counter

labs = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_labs_mod.csv')
adm_dis_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_enc_mod.csv')
vitals = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_vitals.csv')

#Convert the datetime columns
adm_dis_df['adm_date'] = pd.to_datetime(adm_dis_df['adm_date'])
adm_dis_df['dsc_date'] = pd.to_datetime(adm_dis_df['dsc_date'])

labs['result_time'] = pd.to_datetime(labs['result_time'])
vitals['flo_meas_time'] = pd.to_datetime(vitals['flo_meas_time'])

#Clean up the dates. Remove the dates lower than 1970
adm_dis_df =  adm_dis_df[[x>1970 for x in [x.year for x in adm_dis_df['adm_date']]]]
adm_dis_df =  adm_dis_df[[x>1970 for x in [x.year for x in adm_dis_df['dsc_date']]]]


#Column names from admission and discharge
column_names = ['outcome', 'enc_id', 'pat_id', 'age', 'race', 'ethnic', 'length_of_stay']
#Column names from vitals data
vital_names_orig = vitals['flo_meas_name'].unique()

#Column names which represent counts
vital_names_count = [x+'_count' for x in vital_names_orig]
#Column names which represent averages
vital_names_avg = [x+'_avg' for x in vital_names_orig]
#Two lists are jaxtaposed with one another
vital_names = [j for i in zip(vital_names_count,vital_names_avg) for j in i]
#Append these names to column names
column_names.extend(vital_names)
#Column names from lab data
lab_names = labs['component_name'].unique()
#TODO: FInd a better way than REMOVE TEMPERATURE WHICH IS THERE IN BOTH LAB AND VITALS
lab_names = lab_names.tolist()
lab_names.remove('TEMPERATURE')
column_names.extend(lab_names)

adm_dis_ind_start = 0
adm_dis_len = 7

vital_ind_start = adm_dis_ind_start + adm_dis_len
vital_len = len(vital_names)

lab_ind_start = vital_ind_start + vital_len
lab_len = len(lab_names)

adm_records = adm_dis_df.to_records()

num_rows = len(adm_records)

flat_data = np.ndarray( shape= (num_rows,), dtype = {'names':column_names, 'formats': ['object']*len(column_names)})

counting_vitals = Counter(zip(vitals['flo_meas_name'], vitals['enc_id']))
counting_labs = Counter(zip(labs['component_name'], labs['enc_id']))


for i in range(0,num_rows):
    cur_row = adm_records[i]
    enc_id = cur_row['enc_id']
    flat_data[i]['outcome'] = cur_row['case_flag']
    flat_data[i]['enc_id'] = cur_row['enc_id']
    flat_data[i]['pat_id'] = cur_row['pat_id']
    flat_data[i]['age'] = cur_row['age_at_dsc']
    flat_data[i]['race'] = cur_row['race']
    flat_data[i]['ethnic'] = cur_row['ethnic']
    seconds = (cur_row['dsc_date'] - cur_row['adm_date'])/ np.timedelta64(1, 's')
    #Length is stored in days
    flat_data[i]['length_of_stay'] = seconds/3600
    #Extract the number of vitals observed during the encounter
    for name_ind in range(vital_ind_start, vital_ind_start+vital_len):
        name_enc = (column_names[name_ind], enc_id)
        count = counting_vitals[name_enc]
        flat_data[i][column_names[name_ind]] = count
    #Extract the number of labs observed during the encounter
    for name_ind in range(lab_ind_start, lab_ind_start+lab_len):
        name_enc = (column_names[name_ind], enc_id)
        count = counting_labs[name_enc]
        flat_data[i][column_names[name_ind]] = count
    
flat_data_df = pd.DataFrame(flat_data)
flat_data_df.to_csv('/glusterfs/users/HVA/HVA_data/ram/flat_data_v1.csv', index = False)




