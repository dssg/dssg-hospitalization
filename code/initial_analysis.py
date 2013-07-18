import readline
import rlcompleter
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

readline.parse_and_bind('tab: complete')

# Read teh data files
labs = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_labs_mod.csv')
adm_dis_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_enc_mod.csv')
med_clar_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_med_clarity_mod.csv')
med_cent_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_med_centricity_mod.csv')
vitals = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_vitals.csv')

adm_dis_df['adm_date'] = pd.to_datetime(adm_dis_df['adm_date'])
adm_dis_df['dsc_date'] = pd.to_datetime(adm_dis_df['dsc_date'])

labs['result_time'] = pd.to_datetime(labs['result_time'])
vitals['flo_meas_time'] = pd.to_datetime(vitals['flo_meas_time'])

#Clean up the dates. Remove the dates lower than 1970
adm_dis_df =  adm_dis_df[[x>1970 for x in [x.year for x in adm_dis_df['adm_date']]]]
adm_dis_df =  adm_dis_df[[x>1970 for x in [x.year for x in adm_dis_df['dsc_date']]]]

#Clean up where admission date is before 

case_pats = adm_dis_df[adm_dis_df['case_flag'] == 1]
cont_pats = adm_dis_df[adm_dis_df['case_flag'] == 0]
case_stay = case_pats['dsc_date'] - case_pats['adm_date']
cont_stay = cont_pats['dsc_date'] - cont_pats['adm_date']

#Since when done tolist() on timedelta it is stored in nanoseconds
conversion_factor = (10**9)*3600*24
case_stay_days = np.array(case_stay.tolist())
case_stay_days = case_stay_days/conversion_factor
cont_stay_days = np.array(cont_stay.tolist())
cont_stay_days = cont_stay_days/conversion_factor

pdf_pages = PdfPages('Stay_dist_case.pdf')

fig = plt.figure()

case_stay_density = gaussian_kde(case_stay_days)
#xs = np.linspace(0,np.max(case_stay_days),200)
xs = np.linspace(0,80,200)
case_stay_density.covariance_factor = lambda : .25
case_stay_density._compute_covariance()
case_plot, = plt.plot(xs,case_stay_density(xs))
plt.xlabel('Length stay in hospital(Days)')
plt.ylabel('Density')
plt.title('Distribution of length of stay in the hospital ')

cont_stay_density = gaussian_kde(cont_stay_days)
#xs = np.linspace(0,np.max(cont_stay_days),200)
xs = np.linspace(0,80,200)
cont_stay_density.covariance_factor = lambda : .25
cont_stay_density._compute_covariance()
cont_plot,  = plt.plot(xs,cont_stay_density(xs))
plt.legend([case_plot, cont_plot], ["Case Patients", "Control Patients"])

pdf_pages.savefig(fig)

pdf_pages.close()

# Analysis on ages
pdf_pages = PdfPages('Age_dist_case.pdf')
fig = plt.figure()

case_age = np.array(case_pats['age_at_dsc'].tolist())
cont_age = np.array(cont_pats['age_at_dsc'].tolist())
case_age = case_age[np.array([not(math.isnan(x)) for x in case_age])]
cont_age = cont_age[np.array([not(math.isnan(x)) for x in cont_age])]

case_age_density = gaussian_kde(case_age)
xs = np.linspace(0,np.max(case_age),200)
case_age_density.covariance_factor = lambda : .25
case_age_density._compute_covariance()
case_plot, = plt.plot(xs,case_age_density(xs))

cont_age_density = gaussian_kde(cont_age)
xs = np.linspace(0,np.max(cont_age),200)
cont_age_density.covariance_factor = lambda : .25
cont_age_density._compute_covariance()
cont_plot, = plt.plot(xs,cont_age_density(xs))
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age distribution of the patients in the hospital')
plt.legend([case_plot, cont_plot], ["Case Patients", "Control Patients"])
pdf_pages.savefig(fig)
pdf_pages.close()


#Analyzing vitals and lab data for patients

vitals_pat_enc_group = vitals.groupby(['pat_id', 'enc_id'])
min_time_vitals = vitals_pat_enc_group['flo_meas_time'].min()
max_time_vitals = vitals_pat_enc_group['flo_meas_time'].max()

adm_dis = testing[['pat_id','enc_id', 'adm_date', 'dsc_date']]

labs_pat_enc_group = labs.groupby(['pat_id', 'enc_id'])
min_time_labs = labs_pat_enc_group['result_time'].min()
max_time_labs = labs_pat_enc_group['result_time'].max()
