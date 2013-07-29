import readline
import rlcompleter
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

readline.parse_and_bind('tab: complete')
###############################################
# Read the data files
###############################################

labs = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_labs_mod.csv')
adm_dis_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_enc_mod.csv')
vitals = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_vitals.csv')
med_clar_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_med_clarity_mod.csv')
med_cent_df = pd.read_csv('/glusterfs/users/HVA/HVA_data/dr_6988_final_med_centricity_mod.csv')


adm_dis_df['adm_date'] = pd.to_datetime(adm_dis_df['adm_date'])
adm_dis_df['dsc_date'] = pd.to_datetime(adm_dis_df['dsc_date'])

labs['result_time'] = pd.to_datetime(labs['result_time'])
vitals['flo_meas_time'] = pd.to_datetime(vitals['flo_meas_time'])

#Clean up the dates. Remove the dates lower than 1970
adm_dis_df =  adm_dis_df[[x>1970 for x in [x.year for x in adm_dis_df['adm_date']]]]
adm_dis_df =  adm_dis_df[[x>1970 for x in [x.year for x in adm_dis_df['dsc_date']]]]


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

###############################################
#Analysis on stay in the hospital
###############################################

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

###############################################
# Analysis on ages
###############################################

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

###############################################
#Analysis on readmissions
###############################################

adm_dis_pat_group = adm_dis_df.groupby(['pat_id'])
readmission_freq = adm_dis_pat_group['enc_id'].count()
readmission_freq = readmission_freq.tolist()
#Subtract 1 as the first time is not readmission
readmission_freq = [x-1 for x in readmission_freq]
pdf_pages = PdfPages('Readmission_dist.pdf')
fig =plt.figure()
readmission_density = gaussian_kde(readmission_freq)
xs = np.linspace(0,np.max(readmission_freq),200)
readmission_density.covariance_factor = lambda : .25
readmission_density._compute_covariance()
case_plot, = plt.plot(xs,readmission_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Readmission')
plt.ylabel('Density')
plt.title('Readmission distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

###############################################
#Analysis number of vitals per visit
###############################################

vitals_enc_group = vitals.groupby(['enc_id'])
num_vitals_per_visit = vitals_enc_group['fsd_id'].count()
num_vitals_per_visit = num_vitals_per_visit.tolist()
pdf_pages = PdfPages('Vitals_per_visit_dist.pdf')
fig =plt.figure()
vitals_per_visit_density = gaussian_kde(num_vitals_per_visit)
#xs = np.linspace(0,np.max(num_vitals_per_visit),200)
xs = np.linspace(0,3000,200)
vitals_per_visit_density.covariance_factor = lambda : .25
vitals_per_visit_density._compute_covariance()
case_plot, = plt.plot(xs,vitals_per_visit_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Vitals per Visit')
plt.ylabel('Density')
plt.title('Number of vitals per visit distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

###############################################
#Analysis on number of labs per visit
###############################################

labs_enc_group = labs.groupby(['enc_id'])
num_labs_per_visit = labs_enc_group['proc_code'].count()
num_labs_per_visit = num_labs_per_visit.tolist()
pdf_pages = PdfPages('Labs_per_visit_dist.pdf')
fig =plt.figure()
labs_per_visit_density = gaussian_kde(num_labs_per_visit)
#xs = np.linspace(0,np.max(num_labs_per_visit),200)
xs = np.linspace(0,2000,200)
labs_per_visit_density.covariance_factor = lambda : .25
labs_per_visit_density._compute_covariance()
case_plot, = plt.plot(xs,labs_per_visit_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Lab Tests per Visit')
plt.ylabel('Density')
plt.title('Number of Lab Tests per visit distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

###############################################
#Analyzing vitals and lab data for patients
###############################################

vitals_pat_enc_group = vitals.groupby(['pat_id', 'enc_id'])
min_time_vitals = vitals_pat_enc_group['flo_meas_time'].min()
max_time_vitals = vitals_pat_enc_group['flo_meas_time'].max()

adm_dis = testing[['pat_id','enc_id', 'adm_date', 'dsc_date']]

labs_pat_enc_group = labs.groupby(['pat_id', 'enc_id'])
min_time_labs = labs_pat_enc_group['result_time'].min()
max_time_labs = labs_pat_enc_group['result_time'].max()


labs_adm = labs[ labs['enc_id'].isin( adm_dis_df['enc_id'].unique())]

labs_adm_enc_group = labs_adm.groupby(['enc_id'])

num_rows = len( labs_adm['enc_id'].unique())
labs_around_adm = np.ndarray( shape= (num_rows,), dtype= [('pat_id', np.int64), ('enc_id', np.int64), ('before_num', np.int32), ('during_num', np.int32), ('after_num', np.int32)])

cur_row = 0
for enc_id, group in labs_adm_enc_group:
    this_enc = adm_dis_df[ adm_dis_df['enc_id'] == enc_id]
    pat_id = this_enc['pat_id'].tolist()[0]
    adm_date = this_enc['adm_date'].tolist()[0]
    dis_date = this_enc['dsc_date'].tolist()[0]
    before_num = np.sum(group['result_time'] < adm_date)
    after_num = np.sum(group['result_time'] > dis_date)
    during_num = len(group) - (after_num + before_num)
    labs_around_adm[cur_row]['pat_id'] = pat_id
    labs_around_adm[cur_row]['enc_id'] = enc_id
    labs_around_adm[cur_row]['before_num'] = before_num
    labs_around_adm[cur_row]['during_num'] = during_num
    labs_around_adm[cur_row]['after_num'] = after_num
    cur_row = cur_row+1

labs_around_adm_df = pd.DataFrame(labs_around_adm)

###############################################
#Plots on before after during densities
###############################################

num_labs_per_visit = labs_around_adm_df['before_num'].tolist()
pdf_pages = PdfPages('Before_Labs_per_visit_dist.pdf')
fig =plt.figure()
labs_per_visit_density = gaussian_kde(num_labs_per_visit)
#xs = np.linspace(0,np.max(num_labs_per_visit),200)
xs = np.linspace(0,250,200)
labs_per_visit_density.covariance_factor = lambda : .25
labs_per_visit_density._compute_covariance()
case_plot, = plt.plot(xs,labs_per_visit_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Lab Tests Before Visit')
plt.ylabel('Density')
plt.title('Number of Lab Tests Before visit distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

num_labs_per_visit = labs_around_adm_df['during_num'].tolist()
pdf_pages = PdfPages('During_Labs_per_visit_dist.pdf')
fig =plt.figure()
labs_per_visit_density = gaussian_kde(num_labs_per_visit)
#xs = np.linspace(0,np.max(num_labs_per_visit),200)
xs = np.linspace(0,3000,200)
labs_per_visit_density.covariance_factor = lambda : .25
labs_per_visit_density._compute_covariance()
case_plot, = plt.plot(xs,labs_per_visit_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Lab Tests During Visit')
plt.ylabel('Density')
plt.title('Number of Lab Tests During visit distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

num_labs_per_visit = labs_around_adm_df['after_num'].tolist()
pdf_pages = PdfPages('After_Labs_per_visit_dist.pdf')
fig =plt.figure()
labs_per_visit_density = gaussian_kde(num_labs_per_visit)
#xs = np.linspace(0,np.max(num_labs_per_visit),200)
xs = np.linspace(0,300,200)
labs_per_visit_density.covariance_factor = lambda : .25
labs_per_visit_density._compute_covariance()
case_plot, = plt.plot(xs,labs_per_visit_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Lab Tests After Visit')
plt.ylabel('Density')
plt.title('Number of Lab Tests After visit distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

###############################################
# Analysis of patient labs before first admit
###############################################

adm_dis_pat_group = adm_dis_df.groupby(['pat_id'])
first_adm_date_pat = adm_dis_pat_group['adm_date'].min()

num_rows = len(first_adm_date_pat)
labs_before_first_adm = np.ndarray( shape= (num_rows,), dtype= [('pat_id', np.int64), ('before_first_num', np.int64), ('before_first_enc_num', np.int64)])
labs_before_first_adm[:]['before_first_num'] = 0
labs_before_first_adm[:]['before_first_enc_num'] = 0
vitals_before_first_adm = np.ndarray( shape= (num_rows,), dtype= [('pat_id', np.int64), ('before_first_num', np.int64), ('before_first_enc_num', np.int64)])
vitals_before_first_adm[:]['before_first_num'] = 0
vitals_before_first_adm[:]['before_first_enc_num'] = 0

labs_group_pat = labs.groupby(['pat_id'])
vitals_group_pat = vitals.groupby(['pat_id'])
labs
cur_row = 0
for pat_id, group in labs_group_pat:
    try:
        adm_date = first_adm_date_pat.ix[pat_id]
    except Exception as e:
        continue
    before_num = np.sum(group['result_time'] < adm_date)
    before_enc_num = len(group[group['result_time']<adm_date]['enc_id'].unique())
    #print before_num, before_enc_num
    #sys.stdin.readline(1)
    labs_before_first_adm[cur_row]['pat_id'] = pat_id
    labs_before_first_adm[cur_row]['before_first_num'] = before_num
    labs_before_first_adm[cur_row]['before_first_enc_num'] = before_enc_num
    cur_row = cur_row+1
labs_before_first_adm_df = pd.DataFrame(labs_before_first_adm)

pdf_pages = PdfPages('Before_first_encounters.pdf')
fig =plt.figure()
labs_per_visit_density = gaussian_kde(num_labs_per_visit)
#xs = np.linspace(0,np.max(num_labs_per_visit),200)
xs = np.linspace(0,250,200)
labs_per_visit_density.covariance_factor = lambda : .25
labs_per_visit_density._compute_covariance()
case_plot, = plt.plot(xs,labs_per_visit_density(xs))
#plt.hist(readmission_freq)
plt.xlabel('Number of Lab Tests Before Visit')
plt.ylabel('Density')
plt.title('Number of Lab Tests Before visit distribution of the patients in the hospital')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()


cur_row = 0
for pat_id, group in vitals_group_pat:
    try:
        adm_date = first_adm_date_pat.ix[pat_id]
    except Exception as e:
        continue
    before_num = np.sum(group['flo_meas_time'] < adm_date)
    before_enc_num = len(group[group['flo_meas_time']<adm_date]['enc_id'].unique())
    vitals_before_first_adm[cur_row]['pat_id'] = pat_id
    vitals_before_first_adm[cur_row]['before_first_num'] = before_num
    vitals_before_first_adm[cur_row]['before_first_enc_num'] = before_enc_num
    cur_row = cur_row+1
vitals_before_first_adm_df = pd.DataFrame(vitals_before_first_adm)


##################################################################
# YEAR BY YEAR ANALYSIS
##################################################################


# ANALYSIS ON THE NUMBER OF ADMISSIONS PER YEAR
years = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
num_adm = []

for year in years:
    adm_dis_year =  adm_dis_df[[x == year for x in [x.year for x in adm_dis_df['adm_date']]]]
    num_adm.append(len(adm_dis_year['pat_id'].unique()))

pdf_pages = PdfPages('Adm_per_year.pdf')
fig = plt.figure()
width =1
indexes = np.arange(len(num_adm))
plt.bar(indexes,num_adm,width)
plt.xticks(np.arange(len(years))+ width*0.5, years)
plt.xlabel('Year')
plt.ylabel('Number of unique patients in a given year')
plt.title('Histogram of number of patients getting admitted in a given year')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()

# ANALYSIS ON ADMISSIONS IN THE GIVEN YEAR WITHOUT ANY PREVIOUS ADMISSIONS

# None admitted in year 2003 and hence keeping patient id is empty
pat_years = { '2003':[]}
num_adm_wo_prev = []
for year in years:
    a = pat_years.values()
    prev_year_pats = list(itertools.chain(*a))
    adm_dis_year = adm_dis_df[[x == year for x in [x.year for x in adm_dis_df['adm_date']]]]
    this_year_pats = adm_dis_year[ ~(adm_dis_year['pat_id'].isin(prev_year_pats))]['pat_id'].unique()
    pat_years[str(year)] = this_year_pats
    num_adm_wo_prev.append(len(this_year_pats))

pdf_pages = PdfPages('Adms_without_prev_adm_per_year.pdf')
fig = plt.figure()
width =1
indexes = np.arange(len(num_adm_wo_prev))
plt.bar(indexes,num_adm_wo_prev,width)
plt.xticks(np.arange(len(years))+ width*0.5, years)
plt.xlabel('Year')
plt.ylabel('Number of unique patients in a given year')
plt.title('Hist of num of pats admitted in a given year w/o any prev adm')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()


#ANALYSIS ON THE LAB DATA WITH NEW ADMISSIONS ON A GIVEN YEAR
enc_labs_before_year = {'2003':[]}
num_enc_labs_before_year = []
for year in years:
    pats_this_year = pat_years[str(year)]
    labs_this_year_pats = labs[ labs['pat_id'].isin(pats_this_year)]
    labs_before_this_year = labs_this_year_pats[ [x<year for x in [x.year for x in labs_this_year_pats['result_time']]]]
    temp = labs_before_this_year['enc_id'].unique()
    enc_labs_before_year[str(year)] = temp
    num_enc_labs_before_year.append(len(temp))


pdf_pages = PdfPages('Num_encounters_before_year.pdf')
fig = plt.figure()
width =1
indexes = np.arange(len(num_enc_labs_before_year))
plt.bar(indexes,num_enc_labs_before_year,width)
plt.xticks(np.arange(len(years))+ width*0.5, years)
plt.xlabel('Year')
plt.ylabel('Number of encounters before year with first time admission on that year')
plt.title('Hist of encounters before year with first time admission on that year')
plt.show()
pdf_pages.savefig(fig)
pdf_pages.close()


