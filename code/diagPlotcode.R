#read encounter data
data1 = read.csv("/glusterfs/users/HVA/HVA_data/dr_6988_final_enc_mod.csv")
########read csv files with
icd9Group = read.csv("/glusterfs/users/HVA/HVA_data/icd9names.csv")

########3
diag_split = strsplit(as.character(data1[,11]),",")
primary_diag_1 = unlist(lapply(1:length(diag_split),function(x) diag_split[[x]][1]))

primary_diag_2 = unlist(lapply(1:length(diag_split),function(x) diag_split[[x]][1]))

G1 = strsplit(primary_diag_1,"[.]")
group_1 = unlist(lapply(1:length(G1),function(x)G1[[x]][1]))
G2 = strsplit(primary_diag_2,"[.]")
group_2 = unlist(lapply(1:length(G2),function(x)G2[[x]][1]))

case_ix = which(data1$case_flag == 1)
diag_case = c(group_1[case_ix], group_2[case_ix])
diag_name_case =  icd9Group[,2][match(diag_case, icd9Group[,1])]

table_case = table(diag_name_case)
ixx = which(table_case > 250) 
new_table_case = table_case[ixx]

diag_control = c(group_1[-case_ix],group_2[-case_ix])
diag_name_control = icd9Group[,2][match(diag_control, icd9Group[,1])]
table_control = table(diag_name_control)
ixc = which(table_control > 1000)
new_table_control = table_control[ixc]

#bar plot
pdf(file = "/glusterfs/users/HVA/HVA_data/results/PrimaryDiagnosisCase.pdf")
bp_1 = barplot(new_table_case,main ="Primary Diagnosis in Patients who Revisit Hospital within 30 Days",xaxt = "n",ylim=c(0,max(new_table_case)+100))
pos1 = which(new_table_case > 100 & new_table_case < 1500)
text(bp_1[pos1],new_table_case[pos1]+200,cex=0.8,col = "purple",names(new_table_case)[pos1],srt = 90)
heartpos = which(new_table_case > 1500)
text(bp_1[heartpos],new_table_case[heartpos]-100,cex=0.8, col = "purple", names(new_table_case)[heartpos],srt = 90)
dev.off()


pdf(file= "/glusterfs/users/HVA/HVA_data/results/PrimaryDiagnosisControl.pdf")
bp_2 = barplot(new_table_control,main = "Primary Diagnosis in Patients who don't",xaxt = "n",
ylim = c(0,max(new_table_control)+300))
pos2 = which(new_table_control > 1000 & new_table_control < 5000 )
text(bp_2[pos2],new_table_control[pos2]+200,names(new_table_control)[pos2],cex=0.8,srt=90,col="purple")
heartpos = which(new_table_control > 5000)
text(bp_2[heartpos],new_table_control[heartpos]-250,cex=0.8,col="purple",names(new_table_control)[heartpos],srt=90)
dev.off()



