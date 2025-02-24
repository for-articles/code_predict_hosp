***********************************
*    NPR 
***********************************
// After append al npr files in python, we merged with diagnoses file and key file that specifies the group (0= control or 1= intervention)
use "/home/mohsen.g.askar/workbench/NPR_data_all_no_group.dta"
merge m:m lopenr episode_nokkel using "/home/mohsen.g.askar/workbench/tilstandogpros_1948.dta"
rename _merge merge_with_diagnoses
save "/home/mohsen.g.askar/workbench/NPR_data_all_no_group.dta", replace
merge m:1 lopenr using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/lopenr_1948_with_group"
save "/home/mohsen.g.askar/workbench//Files_After_Manipulation/NPR_data_all.dta", replace

// Determine the index date for intervention group 
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/NPR_data_all.dta"
preserve
keep if group==1
codebook lopenr
sort lopenr inndato
// varaible'aar' is not accurate some values of inndato dont match with values of aar
gen year = substr( inndato, 1,4)
destring year, replace
keep if year >= 2018
codebook year
sort inndato //make sure all starts in 2018
sort lopenr inndato
bysort lopenr: gen index_date=_n
keep if index_date==1
codebook lopenr
keep lopenr group inndato
rename inndato index_date
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/intervention_with_index_date.dta", replace

// make a seprate key file for control group to simulate their index_date
restore
preserve
keep if group == 0
codebook lopenr
sort lopenr inndato
bysort lopenr: gen test =_n
keep if test ==1
codebook lopenr
keep lopenr group
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/control_with_index_date.dta", replace

// move to python to make simualted index_dates for the control group (NPR notebook)

// after making the simulated dates to control group, the file was saved to .csv and imported here to avoid the confilct fo saving it as .dta from python then append 
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/control_with_index_date.csv", clear
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/control_with_index_date.dta", replace
append using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/intervention_with_index_date.dta"
codebook group
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/lopenr_1948_with_group_index_date.dta"

// remove patient history which comes after the index date 
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/study_population.dta"
gen index_date_2 = date( index_date, "YMD")
format index_date_2 %td
rename index_date_2 index_date_stata
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/study_population.dta", replace

// merge with index_date 
drop _merge merge_with_diagnoses
merge m:1 lopenr using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/study_population.dta"
sort lopenr inndato
preserve
gen inndato_stata = date(inndato, "YMD")
format inndato_stata %td
sort lopenr inndato_stata
by lopenr: drop if inndato_stata >= index_date_stata //lopenr=66,357 patients

// changes made to NPR to reduce its size
import delimited "\NPR_data_all.csv"
preserve
keep tjenesteenhetlokal tjenesteenhetkode
duplicates drop
export delimited using "\tjenesteenhet_nøkkel.csv"
restore
drop tjenesteenhetlokal
preserve
keep fagenhetlokal fagenhetkode
duplicates drop
export delimited using "\fagenhet_nøkkel.csv"
restore
drop fagenhetlokal
preserve
rename time_since_last_admission time_since_last_admission_days
drop index_date_stata inndato_stata
drop fodselsaar
replace drg_type = "K" if drg_type == "Kirurgisk"
replace drg_type = "M" if drg_type == "Medisinsk"
notes drg_type : M=Medisinisk, K=Kirurgisk
replace polkonaktivitet = trim( polkonaktivitet )
replace aktivitetskategori3 = trim( aktivitetskategori3 )
replace behandlingsniva3 = trim( behandlingsniva3 )
replace drg_type = trim( drg_type )
drop vekt
replace korrvekt = trim( korrvekt )
replace npkopphold_drgbasispoeng = trim( npkopphold_drgbasispoeng )
replace tjenesteenhetkode = trim( tjenesteenhetkode )
replace fagenhetkode = trim( fagenhetkode )
replace episodefag = trim( episodefag )
replace spesialist_1 = trim( spesialist_1 )
replace spesialist_2 = trim( spesialist_2 )
replace spesialist_3 = trim( spesialist_3 )
replace hdg = trim( hdg )
replace kommtjeneste = trim( kommtjeneste )
drop lengthofstay
replace liggetid_periode = trim( liggetid_periode )
replace utskrklardato = trim( utskrklardato )
gen year = substr( inndato, 1, 4)
drop aar
rename year =year_of_admission
tostring institusjonid, replace
replace tilinstitusjonid = trim( tilinstitusjonid )
replace innmatehast = trim( innmatehast )
replace inntilstand = trim( inntilstand )
tostring uttilstand , replace
replace henvtype = trim( henvtype )
replace henvfrainstitusjonid = trim( henvfrainstitusjonid )
replace henvtilinstitusjonid = trim( henvtilinstitusjonid )
replace henvfratjeneste = trim( henvfratjeneste )
replace henvtiltjeneste = trim( henvtiltjeneste )
replace omsnivahenv = trim( omsnivahenv )
replace kodeverdi = trim( kodeverdi )

// to further reduce the size, variable 'institusjonid', 'frainstitusjonid', 'tilinstitusjonid'	shoule be renamed as number from 1 to ... and make separet files as keys

import delimited "\NPR_data_all_reduced_size.csv"
replace frainstitusjonid = trim( frainstitusjonid )
replace frainstitusjonid = "U" if frainstitusjonid == ""
tostring tilinstitusjonid, replace
replace tilinstitusjonid = trim( tilinstitusjonid )
replace tilinstitusjonid = "U" if tilinstitusjonid == ""
replace tilinstitusjonid = "U" if tilinstitusjonid == "."
tostring institusjonid , replace
notes frainstitusjonid : "U"= Unregistered
notes tilinstitusjonid : "U"= Unregistered
preserve
keep institusjonid
duplicates drop
gen institusjonnyid =_n
save "\institusjonid_nokkel.dta"
restore
keep frainstitusjonid
duplicates drop
gen frainstitusjonnyid =_n
tostring frainstitusjonnyid, replace
replace frainstitusjonnyid = "U" in 1
save "\frainstitusjonid_nokkel.dta"
restore
preserve
keep tilinstitusjonid
duplicates drop
gen tilinstitusjonnyid =_n
tostring tilinstitusjonnyid, replace
replace tilinstitusjonnyid = "U" in 1
save "\tilinstitusjonid_nokkel.dta"

use "\temp_NPR_to_merge.dta", clear
merge m:1 institusjonid using "\institusjonid_nokkel.dta"
drop _merge
merge m:1 frainstitusjonid using "\frainstitusjonid_nokkel.dta"
drop _merge
merge m:1 tilinstitusjonid using "\tilinstitusjonid_nokkel.dta"
drop _merge
tostring institusjonnyid, replace
drop institusjonid frainstitusjonid tilinstitusjonid
rename institusjonnyid institusjonid
rename frainstitusjonnyid frainstitusjonid
rename tilinstitusjonnyid tilinstitusjonid
export delimited using "\NPR_data_all_reduced_size.csv", replace

// generate some other indicators 
tostring kodenr, gen (kodenr2)
gen indiaction = substr(kodenr2, 1, 1)
gen diag_importance =0
replace diag_importance=1 if indiaction=="1"
replace diag_importance=2 if diag_importance ==0

gen s1 =0
replace s1 = 1 if spesialist_1==1 | spesialist_2==1 | spesialist_3 ==1
rename s1 spesialist_contact

export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv, replace
//next step is to remove variables that will not be usen and then merge

***********************************
*    KPR 
***********************************


// save files to a new directory as the /archive directory is read-only
use "/home/mohsen.g.askar/archive/Masterfiler for Mohsen/KPR_diagnose_1948.dta"
drop _merge
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_diagnose_1948.dta"
use "/home/mohsen.g.askar/archive/Masterfiler for Mohsen/KPR_takstkode_1948.dta"
drop _merge
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_takstkode_1948.dta"
use "/home/mohsen.g.askar/archive/Masterfiler for Mohsen/KPR_tjenestetype_1948.dta"
drop _merge
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_tjenestetype_1948.dta"

//merge files
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_diagnose_1948.dta"
merge m:m lopenr regning_nokkel using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_takstkode_1948.dta"
rename _merge merge_with_takstkode //101,634 didn't match because they had no diagnosis (by inspecting they are tannlege/fysio visits) == remove them 
merge m:m lopenr regning_nokkel using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_tjenestetype_1948.dta"

drop if merge_with_takstkode == 2
drop merge_with_takstkode _merge

//save to one file 'KPR_data_all.dta'
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all.dta"

// mapping ICPC2 to ICD-10
// separate the rows which are already ICD-10
preserve
keep if diagnosetabell == "ICPC-2"
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_ICPC_Only.dta", replace
restore
preserve
drop if diagnosetabell == "ICPC-2"
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_ICD_Only.dta", replace
//
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_ICPC_Only.dta", replace
codebook diagnosekode
list if strlen( diagnosekode)>3
count if strlen( diagnosekode)>3
codebook diagnosekode if strlen( diagnosekode)>3 //4 ICPC-2 codes have 4 character legnth (22,707 rows). They are (A27,(which is not correct), A981(cytologi pregnency screening), S422, W781)
replace diagnosekode = subinstr( diagnosekode, "A27,", "A27", .)
replace diagnosekode = subinstr( diagnosekode, "S422", "S42", .)
merge m:1 diagnosekode using  "/home/mohsen.g.askar/workbench/Mapping/Map_ICPC_To_ICD_2024_Small.dta"
keep if _merge==3
//ICPC-2 codes wrongly registered (-) were removed (-30 to -69)
//now append with ICD-only
drop _merge ICPC2kodetekst ICD10kodetekst
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_ICPC_Only.dta", replace
append using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_ICD_Only.dta"
save
"/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_mapped_to_ICD.dta"
gen strlegnth = strlen( icd10kode_kpr)
count if strlegnth==38 //wrongly registered codes
drop if strlegnth==38
drop strlegnth
recast str8 icd10kode_kpr

rename ICD10kode   ICD10kode_KPR
rename dato dato_KPR
rename regning_nokkel regning_nokkel_KPR
rename diagnosetabell diagnosetabell_KPR
rename kodenr kodenr_KPR
rename diagnosekode diagnosekode_ICPC_KPR
rename takstkode takstkode_KPR
rename tjenestetype tjenestetype_KPR
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_mapped_to_ICD.dta", replace

// remove patient history which comes after the index date from KPR
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/KPR_data_all_mapped_to_ICD.dta", replace
preserve
merge m:1 lopenr using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/study_population.dta"
sort lopenr dato_KPR
by lopenr: drop if dato_KPR >= index_date_stata
codebook lopenr
drop _merge group
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.dta" //

// make some useful variables fro prediction, 
// we made how many visits 1, 3, 6 months and 1 year and total visits before index date. And also how many visit to fastlege, fys_prv .. osv for each patient (Python), file saved "After_removing_history_before_index_date/KPR_data_all.csv"

// further reduce the datasize as possibel
preserve
keep tjenestetype_kpr
duplicates drop
gen tjenestetype_kpr_2 = "AP" if tjenestetype_kpr == "audioped"
replace tjenestetype_kpr_2= "FL" if tjenestetype_kpr == "fastlege"
replace tjenestetype_kpr_2 = "fys_k" if tjenestetype_kpr == "fys_kom"
replace tjenestetype_kpr_2 = "fys_p" if tjenestetype_kpr == "fys_prv"
replace tjenestetype_kpr_2 = "HS" if tjenestetype_kpr == "hstasjon"
replace tjenestetype_kpr_2 = "KP" if tjenestetype_kpr == "kiroprak"
replace tjenestetype_kpr_2 = "LP" if tjenestetype_kpr == "logoped"
replace tjenestetype_kpr_2 = "OP" if tjenestetype_kpr == "ortopti"
replace tjenestetype_kpr_2 = "RF" if tjenestetype_kpr == "ridefys"
replace tjenestetype_kpr_2 = "TL" if tjenestetype_kpr == "tannlege"
replace tjenestetype_kpr_2= "TP" if tjenestetype_kpr == "tannplei"
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/Keys/KPR/tjenestetype_KPR.dta", replace
restore
preserve
merge m:1 tjenestetype_kpr using  "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/Keys/KPR/tjenestetype_KPR.dta"
drop _merge
drop  tjenestetype_kpr
rename tjenestetype_kpr_2 tjenestetype_kpr

export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv, replace

// generate variable that summerized info from takskoder 

gen inperson_contact_KPR = 0
gen speech_therapy_KPR = 0
gen fys_group_KPR = 0
replace speech_therapy_KPR=1 if takstkode_kpr== "A2b"
replace inperson_contact_KPR =1 if takstkode_kpr== "2ad" | takstkode_kpr=="2dd" | takstkode_kpr=="2cd"
replace fys_group_KPR =1 if takstkode_kpr=="A2g" | takstkode_kpr== "A10" |takstkode_kpr== "B22" | takstkode_kpr== "A2c" | takstkode_kpr=="A8a" | takstkode_kpr=="A2e"
sort lopenr
gen takskode_flag = ( takstkode_kpr == "701a")
egen count_labtest_KPR = total( takskode_flag), by (lopenr)
drop takskode_flag
codebook count_labtest_KPR

// main and secondary diagnosis
tostring kodenr_kpr, gen (kodenr2)
gen indiaction = substr(kodenr2, 1, 1)
gen diag_importance_KPR =0
replace diag_importance_KPR=1 if indiaction=="1"
replace diag_importance_KPR=2 if diag_importance ==0

// save 
export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv, replace


***********************************
*    LMR 
***********************************
// in python, we removed patients that are only in LMR and has no admission history
// we appended LMR in "LMR_data_all.csv" //51,665 unique patients 
// we removed patients that are in LMR and not in NPR //48,404 unique patients

// some data managment to reduce size 
import delimited "/home/mohsen.g.askar/workbench/LMR_data_all.csv"
preserve
keep vare_varenavn vare_varenummer
duplicates drop
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/Keys/LMR/varenr_varenavn.dta"
restore
preserve
drop vare_varenavn
keep pasient_fylkesnavn pasient_fylkesnummer
duplicates drop
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/Keys/LMR/fylkenr_navn.dta"
restore
preserve
drop vare_varenavn pasient_fylkesnavn
drop vare_pakningsstorrelse_enhet
drop pasient_fodselsar
export delimited using /home/mohsen.g.askar/workbench/LMR_data_all.csv, replace

// we made ttt episodes to determine the continous use of medications //44,825 unique patients
// file was saved named 'LMR_Ready_To_Merge.csv'
// move to Stata 
// then patients who died before 2018-2019 were removed //44,786 unique patients
order lopenr legemiddel_atckode_niva5 vare_varenavn
sort lopenr legemiddel_atckode_niva5
gen pasient_dodsarmaned_2 = monthly( pasient_dodsarmaned, "YM")
format pasient_dodsarmaned_2 %tm
drop if pasient_dodsarmaned_2 < tm(2020m1)
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/LMR_data_all.dta"
// now we have the full cohort "study_population.dta" and the LMR cohort "LMR_cohort_final.dta"
drop pasient_dodsarmaned_2 treatment_end_2 treatment_start_2 treatment_end treatment_start treatment_episode lag_carryover carryover lag_rx_dekn_dag resept_dekning_dager lead_ddd_80p_adh lag_ddd_80p_adh ddd_80p_adh lead_ddd lag_ddd delta_dager lead_dato lag_dato sum_ordinasjonantallddd
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/LMR_data_all.dta", replace

//remove patient history which comes after the index date LMR
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/LMR_data_all.dta"
gen utlevering_dato_stata = date( utlevering_dato, "YMD")
format utlevering_dato_stata %td
merge m:1 lopenr using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/study_population.dta"
sort lopenr utlevering_dato_stata
by lopenr: drop if utlevering_dato_stata >= index_date_stata
drop _merge group
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta"

// generte some varaibles indicators for the model
// patient_adherence_status (made in python) and merged in stata 
use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta"
merge m:1 lopenr using "/home/mohsen.g.askar/workbench/adherence_status.dta"
keep if _merge==3
drop _merge
rename adhered_status patient_overall_adherence_status
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta", replace


********************************************
* Merging all datasets to start ML pipeline
*********************************************
//next step is to merge NPR, KPR and LMR in one file
// NPR

import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv", clear
sort lopenr episode_nokkel
drop episode_nokkel episodenrsammedag alder npkopphold_drgbasispoeng tjenesteenhetkode fagenhetkode episodefag spesialist_1 spesialist_2 spesialist_3 kommtjeneste liggetid_periode utskrklardato inndato utdato inntilstand uttilstand henvtype henvfrainstitusjonid henvtilinstitusjonid kodenr2 kodenr
rename yearyear_of_admission year_of_admission
drop indiaction frainstitusjonid tilinstitusjonid index_date polkonaktivitet
replace korrvekt = subinstr( korrvekt , ",", ".", .)
destring korrvekt, replace
destring korrvekt, gen (korrvekt2)
drop korrvekt
rename korrvekt2 korrvekt
replace niva ="NA" if niva ==" "
replace samtykkekompetanse ="NA" if samtykkekompetanse ==" "
tostring aktivitetskategori3, replace
replace aktivitetskategori3 ="NA" if aktivitetskategori3 =="."
tostring behandlingsniva3 , replace
replace behandlingsniva3 ="NA" if behandlingsniva3 =="."
tostring omsorgsniva , replace
codebook drg_type
replace drg_type ="NA" if drg_type ==""
tostring hdg, replace
replace hdg ="NA" if hdg =="."
tostring innmatehast , replace
replace innmatehast ="NA" if innmatehast =="."
replace fagomrade ="NA" if fagomrade ==""
tostring henvfratjeneste , replace
replace henvfratjeneste ="NA" if henvfratjeneste =="."
tostring henvtiltjeneste , replace
replace henvtiltjeneste ="NA" if henvtiltjeneste =="."
tostring omsnivahenv , replace
replace omsnivahenv ="NA" if omsnivahenv =="."
replace kodenavn ="NA" if kodenavn ==""
replace kodeverdi ="NA" if kodeverdi ==""


replace admission_season ="W" if admission_season =="Winter"
replace admission_season ="Su" if admission_season =="Summer"
replace admission_season ="F" if admission_season =="Fall"
replace admission_season ="Sp" if admission_season =="Spring"

export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/NPR_data_all_ready_to_ML.csv, replace


// KPR
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv", clear
sort lopenr regning_nokkel_kpr
drop regning_nokkel_kpr diagnosetabell_kpr kodenr_kpr kodenr2 diagnosekode_icpc_kpr takstkode_kpr index_date dato_kpr indiaction
export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/KPR_data_all_ready_to_ML.csv

// LMR
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all_reduced_size.csv", clear
sort lopenr
drop pasient_dodsarmaned refusjon_icd10kode_verdi refusjon_icpc2kode_verdi utlevering_antallpakninger utlevering_dato utlevering_ddd vare_ddd_enhet vare_ddd_verdi vare_pakningsstorrelse vare_reseptgruppe_verdi vare_varenummer index_date is_anticholinergic

tostring pasient_fylkesnummer, replace
replace pasient_fylkesnummer ="NA" if pasient_fylkesnummer =="."
tostring utlevering_resepttype_verdi , replace

export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/LMR_data_all_ready_to_ML.csv, replace

// Merging all datasets
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/KPR_data_all_ready_to_ML.csv", clear
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/KPR_data_all_ready_to_ML.dta"
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/LMR_data_all_ready_to_ML.csv", clear
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/LMR_data_all_ready_to_ML.dta"
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/NPR_data_all_ready_to_ML.csv", clear
merge m:m lopenr using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/KPR_data_all_ready_to_ML.dta"
rename _merge _merge_NPR_KPR
merge m:m lopenr using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/LMR_data_all_ready_to_ML.dta"
rename _merge _merge_NPR_LMR

drop if _merge_NPR_KPR ==2 //11,646 removed, unique lopenr = 66,399
drop if _merge_NPR_KPR ==2 //103 removed, unique lopenr = 66,357
drop _merge_NPR_KPR _merge_NPR_LMR

export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/Data_all_ready_to_ML.csv, replace

// ATC code with lopenr were saved to make modules 
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ATC_LMR.dta"
duplicates drop
rename legemiddel_atckode_niva5 ATC
export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/To_Make_Networks/ATC_LMR.csv, replace

// ICD from NPR and KPR were saved to make modules 
"/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ICD_10_NPR_KPR.dta"

use "/home/mohsen.g.askar/workbench/Files_After_Manipulation/To_Make_Networks/ICD_10_NPR_KPR.dta"
preserve
keep lopenr kodenavn kodeverdi 
duplicates drop
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/To_Make_Networks/icd_npr.dta"
restore
preserve
keep lopenr icd10kode_kpr
duplicates drop 
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/icd_kpr.dta"
drop if icd10kode_kpr==""
gen kodenavn="ICD-10"
rename icd10kode_kpr kodeverdi
save 
"/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/icd_kpr.dta", replace
append using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/icd_npr.dta"
duplicates drop 
sort lopenr kodeverdi
drop kodenavn
duplicates drop
export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/icd_all.csv, replace

// move to python and use modularity_encoding 
// saved modules are in files "Ready_To_Model/To_Make_Networks/ATC_modules.csv" and "Ready_To_Model/To_Make_Networks/ICD_modules.csv"

// merge module number with the full dataframe 
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ATC_modules.csv"
drop lopenr
rename atc legemiddel_atckode_niva5
duplicates drop
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ATC_modules_To_Merge.dta"
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ICD_modules.csv", clear
drop lopenr
duplicates drop
save "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ICD_modules_To_Merge.dta"


// now we should merge modules_id with the full dataset and remove the code systems 
import delimited "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/Data_all_ready_to_ML.csv", clear
gen strlegnth = strlen( icd10kode_kpr)
drop if strlegnth==38
drop strlegnth
recast str8 icd10kode_kpr
preserve
replace kodeverdi = icd10kode_kpr if kodeverdi=="NA"
merge m:1 kodeverdi using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ICD_modules_To_Merge.dta"
drop if kodeverdi ==""
drop if kodeverdi =="-"
drop if kodeverdi =="0"
drop _merge
merge m:1 legemiddel_atckode_niva5 using "/home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/To_Make_Networks/ATC_modules_To_Merge.dta"
drop _merge
drop icd10kode_kpr kodeverdi
drop legemiddel_atckode_niva5

// handling the missing values in the datset 
// Static values= (sex, municiplilt) --> fill with the same value
// categorical = separte category (NA)
// nuemric= imputation with median values
// final touches on varaibles to ensure correct types and fill missing in categorical varaibles
sort lopenr
replace niva ="NA" if niva ==""
replace samtykkekompetanse ="NA" if samtykkekompetanse ==""
replace aktivitetskategori3 ="NA" if aktivitetskategori3 ==""
rename aktivitetskategori3 aktivitetskategori
replace behandlingsniva3 ="NA" if behandlingsniva3 ==""
rename behandlingsniva3 behandlingsniva
tostring omsorgsniva, replace
replace omsorgsniva ="NA" if omsorgsniva =="."
codebook drg_type
replace drg_type ="NA" if drg_type ==""
replace hdg ="NA" if hdg ==""
replace fagomrade ="NA" if fagomrade ==""
replace henvfratjeneste ="NA" if henvfratjeneste ==""
replace henvtiltjeneste ="NA" if henvtiltjeneste ==""
replace omsnivahenv ="NA" if omsnivahenv ==""
drop kodenavn
replace admission_season ="NA" if admission_season ==""
tostring institusjonid, replace
replace institusjonid ="NA" if institusjonid =="."
replace admissions_trend ="NA" if admissions_trend ==""
tostring diag_importance, replace
replace diag_importance ="NA" if diag_importance ==""
tostring spesialist_contact, replace
replace spesialist_contact ="NA" if spesialist_contact =="."
replace tjenestetype_kpr ="NA" if tjenestetype_kpr ==""
tostring substance_use, replace
replace substance_use ="NA" if substance_use =="."
replace visit_trend ="NA" if visit_trend ==""
tostring inperson_contact_kpr , replace
replace inperson_contact_kpr ="NA" if inperson_contact_kpr =="."
tostring speech_therapy_kpr , replace
replace speech_therapy_kpr ="NA" if speech_therapy_kpr =="."
tostring fys_group_kpr , replace
replace fys_group_kpr ="NA" if fys_group_kpr =="."
tostring diag_importance_kpr , replace
replace diag_importance_kpr ="NA" if diag_importance_kpr =="."
tostring pasient_kjonn_verdi , replace
replace pasient_fylkesnummer ="NA" if pasient_fylkesnummer ==""
tostring utlevering_resepttype_verdi , replace
replace utlevering_resepttype_verdi ="NA" if utlevering_resepttype_verdi =="."
drop treatment_duration_days
tostring patient_overall_adherence_status , replace
replace patient_overall_adherence_status ="NA" if patient_overall_adherence_status =="."
tostring interaction_flag , replace
replace interaction_flag ="NA" if interaction_flag =="."
replace atc_category ="NA" if atc_category ==""
tostring module_number_atc , replace
replace module_number_atc ="NA" if module_number_atc =="."
tostring module_number_icd, replace
drop if module_number_icd=="."
export delimited using /home/mohsen.g.askar/workbench/Files_After_Manipulation/Ready_To_Model/Data_all_ready_to_ML_With_Modules_ID.csv, replace

local vars pasient_fylkesnummer pasient_kjonn_verdi samtykkekompetanse aktivitetskategori hdg innmatehast fagomrade henvfratjeneste henvtiltjeneste omsnivahenv admission_season admissions_trend tjenestetype_kpr visit_trend inperson_contact_kpr speech_therapy_kpr fys_group_kpr diag_importance_kpr utlevering_resepttype_verdi interaction_flag atc_category module_number_atc patient_overall_adherence_status

foreach var of local vars {
	replace `var' = "unknown" if `var' == "NA"
}

export delimited "data_clean.csv", replace
// to python for ML pipeline start
