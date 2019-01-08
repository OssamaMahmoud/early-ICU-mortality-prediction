COPY (
SELECT intime, icud.subject_id, icud.hadm_id, icud.icustay_id, icustay_seq, los_hospital, los_icu, icud.hospital_expire_flag, 

	gender, admission_age, height_first, weight_first,   

	HeartRate_Min, HeartRate_Max, HeartRate_Mean,
	SysBP_Min, SysBP_Max, SysBP_Mean,
	DiasBP_Min, DiasBP_Max, DiasBP_Mean,
	MeanBP_Min, MeanBP_Max, MeanBP_Mean,
	RespRate_Min, RespRate_Max, RespRate_Mean, 
	TempC_Min, TempC_Max, TempC_Mean,
	SpO2_Min, SpO2_Max, SpO2_Mean,
	vfd.Glucose_Min, vfd.Glucose_Max, vfd.Glucose_Mean ,

	--blood work Max, mi --maybe should remove specimen
	SPECIMEN, abgfd.AADO2_MAX, abgfd.BASEEXCESS_MAX, abgfd.BICARBONATE_MAX_bgs,
	abgfd.TOTALCO2_MAX, abgfd.CARBOXYHEMOGLOBIN_MAX, abgfd.CHLORIDE_MAX_bgs,
	abgfd.CALCIUM_MAX, abgfd.HEMATOCRIT_MAX_bgs, abgfd.HEMOGLOBIN_MAX, 
	abgfd.INTUBATED_MAX, abgfd.LACTATE_MAX, abgfd.METHEMOGLOBIN_MAX, abgfd.O2FLOW_MAX,
	abgfd.FIO2_MAX, abgfd.SO2_MAX, abgfd.PCO2_MAX, abgfd.PEEP_MAX, abgfd.PH_MAX, abgfd.PO2_MAX,
	abgfd.POTASSIUM_MAX_bgs, abgfd.REQUIREDO2_MAX, abgfd.SODIUM_MAX_bgs, abgfd.TEMPERATURE_MAX,
	abgfd.TIDALVOLUME_MAX, abgfd.VENTILATIONRATE_MAX, abgfd.VENTILATOR_MAX, abgfd.AADO2_MIN,
	abgfd.BASEEXCESS_MIN, abgfd.BICARBONATE_MIN_bgs, abgfd.TOTALCO2_MIN, abgfd.CARBOXYHEMOGLOBIN_MIN, abgfd.CHLORIDE_MIN_bgs,
	abgfd.CALCIUM_MIN, abgfd.HEMATOCRIT_MIN_bgs, abgfd.HEMOGLOBIN_MIN, 
	abgfd.INTUBATED_MIN, abgfd.LACTATE_MIN, abgfd.METHEMOGLOBIN_MIN, abgfd.O2FLOW_MIN,
	abgfd.FIO2_MIN, abgfd.SO2_MIN, abgfd.PCO2_MIN, abgfd.PEEP_MIN, abgfd.PH_MIN, abgfd.PO2_MIN,
	abgfd.POTASSIUM_MIN_bgs, abgfd.REQUIREDO2_MIN, abgfd.SODIUM_MIN_bgs, abgfd.TEMPERATURE_MIN,
	abgfd.TIDALVOLUME_MIN, abgfd.VENTILATIONRATE_MIN, abgfd.VENTILATOR_MIN ,

	--labs

	lfd.ANIONGAP_min, lfd.ANIONGAP_max, lfd.ALBUMIN_min, lfd.ALBUMIN_max, 
	lfd.BANDS_min, lfd.BANDS_max, lfd.BICARBONATE_min_bgs, lfd.BICARBONATE_max_bgs,
	lfd.BILIRUBIN_min, lfd.BILIRUBIN_max, lfd.CREATININE_min, lfd.CREATININE_max, 
	lfd.CHLORIDE_min, lfd.CHLORIDE_max, 
	lfd.HEMATOCRIT_min, lfd.HEMATOCRIT_max, lfd.HEMOGLOBIN_min, lfd.HEMOGLOBIN_max, 
	lfd.LACTATE_min, lfd.LACTATE_max, lfd.PLATELET_min, lfd.PLATELET_max, lfd.POTASSIUM_min, 
	lfd.POTASSIUM_max, lfd.PTT_min, lfd.PTT_max, lfd.INR_min, lfd.INR_max, lfd.PT_min, lfd.PT_max, 
	lfd.SODIUM_min, lfd.SODIUM_max, lfd.BUN_min, lfd.BUN_max, lfd.WBC_min, lfd.WBC_max ,

	--GCS DONE, minGCS
	gcs.minGCS, gcs.gcsEyes, gcs.gcsverbal, gcs.gcsMotor, gcs.endotrachflag as intubated ,

	oasis.OASIS, oasis.OASIS_PROB, SAPSII.sapsii, SAPSII.sapsii_prob
		-- just gotta add urine


FROM 
	icustay_detail icud, heightweight, 
	vitalsfirst6 vfd, aggbloodgasfirst6 abgfd,
	labsfirst6 lfd, gcsfirst6 gcs, oasis, sapsii


where 
	icud.icustay_id=lfd.icustay_id AND 
	icud.icustay_id=heightweight.icustay_id AND
	icud.icustay_id=vfd.icustay_id AND 
	icud.icustay_id=abgfd.icustay_id AND 
	icud.icustay_id=lfd.icustay_id AND
	icud.icustay_id=gcs.icustay_id AND
	icud.icustay_id=oasis.icustay_id AND
	icud.icustay_id=sapsii.icustay_id AND

	admission_age>17 AND admission_age<150 


ORDER BY icud.subject_id,  icud.intime
) TO STDOUT WITH CSV HEADER;
