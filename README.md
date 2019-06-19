# Early ICU Mortality Predicition #
Predicit hospital mortality for patient admit#ted to ICU, using only data collected from the first 6 hours of ICU admission.

## Results ##
- The Neural Net produced the best accuracy at an accuracy of 77%
![Alt text](/report_and_results/Model_results.jpg?raw=true "Various Model Results")

## Most Important Features ##
![Alt text](/report_and_results/feature_importance.jpg?raw=true "Feature Importance as Identified by RF")
- The most important features from the Random forest are:
  - Mean respiration rate across the 6 hours
  - Maximum pCO2 across the 6 hours
  - Minmum glucose measurement across the 6 hours
  - Admission age

For more results and discussion see [detailed report](/report_and_results/report.pdf "see detailed report")
