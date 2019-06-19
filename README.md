# Early ICU Mortality Predicition #
Predicit hospital mortality for patient admitted to ICU, using only data collected from the first 6 hours of ICU admission.  
Supervised learning algorithms used, accuracy:  
Logistic Regression: 0.73  
Decision Tree: 0.69  
Random Forest: 0.75  
Neural Net: 0.76  
Support Vector Machine(SVM): 0.73  
SVM-rbf-kernal: 0.76  

For more results and discussion see detailed report. 

Conclusion:  
- The Neural Net produced the best accuracy at an accuracy of 76%  
- The most important features from the Logistic Regression and Random forest are:  
  - Mean respiration rate across the 6 hours  
  - Maximum pCO2 across the 6 hours  
  - Minmum glucose measurement across the 6 hours  
  - Admission age  
