1. Data Collection and Preprocessing

   The system uses the CICIDS dataset to train machine learning models for detecting normal and malicious network traffic.
   Datasets include:

       •	Friday-WorkingHours-Morning.pcap_ISCX.csv – Normal traffic data.
    
       •	Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv – Distributed Denial of Service attacks.
    
       •	Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv – Port scanning attacks.
    
       •	Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv – Web-based attacks (SQL injection, XSS, brute force). 

  
2.  Data Preprocessing

  Data preprocessing is performed to improve data quality, remove inconsistencies, and prepare network traffic features for machine learning.
  
  The following preprocessing steps are implemented:
  
  • Dataset Merging: Multiple CSV files are combined into a single dataset, duplicate records are removed, and the index is reset.
  
  • Data Cleaning: Column names are cleaned by removing extra spaces and unwanted characters. Attack labels are standardized into a consistent format.
  
  • Attack Category Mapping: Different attack labels are grouped into seven unified categories:
  
    Normal
    
    DDoS
   
    PortScan
    
    BruteForce
    
    WebAttack
    
    Bot
    
    Infiltration
  
  • Missing Value Handling: Rows with missing labels are removed, while missing and infinite numerical feature values are replaced with zeros.
  
  • Label Encoding: LabelEncoder from Scikit-learn converts categorical attack labels into numerical values.
  
  • Feature Selection: Only numerical network traffic features are selected for machine learning model training.
  
  • Feature Scaling: StandardScaler is applied to normalize feature values and prevent large numerical ranges from dominating model predictions.
  
  • Class Balancing: SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples for minority attack classes to reduce class imbalance and improve detection performance.

<img width="839" height="290" alt="image" src="https://github.com/user-attachments/assets/8477d60c-0b50-4788-a841-1bb4e5ed0ad3" />


3.  Resulting Preprocessed Dataset


  After preprocessing, the final dataset contains:

    o Cleaned and standardized numerical features.
    
    o Seven encoded attack categories.

    o Balanced class distribution using SMOTE.
    
    o Scaled features ready for machine learning.
    
    o Improved data quality for model training and evaluation.
    
    o This preprocessing pipeline helps machine learning models identify patterns in normal and malicious network traffic more effectively.


4.  Model Evaluation


  The trained machine learning models are evaluated using an unseen testing dataset to measure their prediction performance and generalization ability.

  The following evaluation metrics are used:


  Evaluation Metric	With their Purpose
  
    o Accuracy:Measures the overall percentage of correct predictions.
    
    o Confusion Matrix:Shows correct and incorrect predictions for each attack category.
    
    o Precision:Measures how many predicted attacks are actually attacks.
    
    o Recall:Measures how many actual attacks are correctly detected.
    
    o F1-Score:Combines precision and recall into a single performance metric.
    
    o ROC Curve and AUC:Evaluates the model's ability to distinguish between different classes using a One-vs-Rest approach.
    
    o Precision-Recall Curve:Evaluates the balance between precision and recall, especially for minority attack classes.
    
    o Feature Importance: Identifies the most influential network traffic features using Random Forest.
    
    o Model Selection: Random Forest is selected as the preferred model based on its evaluation performance, classification capability, and robustness in detecting different types of network attacks.
    
<img width="858" height="47" alt="image" src="https://github.com/user-attachments/assets/e6cc67b5-1b25-46c5-9ac1-72decbd55fb7" />

5. Model Saving and Deployment

After evaluation, the best-performing machine learning model and preprocessing objects are saved using Python's Pickle module.

    
    o pickle.dump(best_model, open("models/best_model.pkl", "wb"))
    o pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    o pickle.dump(le, open("models/label_encoder.pkl", "wb"))

The following files are saved:

best_model.pkl – Stores the trained machine learning model.

scaler.pkl – Stores the feature scaling object.

label_encoder.pkl – Stores the encoded attack class mapping.

These saved files are loaded into the Flask web application to perform real-time cyberattack prediction without requiring model retraining.
