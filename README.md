# COVID-19-cough-breath-speech-classification
 Objective: Our objective is to classify COVID-19 positive and COVID-19 negative users by analysing cough sounds, breath sounds, and speech signals. The analysis on cough and breath sounds was done on the following 5 classes for the first time: COVID-19 positive with cough, COVID- 19 positive without cough, healthy person with cough, healthy person without cough, and an asthmatic cough. For speech sounds there were only two classes: COVID-19 positive, and COVID-19 negative. 

# Methodology adopted:
![image](https://user-images.githubusercontent.com/97305078/184231317-8cd5456d-bb46-43c3-8451-37fa38f24800.png)

# Experiments: 
Experiment 1: Here we do 3-class classifications from cough sounds to distinguish between COVID-19 positive users who have a cough as a symptom (COVID-19 cough), and COVID-19 negative users who have a cough (non-COVID-19 cough), asthma cough. 
Experiment 2: Expanding our Experiment 1, in this, we perform 5- class cough classification with the following 5 classes: COVID-19 positive users who have a cough as a symptom (COVID-19 cough), COVID-19 positive users who don’t have a cough as a symptom (COVID-19 no cough), COVID-19 negative users who have a cough (non-COVID-19 cough), the COVID-19 negative users who don’t have a cough (non-COVID-19 no cough), and asthma cough.  
Experiment 3: Using breath sounds to distinguish between users who have declared that they are COVID-19 positive (COVID-19 breath), healthy users (non-COVID-19 breath), and users having asthma.  
Experiment 4: Using breath sounds to distinguish between all 5 classes i.e. breath sounds from COVID-19 positive users who have a cough as a symptom (COVID-19 breath), breath sounds from COVID- 19 positive users who don’t have a cough as a symptom, breath sounds from COVID-19 negative users who have a cough, breath sounds from COVID-19 negative users who don’t have a cough, and breath sounds from users having asthmatic cough.  
Experiment 5: Combine two modalities i.e. cough and breath to distinguish between all 5 classes.  
Experiment 6: Distinguish between COVID-19 positive and COVID- 19 negative users from their speech samples only.  
Experiment 7: Here we do a simple binary classification between COVID-19 positive cough sounds and COVID-19 negative cough sounds.

# Feature visualization using t-SNE
![image](https://user-images.githubusercontent.com/97305078/184231572-e0951bd1-3b14-4fa2-b0d6-8cfe21f7ef14.png)

# Classification Results using k-NN classifier (10 fold CV)
![image](https://user-images.githubusercontent.com/97305078/184234148-b8bd0cc0-6780-43b3-9d31-aa983ef22706.png)

# Summary: 
Our proposed model can classify 5 types of cough sounds with an accuracy rate of 71.7%, 5 types of breath sounds with an accuracy rate of 72.2%, and 79.7% of speech sounds. The system offers the highest accuracy rate of 98.9% while performing bi-nary classification on COVID-19 and non-COVID-19 cough sounds.
