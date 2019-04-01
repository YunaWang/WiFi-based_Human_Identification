# WiFi-based_Human_Identification

The proposed system can identify a person with the accuracy of 100\% to 97.80\% from a group of 5 to 25 people respectively, indicate this system is effective. 

# Human identification project processing

Step_1. Preprocessing

          Preprocessing -> Prepro_CSI.m

          Run: Prepro_CSI('../Raw_Data/', '../Result/test/',0);

Step_2. Denoise

          Denoise -> svd4H.m

          Run: svd('../Result/Preprocessing', '../Result/Denoise', 576, 432);

Step_3. Feature extraction

          Feature_Extraction -> gabor.py

Step_4. Classification

          Classifier -> main.m

          Run: load('parameters.mat');
               results = mainhub2_Yuna('../Result/Feature_Extraction', parameters);

*************************************************************************************************

# WiFi based human identification dataset: Raw_Data

1) 26 people data have been collected.

2) Each person has 2 directions data, in/out.

3) Each person in each diraction need to be collected 10 times.

4) For 25 people identifaication, we did not use Liu's data.


          
