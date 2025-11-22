# aerospace_trajectory_anomaly_detection
This project implements a complete pipeline for detecting anomalies in aircraft trajectories using real flight data from the OpenSky Network.   

It combines data preprocessing, rule-based detection methods, and lightweight machine-learning algorithms to identify abnormal behaviors such as sudden altitude changes, speed deviations, or irregular flight paths.  

The project includes:   

- Real trajectory datasets sourced from OpenSky Network 

- Data cleaning and feature extraction (altitude, speed, heading, geo-coordinates)  

- Baseline anomaly detection (thresholds, derivatives)  

- ML-based outlier detection    

- Visualizations of normal vs. anomalous segments   

- Optional interactive interface (Streamlit)    

The goal is to explore trajectory analysis techniques relevant to aerospace systems, focusing on robustness, safety, and real-world data interpretation.    

## OpenSky API  

### Prerequisites   
- Python 3.9+   
- pip or conda  

### Install dependencies    

``` pip install -r requirements.txt ``` 

Typical dependencies include :    
- numpy 
- pandas    
- matplotlib    
- scikit-learn  

### OpenSky CLIENT API
Online access to OpenSky :  
To access data, you must first ensure to follow throught the specified steps :  
1. Create an account on the OpenSky Network web site (link : )   
2. On your **Account** page (default one) download your credentials     
3. Add a ``` .env ``` file at the racine of the project and enter your credentials like below :     
   ``` OPEN_SKY_CLIENT_ID=your_client_id    OPEN_SKY_CLIENT_SECRET=your_client_secret ```   

### Usage   
Make sure to move into the ```src/``` folder to execute the following commands  

#### Trajectory loader  
Fetch the ```.csv``` file into the ```data/raw``` folder with the chosen aircraft list :    
``` python trajectory_loader.py ```     

Default aircrafts icao24 are available, but feel free to modify the ``` trajectory_loader.py ``` file for new data  
  
#### Preprocess raw data    
Clean and sort the data to create anomaly detection features    
``` python data_preprocessing.py ```    


