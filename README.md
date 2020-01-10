## Seq2Cluster: Learning Anomalies in Asynchronous Co-evolving Time Series
---  
### Overview  
to be updated
### DataSet  
* **TwoLeadECG**  
TwoLeadECG is an ECG dataset taken from physionet by Eamonn Keogh. Specifically, the data is from MIT-BIH Long-Term ECG Database (ltdb) Record ltdb/15814, begin at time 420, ending at 1019. The task is to distinguish between signal 0 and signal 1.([dataset](http://www.timeseriesclassification.com/description.php?Dataset=TwoLeadECG))
* **FacesUCR**  
FacesAll is rotationally aligned version of FaceAll with a different train/test split.([dataset](http://www.timeseriesclassification.com/description.php?Dataset=FacesUCR))
* **ProximalPhalanxTW**  
([dataset](http://www.timeseriesclassification.com/description.php?Dataset=ProximalPhalanxTW))
* **MedicalImages**  
The data are histograms of pixel intensity of medical images. The classes are different human body regions. This dataset was donated by Joaquim C. Felipe, Agma J. M. Traina and Caetano Traina Jr.([dataset](http://www.timeseriesclassification.com/description.php?Dataset=MedicalImages))
### Requirements  
* python 3.*
* tensorflow 1.14
* numpy
* matplotlib
* pwlf (time series segmenatation)
* tqdm
* scikit-learn
### Usage  
For dataset TwoLeadECG  
```
sh run_tlecg.sh
```
For dataset FacesUCR  
```
sh run_faceucr.sh
```
For dataset ProximalPhalanxTW  
```
sh run_pptw.sh
```
For dataset MedicalImages  
```
sh run_medical.sh
```
---  
