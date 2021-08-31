# causal inference and machine learning for earth system science
###
The repository contains exemplary studies and codes of using advanced causal inference and machine learning methods to solve critical earth system scientifical problems. In this study, we used causal inference method to capture the causal relationship between evaporative fraction (EF, indicator of surface energy partitioning patterns) and other variables such as air temperature, gross primary productivity, downwelling shortwave radiation, precipitation, vapor pressure deficit, and soil moisture, and leaf area index. Then, we compared the difference of the causal networks calculated by the observed data and CMIP6 data, to reveal the uncertainties of model structures. In addition, we built machine learning surrogate models to investigate and reduce CMIP6 model biases originated from parameters and diverse model structures.  
###
### Functions  
In the “causal_inference_ann_surrogate_demo.py” file, there are four important functions used in our research.    
1) Causal relationship detection in observations  
Here, we used PCMCI frame work to detect the causality between EF and other variables using observed data from fluxnet and MODIS data, and code can be seen in the function named “causal_inference_obs()”. PCMCI is proposed by Jakob Runge, which is a good tool to detect and quantify causal associations in large nonlinear time series datasets. Detailed tutorial of PCMCI can be seen in (https://github.com/jakobrunge/tigramite).  
2) Causal relationship detection in models of CMIP6  
The function of "causal_inference_cmip()" calculates the causal relationships between EF and other variables using CMIP6 simulated variables.   
3) Machine learning surrogates models in CMIP6  
Here we built Artificial Neural Network (ANN) models to surrogate CMIP6 models. The related code can be seen in “CMIP_ANN_surrogate()”.  
4) Observation calibrated ML surrogate models  
In the function “CMIP_ANN_surrogate_finetune()”, We used observation to finetune the ANN surrogate model.
  
### Data availability  
The original observed datasets from fluxnet are available at https://fluxnet.org/. The CMIP datasets are available at https://esgf-node.llnl.gov/projects/cmip6/. The observed LAI variables from MODIS satellite data product MCD15A3H are available at https://lpdaac.usgs.gov/products/mcd15a3hv006/.    
### Note  
Details of the research will be seen in the paper “Understanding and reducing the biases of land surface energy flux partitioning in CMIP6 land models” (The manuscript is ready to submit, and it will be online available soon). If you have any questions of the code, please contact yuankunxiaojia@gmail.com or lifa.lbnl@gmail.com.  