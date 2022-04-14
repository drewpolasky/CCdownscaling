This package provides implementation of several statistical climate downscaling techniques, as well as evaluation tools for downscaling outputs. 

installation:    
numpy    
scipy    
tensorflow=1.13.1    
xarray  
pandas    
scikit-learn   
matplotlib   
seaborn   
statsmodels    
sliced     
  

use:
An example use case for downscaling precipitation at Chicago O'Hare airport can be found in the example folder.
This example requires some example data, which can be downloaded from: 


once that data is in place, the example can be run with: 
python ./example/ohare_example.py 

And runs through several downscaling methods, including SOM, random forest, and quantile mapping. All these methods are then compared on PDF skill score, KS test, RMSE, bias, and autocorrelation, along with the undownscaled values from the NCEP reanalysis











