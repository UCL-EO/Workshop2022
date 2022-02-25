# Workshop2022
Workshop materials for crop monitoring in Ghana

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/binder-sandbox/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FUCL-EO%252FWorkshop2022%26urlpath%3Dlab%252Ftree%252FWorkshop2022%252Fnotebooks%26branch%3Dmain)

[docs](https://ucl-eo.github.io/Workshop2022/docs/index.html)
  
Welcome to the Using Earth observation for Crop Monitoring (iEOCM) Workshop! You will find the following content:  

Tamale Workshop
===============

MORNING SESSION  [09:00-11:00 2 hours]
* Welcome: GSSTI Director [5 mins]
* Welcome: Structure of the workshop: Kofi Asare GSSTI [5 mins]
* Perspectives and needs: MOFA regional director [20 mins]
* Cropping issues in North of Ghana: ADRA Kankam Boadu [20 mins]
* Overview and achioevements: Prof Lewis [20 mins]
* Role of ground data and summary findings from 2021: Kofi Asare GSSTI [20 mins]
* Weather and yield in-season forecast: Nicola Pounder, Assimila [20 mins]
* Discussion and future priorities [GSSTI to chair] [15 mins]


Accra Workshop
===============

MORNING SESSION  [09:00-11:00 2 hours]
* Welcome: GSSTI Director [5 mins]
* Welcome: Structure of the workshop: Kofi Asare GSSTI [5 mins]
* Perspectives and needs: MOFA yield estimation director [20 mins]
* Overview and achioevements: Prof Lewis [20 mins]
* Role of ground data and summary findings from 2021: Kofi Asare GSSTI [20 mins]
* Climate and yield: Dilys MacArthy [20 mins]
* Weather and yield in-season forecast: Nicola Pounder, Assimila [20 mins]
* Discussion and future priorities [GSSTI to chair] [15 mins]


Practical Session [11:20:12:30] [1:30-2:00]
===========================================

[Practicals](https://mybinder.org/v2/gh/jgomezdans/binder-sandbox/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FUCL-EO%252FWorkshop2022%26urlpath%3Dlab%252Ftree%252FWorkshop2022%252Fnotebooks%26branch%3Dmain):

-   [DEA showcase]: DEA Edward Boamah
-   [Biophysical binder (Feng)](https://github.com/UCL-EO/Workshop2022/issues/13#issue-1150356786): EO & field data explorer
-   [Weather data and within-season demonstrator??] (Nicola)
-   [Model data assimilation] (Jose/Hongyuan): Crop model sliders, DA-Me, yield context plot
-   [Maize mapper](notebooks/CAU_Interface_Ghana_V1.ipynb)  **** CAU classification,  MOFA yield (GEE) Qingling
-   [Field measurements demo] Kenneth
-   [LAI anywhere + DA-Me] Kofi
-   [ensemble function] Hongyuan


Workshop technical presentations. (15 mins)
==========================================

-  Crop modelling & ensembles: (Hongyuan)
-  EO interpretation (talk on issues and methods) (feng)
-  Classification (cau)
-  DEA classification in Ghana - details??
-  Crop model calibration (mcmc - cau for China)
-  Technical DA for in-field (Jose)
-  GSSTI or other interested group items? (Who/what?)
-  GSSTI - data collection technical details and lesson learned
-  UoGhana - crop type mapping??

## Layout
* `data`: folder for data. Keep small e.g. CSV files etc here. If you need larger data, download via `postBuild` script. `carto` subfolder has maps of Ghana.
* `figures`: Figures, logos, ....
* `notebooks`: Where notebooks go
* `presentations`: pdf version of presentations (needed?)
* `wkshp_codes`: python code to be used by notebooks
