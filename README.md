# Workshop2022

Online practicals:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/binder-sandbox/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FUCL-EO%252FWorkshop2022%26urlpath%3Dlab%252Ftree%252FWorkshop2022%252Fnotebooks%26branch%3Dmain)

[Workshop information webpage](https://ucl-eo.github.io/Workshop2022)


Installation
============

1. Install Anaconda: https://docs.anaconda.com/anaconda/install/index.html

2. Download the repository:
```
git clone https://github.com/UCL-EO/Workshop2022.git
```
3. Install packages:
```
cd Workshop2022/
conda env create -n uclnceo --force -f environment.yml
```
4. Enable extensions

```
#!/bin/bash

# activate uclnceo envrionment
conda activate uclnceo

jupyter serverextension enable --py nbgitpuller --sys-prefix
jupyter contrib nbextension install --user
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable python-markdown/main

# Install a JupyterLab extension for demonstration purposes
jupyter labextension install @jupyterlab/geojson-extension 
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-leaflet
jupyter trust notebooks/*.ipynb *.ipynb

# kernel setup
python -m ipykernel install --name=conda-env-uclnceo  --display-name 'conda env:uclnceo' --user
```

5. Run `jupyter-lab` and load the notebooks.


Welcome to the Using Earth observation for Crop Monitoring (iEOCM) Workshop! You will find the following content:  

Tamale Workshop
===============

MORNING SESSION  [09:00-11:00 2 hours]
* Welcome: GSSTI Director [5 mins]
* Welcome: Structure of the workshop: Kofi Asare GSSTI [5 mins]
* Perspectives and needs: MOFA regional director [20 mins]
* Cropping issues in North of Ghana: ADRA Kankam Boadu [20 mins]
* [Overview and achievements](https://www.icloud.com/iclouddrive/071-Ewsowz1xVjf694rdhIa7g#Workshop_2022_Lewis_Nr_FinalV3): Prof Lewis [20 mins]
* Role of ground data and summary findings from 2021: Kofi Asare GSSTI [20 mins]
* [Weather and yield in-season forecast](https://assimila.egnyte.com/dl/ABShXrvth8/weather_and_in-season_forecasting.pptx_): Nicola Pounder, Assimila [20 mins]
* Discussion and future priorities [GSSTI to chair] [15 mins]


Accra Workshop
===============

MORNING SESSION  [09:00-11:00 2 hours]
* Welcome: GSSTI Director [5 mins]
* Welcome: Structure of the workshop: Kofi Asare GSSTI [5 mins]
* Perspectives and needs: MOFA yield estimation director [20 mins]
* [Overview and achievements](https://www.icloud.com/iclouddrive/071-Ewsowz1xVjf694rdhIa7g#Workshop_2022_Lewis_Nr_FinalV3): Prof Lewis [20 mins]
* Role of ground data and summary findings from 2021: Kofi Asare GSSTI [20 mins]
* Climate and yield: Dilys MacCarthy [20 mins]
* [Weather and yield in-season forecast](https://assimila.egnyte.com/dl/ABShXrvth8/weather_and_in-season_forecasting.pptx_): Nicola Pounder, Assimila [20 mins]
* Discussion and future priorities [GSSTI to chair] [15 mins]


Practical Session [11:20:12:40] [1:40-2:00]
===========================================

Groups of 5, run each session x 4 

[Practicals](https://mybinder.org/v2/gh/jgomezdans/binder-sandbox/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FUCL-EO%252FWorkshop2022%26urlpath%3Dlab%252Ftree%252FWorkshop2022%252Fnotebooks%26branch%3Dmain):

• You may want to [register for a GEE account](https://earthengine.google.com/signup/) before starting the practicals.

-   [DEA showcase]: DEA Edward Boamah
-   [Biophysical binder (Feng)](https://github.com/UCL-EO/Workshop2022/issues/13#issue-1150356786): EO & field data explorer
-   [Weather data and within-season demonstrator??] (Nicola)
-   [Model data assimilation](https://docs.google.com/presentation/d/1D8o2c1XbBGy-455h7CiLuMQjWnUDZKK1N10Qq9OO_9g/edit?usp=sharing) (Jose/Hongyuan): Crop model sliders, DA-Me, yield context plot
-   [Maize mapper](https://github.com/UCL-EO/Workshop2022/blob/main/notebooks/Q2_Mazie_Mapper.ipynb)  GEE visualisation and MOFA yield (Qingling)
-   [Field measurements demo] Kenneth


Workshop technical presentations 2:30-4:00
==========================================

(15 mins)

-  Classification (cau)
-  DEA classification in Ghana 
-  [Crop modelling & ensembles](https://github.com/UCL-EO/Workshop2022/blob/main/presentations/WOFOST%20Crop%20Model%20and%20Ensemble%20Generation-Hongyuan.pptx): (Hongyuan)
-  [EO interpretation](https://liveuclac-my.sharepoint.com/:p:/g/personal/ucfafyi_ucl_ac_uk/EVlH9lVhJnlKi4oHdItIhLkBoSCtOasAouhNuseoSlaVeg?e=7egrDY) (talk on issues and methods) (feng)
-  [Introduction to data assimilation](https://docs.google.com/presentation/d/1D8o2c1XbBGy-455h7CiLuMQjWnUDZKK1N10Qq9OO_9g/edit?usp=sharing) (Jose/Hongyuan/CAU)
-  [Ensemble Crop DA](https://docs.google.com/presentation/d/1S3TkJICEMmKcBnz8WcAWNWjOXrDoAWFizELSg1VRELg/edit?usp=sharing) (jose)
-  Discussion and Reflections: GSSTI (Kofi to lead)

## Layout
* `data`: folder for data. Keep small e.g. CSV files etc here. If you need larger data, download via `postBuild` script. `carto` subfolder has maps of Ghana.
* `figures`: Figures, logos, ....
* `notebooks`: Where notebooks go
* `presentations`: pdf version of presentations (needed?)
* `wkshp_codes`: python code to be used by notebooks
