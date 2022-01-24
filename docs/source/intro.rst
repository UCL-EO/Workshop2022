Crop models
===============================

Purpose of this section
------------------------
This section tells us all about crop models.

Overview of the 'Monteith' approach
------------------------------------

In the production efficiency approach to modelling NPP (or GPP) (the 'Monteith' approach, (Monteith, 1972, 1977)), a linear relationship is assumed between (non limited) canopy photosynthesis and absorbed PAR (Photosynthetically Active Radiation, i.e. the amount of shortwave radiation that is absorbed by the canopy).


.. math::

    GPP  = PAR \times   f_{PAR}  \times  (\epsilon \times  C_{drm}) \times  scalars


.. math::

   NPP = GPP - R_a


Here, :math:`GPP` is the Gross Primary Productivity (:math:`g C m^{-2}`), :math:`\epsilon` is the 'light use efficiency' (LUE)  (g dry matter per MJ PAR), :math:`C_{drm}` is the carbon content of dry matter (0.45 :math:`gC g^{-1}`), :math:`f_{PAR}` is  the fraction of PAR absorbed by the canopy (also known as fAPAR), and :math:`PAR` is Photosynthetically Active Radiation (:math:`MJ m^{-2}`) or the downwelling shortwave radiation multipled by the the PAR fraction  of direct and diffuse illumination taken together. 

The :math:`scalars` represent multiplicative  environmental constraints that are typically meteorologically derived (i.e. limiting factors).

:math:`NPP` is the Net Primary Productivity (:math:`g C m^{-2}`) and :math:`R_a` is the autotrophic respiration (:math:`g C m^{-2}`).

