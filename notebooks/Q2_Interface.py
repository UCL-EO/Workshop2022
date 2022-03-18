#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mofa_empirical import *
import geemap, ee
from mpl_toolkits import mplot3d
from datetime import datetime
import ipywidgets as widgets
from ipyleaflet import WidgetControl, Marker
from os import path


# In[2]:


# ****** widgets of interactive system
#year_range, crop_mask, data_type

submit = widgets.Button(description='Submit', button_style='primary', tooltip='Click me')
clear = widgets.Button(description='Clear')

style = {'description_width': 'initial'}

region_choice = widgets.Dropdown(
    description='Region',
    options=[
        'Northern Region',
        'Greater Accra Region',
        'Northern & Greater Accra Regions',
        'Ghana (whole country)'
    ],
    style = style,
    disabled = False,
    value='Northern Region'
)

year_range = widgets.Dropdown(
    description='Year',
    options=[
        str(x) for x in range(2006, 2021)
    ],
    style = style,
    value='2006'
)

crop_mask = widgets.Dropdown(
    description='Maize Mask',
    options=[
        'MODIS Land Cover Type 1 (IGBP) Grasslands: dominated by herbaceous annuals (<2m)',
        'MODIS Land Cover Type 2 (UMD) Grasslands: dominated by herbaceous annuals (<2m)',
        'MODIS Land Cover Type 3 (LAI) Grassland: dominated by herbaceous annuals (<2m) including cereal croplands',
        'MODIS Land Cover Type 2 (UMD) Cropland: at least 60% of area is cultivated cropland',
        'CAU Maize classification'
    ],
    style = style,
    disabled = False,
    value='MODIS Land Cover Type 2 (UMD) Grasslands: dominated by herbaceous annuals (<2m)'
)

data_source = widgets.Dropdown(
    description='Data source',
    options=[
        'Max LAI over growing season, between Jun and Nov',
        'ERA5 accummulated precipitation (mm) between Jun and Nov',
        'ERA5 mean temperature at 2m (deg K) between Jun and Nov',
        'CAU Maize Map',
        'MOFA yield statistics (1000 kg/ha)',        
        'Empirical Yield = MaxLAI*1500-700 (kg/ha)',
        'Maize Planting Area (ha)'
    ],
    style = style,
    value='Max LAI over growing season, between Jun and Nov'
)

# *** checkbox widgets ***
checkbox_mask = widgets.Checkbox(
    value = True, 
    description = "Apply maize mask",
    style = style)

output_widget = widgets.Output(layout={'border': '1px solid black'})
output_control = WidgetControl(widget=output_widget, position='bottomright')


# In[3]:


def create_maize_area_map_northern(aoi=ghana_district):
    # TO DO: can use "Crop_production_Data_SRID.csv" file instead
    df_crop_area = pd.read_csv('Qtrend/crop_area-Jose.csv', dtype={'name_adm2': str, 'maiz_r': np.float64})                                .sort_values('maiz_r', ascending=False)

    
    # set a new varible with value from df_maize TO CREATE A NEW MAP LAYER
    features = [] # list of ee.Features
    for i, dist in enumerate(aoi.aggregate_array('ADM2_NAME').getInfo()): 
        aoi_dist = aoi.filter(ee.Filter.eq('ADM2_NAME', dist))                                        .first().geometry()
        if dist in df_crop_area['name_adm2'].unique():
            
            maiz_a = df_crop_area[df_crop_area['name_adm2']==dist]['maiz_r']
            area = maiz_a.values[0] if len(maiz_a) >0 else -999
        else:
            #print(dist, 'has no planting area recorded.')
            area = -999
            
        new_feature = ee.Feature(aoi_dist)
        new_feature = new_feature.set({'id':i, 'maize_area':area, 'ADM2_NAME': dist})

        features.append( new_feature ) #aoi_dist.set('yield', yi) )
        
    maize_areas = ee.FeatureCollection(features)
    return maize_areas


# In[17]:


def create_mofa_yield_map(year, withPlantingArea=False, region='northern'):
    '''region options: 'northern', 'greater accra', or 'all'  '''
    region = region.upper()
    
    df = pd.read_csv('Qtrend/Ghana_Distirct_level_Crop_Yield_Data-upto2020.csv', sep=',', header=0)
    df_maize = df[(df['CROP']=='MAIZE') & (df['YEAR']== year)]
    
    if region!='ALL':
        df_maize = df_maize[df_maize['REGION'] == region]
    
    # Update Nothern districts names to match GUAL ADM2_NAMEs:
    df_maize = df_maize.replace('Tamale Metro','Tamale North Sub Metro')    
    df_maize = df_maize.replace('Central Gonja','Gonja Central')
    df_maize = df_maize.replace('East Mamprusi','Mamprusi East')
    df_maize = df_maize.replace('Tamale Metro','Tamale North Sub Metro')
    df_maize = df_maize.replace('Yendi','Yendi Municipal')
    df_maize = df_maize.replace('Ledzokuku-Krowor', 'Ledzokuku / Krowor')
    df_maize = df_maize.replace('Tatale/Sanguli', 'Tatale')
        
    # Update Other ditricts names too:
    df_maize = df_maize.replace('Bekwai muni (Amansie East)', 'Bekwai Municipal')
    df_maize = df_maize.replace('Bekwai municipal (Amansie East)', 'Bekwai Municipal')
    df_maize = df_maize.replace('Ejura Sekyedumase','Ejura Sekye Dumase')
    df_maize = df_maize.replace('Sekyere West', 'Mampong Municipal')
    df_maize = df_maize.replace('Mampong Municipal (Sekyere West) ', 'Mampong Municipal')    
    df_maize = df_maize.replace('Ejusu Juaben', 'Ejisu Juaben')
    df_maize = df_maize.replace('Bosumtwe-Atwima-Kwanwoma', 'Bosomtwe /Atwima / Kwanwoma')
    df_maize = df_maize.replace('Offinso', 'Offinso Municipal')
    df_maize = df_maize.replace('Obuasi Municipal (Adansi West)', 'Obuasi Municipal')
    df_maize = df_maize.replace('Adansi  South (East)', 'Adansi South')
    df_maize = df_maize.replace('K. M. A.', 'Kma')
    df_maize = df_maize.replace('Sunyani', 'Sunyani Municipal')
    
    def copy_dist_yield(dist_in, dist_out):
        df_tmp = df_maize[df_maize['DISTRICT']==dist_in]
        df_tmp.replace(dist_in, dist_out)
        print(df_tmp)
        df_maize = df_maize.append(df_tmp)
        
    #copy_dist_yield('Asutifi', 'Asutifi South')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Asutifi'].replace('Asutifi', 'Asutifi South'))
    df_maize = df_maize.replace('Asutifi', 'Asutifi North') # copy to South    
    df_maize = df_maize.replace('Dormaa', 'Dormaa Municipal')
    #copy_dist_yield('Sene', 'Sene East')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Sene'].replace('Sene', 'Sene East'))
    df_maize = df_maize.replace('Sene', 'Sene West') # East
    #copy_dist_yield('Nkoranza', 'Nkoranza North')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Nkoranza'].replace('Nkoranza', 'Nkoranza North'))
    df_maize = df_maize.replace('Nkoranza', 'Nkoranza North') # South
    df_maize = df_maize.replace('Techiman', 'Techiman Municipal')
    #copy_dist_yield('Upper Denkyira', 'Upper Denkyira West')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Upper Denkyira'].replace('Upper Denkyira', 'Upper Denkyira West'))
    df_maize = df_maize.replace('Upper Denkyira', 'Upper Denkyira East') # West
    df_maize = df_maize.replace('Twifo-Herman/L. Denkyira', 'Twifo Lower Denkyira')
    df_maize = df_maize.replace('Abura-Asebu-Kwamankese', 'Abura / Asebu / Kwamankese')
    df_maize = df_maize.replace('Asikuma-Odoben-Brakwa', 'Asikuma / Odoben / Brakwa')
    df_maize = df_maize.replace('Komenda-Edina-Eguafo-Abirem', 'Komenda Edna Eguafo / Abirem')
    df_maize = df_maize.replace('Ajumako-Essiam-Enyana', 'Ajumako-Enyan-Esiam')
    #copy_dist_yield('Awutu-Efutu-Senya', 'Awutu Senya West')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Awutu-Efutu-Senya'].replace('Awutu-Efutu-Senya', 'Awutu Senya West'))
    df_maize = df_maize.replace('Awutu-Efutu-Senya', 'Awutu Senya East Municipal') #'Awutu Senya West' 
    #copy_dist_yield('Gomoa', 'Gomoa East')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Gomoa'].replace('Gomoa', 'Gomoa East'))
    df_maize = df_maize.replace('Gomoa', 'Gomoa West') # East?
    df_maize = df_maize.replace('Cape Coast', 'Cape Coast Metro')
    #copy_dist_yield('Agona', 'Agona East')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Agona'].replace('Agona', 'Agona East'))
    df_maize = df_maize.replace('Agona', 'Agona West') # East?
    df_maize = df_maize.replace('Agona West (Swedru)', 'Agona West') # post-2012
    df_maize = df_maize.replace('Suhum Kraboa Coaltar', 'Suhum Municipal')
    df_maize = df_maize.replace('New Juaben', 'New Juaben Municipal')
    #copy_dist_yield('Afram Plains', 'Kwahu Afram Plains North')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Afram Plains'].replace('Afram Plains', 'Kwahu Afram Plains North'))
    df_maize = df_maize.replace('Afram Plains', 'Kwahu Afram Plains South') # Kwahu Afram Plains North?
    #copy_dist_yield('Manya Krobo', 'Upper Manya')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Manya Krobo'].replace('Manya Krobo', 'Upper Manya'))
    df_maize = df_maize.replace('Manya Krobo', 'Lower Manya') # Upper Manya
    df_maize = df_maize.replace('Lower Manya Krobo', 'Lower Manya')
    df_maize = df_maize.replace('Upper Manya Krobo', 'Upper Manya')
    df_maize = df_maize.replace('Tema', 'Tema Metropolis')
    df_maize = df_maize.replace('Bunkpurugu/Yunyoo', 'Bunkpurugu Yonyo')
    df_maize = df_maize.replace('Saboba/Cheriponi', 'Saboba')
    #copy_dist_yield('Tolon/Kumbungu', 'Kumbungu')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Tolon/Kumbungu'].replace('Tolon/Kumbungu', 'Kumbungu'))
    df_maize = df_maize.replace('Tolon/Kumbungu', 'Tolon') # Kumbungu
    #copy_dist_yield('Zabzugu/Tatale','Tatale')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Zabzugu/Tatale'].replace('Zabzugu/Tatale','Tatale'))
    df_maize = df_maize.replace('Zabzugu/Tatale', 'Zabzugu') # Tatale
    #copy_dist_yield('Builsa', 'Builsa South')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Builsa'].replace('Builsa', 'Builsa South'))
    df_maize = df_maize.replace('Builsa', 'Builsa North') # South
    #copy_dist_yield('Kasina-Nankana', 'Kasena Nankana East')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Kasina-Nankana'].replace('Kasina-Nankana', 'Kasena Nankana East'))
    df_maize = df_maize.replace('Kasina-Nankana', 'Kasena Nankana West') # Kasena Nankana East
    #copy_dist_yield('Talensi Nabdam', 'Nabdam')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Talensi Nabdam'].replace('Talensi Nabdam', 'Nabdam'))
    df_maize = df_maize.replace('Talensi Nabdam', 'Talensi') # Nabdam
    df_maize = df_maize.replace('Sissala West', 'Sissala  West')
    #copy_dist_yield('Jirapa-Lambussie', 'Lambussie Karni')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Jirapa-Lambussie'].replace('Jirapa-Lambussie', 'Lambussie Karni'))
    df_maize = df_maize.replace('Jirapa-Lambussie', 'Jirapa') # copy to 'Lambussie Karni'
    df_maize = df_maize.replace('Nadowli', 'Nadowli-Kaleo')
    #copy_dist_yield('Akatsi', 'Akatsi South')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Akatsi'].replace('Akatsi', 'Akatsi South'))
    df_maize = df_maize.replace('Akatsi', 'Akatsi North') # South
    df_maize = df_maize.replace('Keta', 'Keta Municipal')
    #copy_dist_yield('Ketu', 'Ketu South')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Ketu'].replace('Ketu', 'Ketu South'))
    df_maize = df_maize.replace('Ketu', 'Ketu North') # South
    #copy_dist_yield('North Dayi (Kpando)', 'Kpando Municipal')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='North Dayi (Kpando)'].replace('North Dayi (Kpando)', 'Kpando Municipal'))
    df_maize = df_maize.replace('North Dayi (Kpando)', 'North Dayi') # 'Kpando Municipal'
    df_maize = df_maize.replace('Kpando Municipal (North Dayi)', 'North Dayi')     
    df_maize = df_maize.replace('Hohoe', 'Hohoe Municipal')
    #copy_dist_yield('Nkwanta', 'Nkwanta South')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Nkwanta'].replace('Nkwanta', 'Nkwanta South'))
    df_maize = df_maize.replace('Nkwanta', 'Nkwanta North') # South
    df_maize = df_maize.replace('Ho', 'Ho Municipal')
    df_maize = df_maize.replace('Adaklu Anyigbe', 'Adaklu')
    df_maize = df_maize.replace('Shama Ahanta', 'Shama')
    df_maize = df_maize.replace('West Ahanta', 'Ahanta West')
    df_maize = df_maize.replace('Mporhor Wassa East', 'Wassa East')
    df_maize = df_maize.replace('Wassa West', 'Wassa Amenfi West')
    df_maize = df_maize.replace('Amenfi  West (Wassa Amenfi)', 'Wassa Amenfi Central') #?
    df_maize = df_maize.replace('Amenfi East', 'Wassa Amenfi East')
    df_maize = df_maize.replace('East Nzema', 'Nzema East')
    #copy_dist_yield('Aowin -Suaman', 'Suaman')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Aowin -Suaman'].replace('Aowin -Suaman', 'Suaman'))
    df_maize = df_maize.replace('Aowin -Suaman', 'Aowin') # Suaman
    df_maize = df_maize.replace('Bibiani-Anhwiaso-Bekwai', 'Sefwi Bibiani-Anhwiaso Bekwai')
    df_maize = df_maize.replace('Sefwi-Wiaso', 'Sefwi-Wiawso')
    #copy_dist_yield('Bia', 'Bia West')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='Bia'].replace('Bia', 'Bia West'))
    df_maize = df_maize.replace('Bia', 'Bia East') # West?
    df_maize = df_maize.replace('BOLGA MUNICIPAL', 'Bolgatanga Municipal')
    df_maize = df_maize.replace('BOLGA', 'Bolgatanga Municipal')
    df_maize = df_maize.replace('KwabreEast', 'Kwabre') 
    df_maize = df_maize.replace('Effutu (Winneba)', 'Effutu')
    df_maize = df_maize.replace('Birim Central', 'Birim Municipal')
    df_maize = df_maize.replace('Adenta-Madina', 'Adenta')
    df_maize = df_maize.append(df_maize[df_maize['DISTRICT']=='TALENSI NABDAM'].replace('TALENSI NABDAM', 'Talensi'))
    df_maize = df_maize.replace('TALENSI NABDAM', 'Nabdam')
    df_maize = df_maize.replace('Sekondi-Takoradi Municipal (STMA)', 'Sekondi Takoradi')
    df_maize = df_maize.replace('Prestea Huni-Valley ', 'Prestea / Huni Valley')
    df_maize = df_maize.replace('Sefwi Akontombra', 'Akontombra')
        
    #print(df_maize['DISTRICT'].unique())
    
    if withPlantingArea:
        ''' # use "Crop_production_Data_SRID.csv" file instead
        df_crop_area = pd.read_csv('Qtrend/crop_area-Jose.csv', \
                            dtype={'name_adm2': str, 'maiz_r': np.float64})\
                            .sort_values('maiz_r', ascending=False)'''

        df_crop_area = pd.read_csv('Qtrend/Crop_production_Data_SRID.csv',                             dtype={'DISTRICT': str, 'AREA (Ha)': np.float64, 'YEAR': int})                            .sort_values('AREA (Ha)', ascending=False)
        #print(df_crop_area['CROP'].unique())
        df_crop_area = df_crop_area[(df_crop_area['CROP']=='Maize') & (df_crop_area['YEAR']==year)]
        
    # set a new varible with value from df_maize TO CREATE A NEW MAP LAYER
    yield_features = [] # list of ee.Features
    for i, dist in enumerate(df_maize['DISTRICT'].unique()): 
        if dist in ghana_district.aggregate_array('ADM2_NAME').getInfo():
            yi = df_maize[df_maize['DISTRICT']==dist]['YIELD'].values[0]

            aoi_dist = ghana_district.filter(ee.Filter.eq('ADM2_NAME', dist))                                        .first().geometry()
            
            new_feature = ee.Feature(aoi_dist)
            new_feature = new_feature.set({'id':i, 'yield': yi, 'ADM2_NAME': dist})
            
            if withPlantingArea:
                #maiz_a = df_crop_area[df_crop_area['name_adm2']==dist]['maiz_r']
                maiz_a = df_crop_area[df_crop_area['DISTRICT']==dist]['AREA (Ha)']
                area = maiz_a.values[0] if len(maiz_a) >0 else -999
                new_feature = new_feature.set({ 'maize_area':area})

            #print(dist, '*******', new_feature.get('yield').getInfo())
            
            ## TO DO: PRE-STORE this layer, add re use 
            
            yield_features.append( new_feature ) #aoi_dist.set('yield', yi) )
        else:  print(dist, 'not in GAUL names')
            
    mofa_yield = ee.FeatureCollection(yield_features)
    #print(mofa_yield.first().getInfo())
    
    return mofa_yield

'''
# In[19]:


# make MOFA YIELD VECTOR to be stored: pre-2017, two layers: yield and planting area
for year in range(2006, 2017):
    print(year)
    mofa_yield = create_mofa_yield_map(year, withPlantingArea=True, region='all')
    geemap.ee_to_shp(mofa_yield, f'Qtrend/shp/MOFA_Yield_and_Area_vector_{year}.shp')   


# In[20]:


# make MOFA YIELD VECTOR to be stored: 1 layer of yield from 2017, as there is no planting area data
for year in range(2017, 2021):
    print(year)
    mofa_yield = create_mofa_yield_map(year, withPlantingArea=False, region='all')
    geemap.ee_to_shp(mofa_yield, f'Qtrend/shp/MOFA_Yield_vector_{year}.shp')   


# In[50]:


# make MOFA YIELD RASTER to be stored: single layer without Planting area
for year in range(2006, 2021):
    print(year)
    mofa_yield = create_mofa_yield_map(year, withPlantingArea=False, region='all')
    image = mofa_yield.reduceToImage(properties=['yield'], reducer=ee.Reducer.first()).rename('yield')
    #Map.addLayer(image)
    geemap.ee_export_image(image, filename=f'Qtrend/MOFA_Yield_Image_{year}.tif', scale=500,                            region=ghana_country.geometry(), file_per_band=False)
#Map   


# In[ ]:


# make MOFA AREA RASTER to be stored: single layer without Planting area
for year in range(2013, 20138):
    print(year)
    mofa_area = create_maize_area_map(year, aoi=ghana_district)
    image = mofa_area.reduceToImage(properties=['maize_area'], reducer=ee.Reducer.first()).rename('area')
    #Map.addLayer(image)
    geemap.ee_export_image(image, filename=f'Qtrend/MOFA_Area_Image_{year}.tif', scale=500,                            region=ghana_country.geometry(), file_per_band=False)   
'''


# In[21]:


def create_maize_area_map(year, aoi=ghana_district):
    # DONE: use "Crop_production_Data_SRID.csv" file instead
    df_crop_area = pd.read_csv('Qtrend/Crop_production_Data_SRID.csv',                             dtype={'DISTRICT': str, 'AREA (Ha)': np.float64, 'YEAR': int})                            .sort_values('AREA (Ha)', ascending=False)
    df_crop_area = df_crop_area[(df_crop_area['CROP']=='Maize') & (df_crop_area['YEAR']==year)]
        
    # set a new varible with value from df_maize TO CREATE A NEW MAP LAYER
    features = [] # list of ee.Features
    for i, dist in enumerate(aoi.aggregate_array('ADM2_NAME').getInfo()): 
        aoi_dist = aoi.filter(ee.Filter.eq('ADM2_NAME', dist))                                        .first().geometry()
        if dist in df_crop_area['DISTRICT'].unique():
            
            maiz_a = df_crop_area[df_crop_area['DISTRICT']==dist]['AREA (Ha)']
            area = maiz_a.values[0] if len(maiz_a) >0 else -999
            #print(dist, 'planting area recorded.') print(maiz_a)
        else:
            #print(dist, 'has no planting area recorded.')
            area = -999
            
        new_feature = ee.Feature(aoi_dist)
        new_feature = new_feature.set({'id':i, 'maize_area':area, 'ADM2_NAME': dist})

        features.append( new_feature ) #aoi_dist.set('yield', yi) )
        
    maize_areas = ee.FeatureCollection(features)
    return maize_areas


# In[22]:


def clear_clicked(button):
    Map.clear()
    Map.addLayer(outline2, {'palette': '999999'}, 'Ghana districts')
    Map.addLayer(outline, {'palette': '000000'}, 'Northern & Greater Accra districts')
    
    for control in [year_range, crop_mask, data_source, region_choice, region_choice]:
        control.index = 0
    checkbox_mask.value = True
    year_range.disabled = False
    
    
def checkmask_clicked(button):
    if checkbox_mask.value == True:
        crop_mask.disabled = False
    else:
        crop_mask.disabled = True
        
    
def submit_clicked(button):
    
    '''Add choice of region to focus on'''
    if region_choice.index == 0:
        aoi = northern_district
        Map.center = tamale_centre
        Map.zoom = 8
        #Map.setCenter(lon, lat, zoom)
    elif region_choice.index == 1:
        aoi = greateraccra_district
        Map.center = accra_centre
        Map.zoom = 9
    elif region_choice.index == 2:
        aoi = northern_district.merge(greateraccra_district) 
        Map.center = zoom_centre
        Map.zoom = 7
    elif region_choice.index == 3:
        ghana_province = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM0_NAME', 'Ghana'))
        aoi = ghana_province
        Map.center = zoom_centre
        Map.zoom = 7
        #Map.remove_layer('Northern & Accra districts')
        
    region_name = list(set(aoi.aggregate_array('ADM1_NAME').getInfo()))
    #print(region_name)
    
    #Map.clear_colorbar()
    output_widget.clear_output()
    if len(Map.layers) > base_no_layers: #>3
        # layers[-1] should be the vector polygpons
        Map.remove_layer(Map.layers[-2])
    if len(Map.controls) > base_no_controls: #>8
        #  to remove pallete
        Map.remove_control(Map.controls[-1])
        
    year = int(year_range.value)
    iMask = crop_mask.index if checkbox_mask.value else -1
    iData = data_source.index
        
    if iData == 0: #'Max LAI over growing season': 
        band_yearly, band_reduced = load_modis_band('Lai', year, aoi, 'max')
        data = band_reduced.clip(aoi)
        vis = colorized_vis['Lai']
        title = f'Max LAI {year}'
        legend_label = 'Max LAI over growing season'
        
    elif iData == 1: #'ERA5 accummulated precipitation (mm)'
        era5_in_season = load_era_band('total_precipitation', year, aoi)
        data = era5_in_season.sum().clip(aoi).multiply(100)
        vis = colorized_vis['Prec']
        title = f'Precipitation {year}'
        legend_label = 'Accummulated precipitation (mm)'
        
    elif iData == 2: #'ERA5 temperature at 2m'
        era5_in_season = load_era_band('temperature_2m', year, aoi)
        accum_temp = era5_in_season.sum().clip(aoi)
        data = accum_temp.divide(6) #.subtract(273.15) # six monthly temp    
        vis = {
                'min': 295.0,
                'max': 302.0,
                'palette': [
                "#000080","#0000D9","#4000FF","#8000FF","#0080FF","#00FFFF",
                "#00FF80","#80FF00","#DAFF00","#FFFF00","#FFF500","#FFDA00",
                "#FFB000","#FFA400","#FF4F00","#FF2500","#FF0A00","#FF00FF"] 
                }
        title = f'Temperaure {year}'
        legend_label = 'Mean temperature at 2m, Degree K'
        
    elif iData == 3: #'CAU Maize Map'
        maize_imgCol = ee.ImageCollection('users/xianda19/classification_result/2021/Ghana/maize_20210501_20211011_100percentSamples')
        data = maize_imgCol.mosaic().selfMask() 
        vis = {
                'min':0, 
                'max':1
                }
        title = "CAU maize layer"
        
        # ---------use the next two lines if to erode S2 to MODIS resolution:-------
        #kernel = ee.Kernel.square(1, 'meters')
        #data = data.focalMin(kernel = kernel, iterations=25)
        #---------------------------------------------------------------------------
        
    elif iData == 4: #'MOFA yield statistics'
        if liveProcess == True:
            for i, name in enumerate(region_name):
                print('Processing', name, '...')
                if i == 0:
                    mofa_yield = create_mofa_yield_map(year, region=name)
                else:
                    mofa_yield = mofa_yield.merge(create_mofa_yield_map(year, region=name))
        else: # load from pre-made vector layer
            if int(year_range.value) >= 2017:
                gee_fn = f'projects/ee-qinglingwu/assets/MOFA/MOFA_Yield_vector_{year}'
            else:
                gee_fn = f'projects/ee-qinglingwu/assets/MOFA/MOFA_Yield_and_Area_vector_{year}'         
            mofa_yield = ee.FeatureCollection(gee_fn) 
            
        data = mofa_yield.reduceToImage(properties=['yield'], reducer=ee.Reducer.first()).rename('yield').clip(aoi)
            
        vis = {
                'min': 0.5,
                'max': 3.5,
                'palette': ['FCFDBF', 'FDAE78', 'EE605E', 'B63679', '711F81', '2C105C']
                }
        title = f'MOFA Yield {year}'
        legend_label = 'MOFA Yield, 1000 kg/ha'
                  
    elif iData == 5: #'Empirical Yield = MaxLAI*1500-700 (kg/ha)'
        band_yearly, band_reduced = load_modis_band('Lai', year, aoi, 'max')
        maxlai_img = band_reduced.clip(aoi)
        data = maxlai_img.multiply(1.5).subtract(0.7).rename('yield_empirical')
        vis = {
                'min': 0.5,
                'max': 5.5,
                'palette': ['FCFDBF', 'FDAE78', 'EE605E', 'B63679', '711F81', '2C105C']
                }
        title = f'Empirical Yield calculated from MaxLAI {year}'
        legend_label = 'Empirical Yield, 1000 kg/ha'
        
    elif iData == 6: # Planted area of maize 
        if liveProcess == True:
            
            if region_choice.index == 0:
                # change YEARS option : disable
                year_range.disabled = True
                areas = create_maize_area_map_northern(aoi=aoi)
            else:
                areas = create_maize_area_map(year, aoi=aoi)
        
        else: # load from pre-made vector layer
            areas = ee.FeatureCollection(f'projects/ee-qinglingwu/assets/MOFA/MOFA_Yield_and_Area_vector_{year}')
       
        data = areas.reduceToImage(properties=['maize_area'], reducer=ee.Reducer.first()).rename('area').clip(aoi)
        
        if region_choice.index in [1,2]: # with accra
            vis = {
                'min': 50,
                'max': 1500,
                'palette': ['FCFDBF', 'FDAE78', 'EE605E', 'B63679', '711F81', '2C105C']
                }
        else:
            vis = {
                'min': 1000,
                'max': 8000,
                'palette': ['FCFDBF', 'FDAE78', 'EE605E', 'B63679', '711F81', '2C105C']
                }
        title = f'Maize Planting area'
        legend_label = 'maize area, ha' # TO-DO: double check if this is ha?
    
    if checkbox_mask.value:
        if iMask == 0: #LC_Type1 Grassland
            lc_crop, lc_all = load_modis_lc('LC_Type1', year, aoi, cropclasses=[10])
        elif iMask == 1: #LC_Type2 Grassland
            lc_crop, lc_all = load_modis_lc('LC_Type2', year, aoi, cropclasses=[10])
        elif iMask == 2: #LC_Type3 Grassland
            lc_crop, lc_all = load_modis_lc('LC_Type3', year, aoi, cropclasses=[1])
        elif iMask == 3: #LC_Type2 Cropland
            lc_crop, lc_all = load_modis_lc('LC_Type2', year, aoi, cropclasses=[12])
        elif iMask == 4: # CAU
            #kernel = ee.Kernel.square(10, units='meters') 
            maize_imgCol = ee.ImageCollection('users/xianda19/classification_result/2021/Ghana/maize_20210501_20211011_100percentSamples')
            lc_crop = maize_imgCol.mosaic().selfMask()           
            #lc_crop = lc_crop.focalMin(kernel = kernel, iterations=25)
            
        data = data.mask(lc_crop)
    
    Map.addLayer(data, vis, title)
    
    if iData != 3:
        Map.add_colorbar(vis, label=legend_label, layer_name=title,
                        orientation="vertical", transparent_bg=True)

    
    #previous_iData = iData
    # test prints
    #with output_widget: print(year, iMask, iData)
    if region_choice.index == 3:
        Map.addLayer(outline2, {'palette': '999999'}, 'Ghana districts')
    else:
        Map.addLayer(outline, {'palette': '000000'}, 'Northern & Greater Accra districts')
    #Map.addLayer(outline1, {'palette': '000000'}, 'Greater Accra districts')
        


# In[23]:


def handle_click(**kwargs):
    latlon = kwargs.get('coordinates')
    if kwargs.get('type') == 'click':
        Map.default_style = {'cursor': 'wait'}
        
        with output_widget:
            output_widget.clear_output()
            if len(Map.controls) > base_no_controls:
                Map.remove_control(Map.controls[-1])
                
            print(f'{latlon[0]:0.5f}', f'{latlon[1]:0.5f}')
            poi = ee.Geometry.Point(latlon[::-1])
            
            province = ghana_district.filterBounds(poi).aggregate_array('ADM1_NAME').getInfo()
            district = cities.filterBounds(poi).aggregate_array('ADM2_NAME').getInfo()
            country = cities.filterBounds(poi).aggregate_array('ADM0_NAME').getInfo()
            
            if len(province) > 0:
                # poi within ghana
                print(f"{district[0]}, {province[0]}")
            else:
                print(f"{district[0]}, {country[0]}")
            
    Map.default_style = {'cursor': 'pointer'}
    
    
def on_data_changed(change):
    if change['type'] == 'change' and change['name'] == 'value':
        #print("changed to %s" % change['new'])
        if change['new'] == 'Maize Planting Area (ha)': #if data_source.index == 6: # planting area
            if region_choice.index == 0: # Northern region
                year_range.disabled = True
            else:
                year_range.disabled = False
                year_range.options = [str(x) for x in range(2013, 2018)]
                if not (year_range.value in year_range.options):
                    year_range.value = '2013'
        elif change['new'] == 'CAU Maize Map':
            year_range.disabled = True
        else:
            year_range.disabled = False
            year_range.options = [str(x) for x in range(2006, 2021)]
            

def on_region_changed(change):
    if change['type'] == 'change' and change['name'] == 'value':
        #print("changed to %s" % change['new'])
        if data_source.value == 'Maize Planting Area (ha)':
            if change['new'] != 'Northern Region': 
                year_range.disabled = False
                year_range.options = [str(x) for x in range(2013, 2018)]
                if not (year_range.value in year_range.options):
                    year_range.value = '2013'
            else: # Northern
                year_range.disabled = True
                year_range.options = [str(x) for x in range(2006, 2021)]
        elif data_source.value == 'MOFA yield statistics (1000 kg/ha)':
            year_range.options = [str(x) for x in range(2006, 2021)]


# In[24]:


liveProcess = False

# ###### define the study area and corresponding maize mask ######
cities = ee.FeatureCollection("FAO/GAUL/2015/level2")
ghana_district = cities.filter(ee.Filter.eq('ADM0_NAME', 'Ghana'))
northern_district = ghana_district.filter(ee.Filter.eq('ADM1_NAME', 'Northern'))
greateraccra_district = ghana_district.filter(ee.Filter.eq('ADM1_NAME', 'Greater Accra'))

outline = ee.Image().paint(northern_district.merge(greateraccra_district), 0, 1)
outline2 = ee.Image().paint(ghana_district, 0.2, 0.5)

Map = geemap.Map(center=tamale_centre, zoom=8) 
Map.addLayer(outline2, {'palette': '999999'}, 'Ghana districts')
Map.addLayer(outline, {'palette': '000000'}, 'Northern & Accra districts')

#Map.add_control(output_control)
full_map = widgets.VBox([
  widgets.HBox([Map]),
  widgets.HBox([year_range, crop_mask, data_source], layout=widgets.Layout(height="40px")),
  widgets.HBox([region_choice, checkbox_mask, submit, clear], layout=widgets.Layout(height="40px"))
])

output_widget = widgets.Output(layout={'border': '1px solid black'})
output_control = WidgetControl(widget=output_widget, position='bottomleft')
Map.add_control(output_control)

for control in [year_range, crop_mask, data_source, region_choice, region_choice]:
    control.index = 0
checkbox_mask.value = True
year_range.disabled = False

    
base_no_layers = len(Map.layers)
base_no_controls = len(Map.controls)

Map.on_interaction(handle_click)
submit.on_click(submit_clicked)
clear.on_click(clear_clicked)
checkbox_mask.observe(checkmask_clicked)
data_source.observe(on_data_changed)
region_choice.observe(on_region_changed)


#display(full_map)


# # Interactive Map for inspecting time series

# In[33]:


maize_area = create_maize_area_map_northern(aoi=northern_district.merge(greateraccra_district))
maize_area_data = maize_area.reduceToImage(properties=['maize_area'], reducer=ee.Reducer.first()).rename('area')
vis = {
        'min': 6000,
        'max': 16000,
        'palette': ['FCFDBF', 'FDAE78', 'EE605E', 'B63679', '711F81', '2C105C']
        }


# In[39]:


Map_Northern = geemap.Map(center=farm_7021YAM, zoom=8)
Map_Northern.addLayer(outline, {}, 'Districts')

# Add an output widget to the map
output_widget1 = widgets.Output(layout={'border': '1px solid black'})
output_control1 = WidgetControl(widget=output_widget, position='bottomleft')
Map_Northern.add_control(output_control1)

Map_Northern.addLayer(maize_area_data, vis, 'Maize Planting Area', opacity=0.7)
Map_Northern.add_colorbar(vis, label='maize area, ha', layer_name='Maize Planting Area',
                        orientation="horizontal", transparent_bg=True)

dummy_file = open("Qtrend/Qtrend_ALL_NORTHERN_DISTRICTS.png", "rb")
image = dummy_file.read()
dummy1 = widgets.Image(
    value=image,
    format='png',
    width=300,
    height=400,
)
image_widget1 = WidgetControl(widget=dummy1, position='topright')
Map_Northern.add_control(image_widget1)

def refresh_trend_map():
    Map_Northern.clear_controls()
    Map_Northern.add_colorbar(vis, label='maize area, ha', layer_name='Maize Planting Area',
                        orientation="horizontal", transparent_bg=True)
    Map_Northern.add_control(output_control)
    Map_Northern.add_control(image_widget)

#orignal_contrs = Map_Northern.controls

# Capture user interaction with the map
def handle_interaction(**kwargs):
    latlon = kwargs.get('coordinates')
    if kwargs.get('type') == 'click':
        Map_Northern.default_style = {'cursor': 'wait'}
        
        with output_widget1:
            output_widget1.clear_output()
            while len(Map_Northern.controls) >= base1_no_controls:
                Map_Northern.remove_control(Map_Northern.controls[-1])
                
            print(f'{latlon[0]:0.5f}', f'{latlon[1]:0.5f}')
            poi = ee.Geometry.Point(latlon[::-1])
            #point = ee.Image().paint(poi, 0, 10)
            #Map_Northern.addLayer(point)
                        
            district = northern_district.merge(greateraccra_district).filterBounds(poi).aggregate_array('ADM2_NAME').getInfo()
            if len(district) > 0:
                district = district[0]
                print(district)
                
                #polygon = ee.Image().paint(**{'featureCollection': aoi_dist })
                #polygonIntersects = polygon.geometry().intersects(poi)
                #print(polygonIntersects.getInfo())
            
                short_name = ''.join(district.split(' ')).replace('/', '_')
                
                img_fn = f"Qtrend/Qtrend_{short_name}.png"
                if path.exists(img_fn):
                    file = open(img_fn, "rb")
                else:
                    file =  open("Qtrend/Qtrend_ALL_NORTHERN_DISTRICTS.png", "rb")
                image = file.read()
                img_widget = widgets.Image(
                    value=image,
                    format='png',
                    width=300,
                    height=400,
                )
                image_widget1 = WidgetControl(widget=img_widget, position='topright')
                Map_Northern.add_control(image_widget1)
                
                img_fn_qc = f"Qtrend/Qtrend_{short_name}_ModisHist_2015.png"
                if path.exists(img_fn_qc):
                    file = open(img_fn_qc, "rb")
                    image_qc = file.read()
                    qc_widget = widgets.Image(
                        value=image_qc,
                        format='png',
                        width=300,
                        height=400,
                    )
                    qc_widget = WidgetControl(widget=qc_widget, position='topright')
                    Map_Northern.add_control(qc_widget)
            else:
                district = cities.filterBounds(poi).aggregate_array('ADM2_NAME').getInfo()
                country = cities.filterBounds(poi).aggregate_array('ADM0_NAME').getInfo()
                print(f"{district[0]}, {country[0]}")
            
    Map_Northern.default_style = {'cursor': 'pointer'}
    
#base1_no_layers = len(Map_Northern.layers)
base1_no_controls = len(Map_Northern.controls)
#print(base1_no_controls)

Map_Northern.on_interaction(handle_interaction)
#Map_Northern


