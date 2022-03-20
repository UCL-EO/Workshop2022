import os
import sys
import numpy as np
import requests
import json
import datetime
import pandas as pd
from pygeotile.tile import Tile
from shapely import geometry
from bqplot import Lines, Figure, LinearScale, DateScale, Axis, Boxplot, Scatter
from ipywidgets import Dropdown, FloatSlider, HBox, VBox, Layout, Label, jslink, Layout, SelectionSlider, Play, Tab, Box, Button, HTML
from ipyleaflet import Map, WidgetControl, LayersControl, ImageOverlay, GeoJSON, Marker, Icon, ScaleControl, basemaps, DivIcon, MarkerCluster
from ipywidgets import Image as widgetIMG
from ipyevents import Event
from bqplot import ColorScale, FlexLine, ColorAxis
from bqplot import Label as bqLabel

sys.path.insert(0, './python/')
from map_utils import get_lai_gif, get_pixel, get_field_bounds, da_pix, get_lai_color_bar, get_wofost_yield, get_wofost_yield_unc
from map_utils import debounce, load_s2_bios, get_field_geo_transform_S2_30PYR, get_s2_bounds, latlon_2_xy, get_pixel_s2_bios
from wofost_utils import create_ensemble, wofost_parameter_sweep_func, get_era5_gee
from wofost_utils import ensemble_assimilation

from ipywidgets import Image as ImageWidget
import base64

df = pd.read_csv('data/Ghana_ground_data_v3.csv')

yield_df = pd.read_csv('data/Yield_Maize_Biomass_V2.csv').iloc[:, :3]
codes = np.array([i[:-2] for i in yield_df.FID]).reshape(-1, 3)[:, 0]
yields = np.array(yield_df.iloc[:, 1]).reshape(-1, 3)
field_yields = dict(zip(codes, yields.tolist()))

# import IPython
# url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png'
# maize_img_data = IPython.display.Image(url, width = 300)
# maize_img = ImageWidget(
#   value=maize_img_data.data,
#   format='Png', 
#   width='20',
#   height='20',
# )

zoom = 10

basemap = basemaps.OpenStreetMap.Mapnik
basemap['url'] = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
basemap['name'] = 'Google satellite'
basemap['html_attribution'] = 'Google satellite'
basemap['attribution'] = 'Google satellite'

defaultLayout=Layout(width='100%', height='760px')
my_map = Map(center=(9.8771, -0.6062), zoom=zoom, scroll_wheel_zoom=True, max_zoom = 19, layout=defaultLayout, basemap=basemap)



with open('./data/Biophysical_Data_Collection_Polygons_V1.geojson', 'r') as f:
    data = json.load(f)
features_with_yield = []
for feature in data['features']:
    if feature['properties']['Field_ID'] in codes:
        features_with_yield.append(feature)
data['features'] = features_with_yield


field_ids = [feat['properties']['Field_ID'] for feat in data['features']]

dropdown = Dropdown(
    options=sorted(codes.tolist()),
    value=codes[0],
    description="Field ID:",
)
field_id = field_ids[0]

field_bounds, doys = get_field_bounds(field_id)
# url, field_bounds, doys, yield_colorbar_f = get_lai_gif(field_id)

# image = yield_colorbar_f.getvalue()
# output = widgetIMG(value=image, format='png',)
# colorbar = WidgetControl(widget=output, position='bottomleft', transparent_bg=False)
# my_map.add_control(colorbar)

poly = geometry.Polygon(data['features'][0]['geometry']['coordinates'][0])
my_map.center = (poly.centroid.y, poly.centroid.x)

my_map.add_control(ScaleControl(position='bottomleft'))

x_scale = LinearScale(min = 200, max = 365)
y_scale = LinearScale(min = 0, max = 3)
x = np.arange(200, 365)
y = np.zeros_like(x)

lines = Lines(x=x, y=y, scales={"x": x_scale, "y": y_scale})

tick_style = {'font-size': 8}
ax_x = Axis(label="DOY", scale=x_scale, num_ticks=5, tick_style=tick_style)
ax_y = Axis(label='LAI', scale=y_scale, orientation="vertical", side="left", num_ticks=4, tick_style=tick_style)

figure = Figure(
    axes=[ax_x, ax_y],
    title=field_id,
    marks=[lines],
    animation_duration=500,
    layout={"max_height": "250px", "max_width": "400px"},
)

# widget_control1 = WidgetControl(widget=figure, position="bottomright")
# my_map.add_control(widget_control1)


line = Lines(x=x, y=y, scales={"x": x_scale, "y": y_scale})

fig_layout = Layout(width='auto', height='auto', max_height='120px', max_width='180px')
#fig_layout = Layout(width='1%', height='20%')

figy=[]
for i in range(3):
    figx=[]
    for j in range(2):
        fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line], 
                           title=field_id, 
                           animation_duration=500, 
                           title_style = {'font-size': '8'},
                           fig_margin = dict(top=16, bottom=16, left=16, right=16))
        fig.title = field_id
        figx.append(fig)
    figy.append(HBox(figx))
widget_control2 = WidgetControl(widget=VBox(figy, align_content = 'stretch'), position='topright')
# my_map.add_control(widget_control2)



fig_layout = Layout(width='auto', height='auto', max_height='120px', max_width='180px')
#fig_layout = Layout(width='1%', height='20%')

# doys = np.arange(200, 365)
sels = np.zeros((6, len(doys), 10))

tick_style = {'font-size': 8}
names = ['Blue', 'Green', 'Red', 'NIR', 'NDVI', 'Lai']
line_axs = []
for ii in range(6):
    y_scale = LinearScale(min = sels[ii].T.min(), max = sels[ii].T.max())
    ax_x = Axis(label="DOY", scale=x_scale, num_ticks=5, tick_style=tick_style)
    ax_y = Axis(label=names[ii], scale=y_scale, orientation="vertical", side="left", num_ticks=4, tick_style=tick_style)
    line = Lines(x=doys, y=sels[ii].T, scales={"x": x_scale, "y": y_scale})
    line.colors = ['#81d8d0']
    line.stroke_width = 0.1
    line_axs.append([line, ax_x, ax_y])

ref_lines = []
for ii in range(5):
    line, ax_x, ax_y = line_axs[ii]
    ref_line = Lines(x=doys, y=np.ones_like(doys) * np.nan, scales = line.scales, line_style='dotted', marker='circle', marker_size=4, colors = ['#c0c0c0'])
    ref_lines.append(ref_line)

good_ref_lines = []
line_colors = ['#3399ff', '#008000', '#ff6666', '#990000', '#20b2aa']
for ii in range(5):
    line, ax_x, ax_y = line_axs[ii]
    good_ref_line = Lines(x=doys, y=np.ones_like(doys) * np.nan, scales = line.scales, line_style='dotted', marker='circle', marker_size=4, colors = [line_colors[ii]])
    good_ref_lines.append(good_ref_line)

figy=[]
for i in range(3):
    figx=[]
    for j in range(2):
        if i*2+j < 5:
            ref_line = ref_lines[i*2+j]
            good_ref_line = good_ref_lines[i*2+j]
            line, ax_x, ax_y = line_axs[i*2+j]
            fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, ref_line, good_ref_line], 
                               title=field_id, 
                               animation_duration=500, 
                               title_style = {'font-size': '8'},
                               fig_margin = dict(top=16, bottom=16, left=16, right=16))

        else:
            line, ax_x, ax_y = line_axs[i*2+j]
            var_line = Lines(x=doys, y=np.zeros_like(doys), scales = line.scales)
            line_axs.append(var_line)
            fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, var_line], 
                               title=field_id, 
                               animation_duration=500, 
                               title_style = {'font-size': '8'},
                               legend_text={'font-size': 7},
                               legend_location = 'top-left',
                               legend_style={'width': '40%', 'height': '30%', 'stroke-width':0},
                               fig_margin = dict(top=16, bottom=16, left=26, right=16))

        fig.title = names[i*2+j]
        figx.append(fig)
    figy.append(HBox(figx))
# display(VBox(figy, align_content = 'stretch'))

fig_box = VBox(figy, align_content = 'stretch')
# widget_control1 = WidgetControl(widget=VBox(figy, align_content = 'stretch'), position='topright')


cab_fig = figy[2].children[0]
ndvi_fig = figy[2].children[0]
lai_fig = figy[2].children[1]

# field_cab_boxes = Boxplot(x=field_doys[:-1], y=field_cabs[:-1], 
#                           scales=cab_fig.marks[0].scales, box_fill_color='gray')
# field_cab_boxes.auto_detect_outliers=False
# # field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
# #                       scales=cab_fig.marks[0].scales, colors=['orange'])
# field_cab_boxes.stroke = 'red'
# field_cab_boxes.box_fill_color = 'blue'
# field_cab_boxes.opacities = [0.5]
# field_cab_boxes.box_width=5
# cab_fig.marks = cab_fig.marks[:2] + [field_cab_boxes,]


field_lai_boxes = Boxplot(x=doys, y=doys[None]*np.nan, 
                          scales=lai_fig.marks[1].scales, box_fill_color='gray')
field_lai_boxes.auto_detect_outliers=False
# field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
#                       scales=cab_fig.marks[0].scales, colors=['orange'])
field_lai_boxes.stroke = 'red'
field_lai_boxes.box_fill_color = 'blue'
field_lai_boxes.opacities = [0.4]
field_lai_boxes.box_width=5


lai_dot = Lines(x=doys[:1], y=[0,], scales=lai_fig.marks[1].scales,line_style='dotted', marker='circle', marker_size=45, colors = ['red'])

wofost_lai = Lines(x=doys, y=np.zeros_like(doys)*np.nan, scales = lai_fig.marks[1].scales)

field_med_lai_line = Lines(x=doys, y=np.zeros_like(doys)*np.nan, 
                           scales = lai_fig.marks[1].scales,  
                           colors = ['#fe217f'], display_legend=True, labels = ['Field LAI median'])


lai_fig_dvs_labels = bqLabel(x = [0, 0], 
                     y = [0, 0],  
                     text=["DVS=1", "DVS=2"], 
                     default_size=0,
                     scales=lai_fig.marks[1].scales,
                     colors = ['#31a354','#feb24c'])


lai_fig_dvs1_vline = Lines(x=[np.nan, np.nan], y=[0, 0], scales=lai_fig.marks[1].scales,
                   line_style='solid', colors=['#31a354'], stroke_width=1)

lai_fig_dvs2_vline = Lines(x=[np.nan, np.nan], y=[0, 0], scales=lai_fig.marks[1].scales,
                   line_style='solid', colors=['#feb24c'], stroke_width=1)


lai_fig.marks = lai_fig.marks[:2] + [field_lai_boxes, lai_dot, wofost_lai, field_med_lai_line, lai_fig_dvs1_vline, lai_fig_dvs2_vline, lai_fig_dvs_labels]
var_line = line_axs[-1]

var_line.labels = ['Pixel LAI']
var_line.display_legend = True




# box_layout = Layout(display='flex',
#                 flex_flow='column',
#                 align_items='center',
#                 width='100%')

# from ipywidgets import IntSlider

# def read_wofost_data(fname):
#     f = np.load(fname, allow_pickle=True)
#     parameters = f.f.parameters
#     t_axis = f.f.t_axis
#     samples = f.f.samples
#     lais = f.f.lais
#     yields = f.f.yields
#     DVS = f.f.DVS
#     print("loading simulations")
#     doys = [int(datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%j')) for i in t_axis]
#     return parameters, t_axis, samples, lais, yields, DVS, doys

# parameters, t_axis, samples, lais, yields, DVS, simu_doys = read_wofost_data('data/wofost_sims_dvs150.npz')

def read_wofost_data(lat, lon, year):
    """Reads ensemble from JASMIN"""
    # lat, lon = my_map.center
    # lat, lon = (lat // 0.1) * 0.1, (lon // 0.1) * 0.1
    print(lat, lon, year)
    f = create_ensemble(lat, lon, year, 10000)
    max_lai = np.nanmax(f.f.LAI, axis=1)
    y = f.f.Yields.astype(float)
    lai = f.f.LAI.astype(float)
    param_names = paras

    param_array = np.array([f[i]
                            for i in param_names]).astype(float)

    # pred_yield = max_lai * 1500 - 700
    # passer = np.abs(pred_yield - y) < 60000.
    # sim_yields = y[passer]
    # sim_lai = lai[passer, :]
    # param_array = param_array[:, passer]
    sim_yields = y
    sim_lai = lai
    
    
    sim_times = f.f.sim_times

    doys = [int(x.strftime("%j")) for x in sim_times]
    return param_array, sim_times, sim_lai, sim_yields, doys




# Button needs to be centered
assimilate_me_button = Button(
    description='Auto Fit',
    disabled=False,
    button_style='danger', 
    tooltip='Click me',
    icon='',
    layout = Layout(display='flex',
                    flex_flow='horizontal',
                    align_items='center',
                    justify_content="center",
                    width='30%'))

ensemble_at_location = {}
def assimilate_me(b):
    # Needs obs LAI & obs LAI times
    # global pix_lai, doys
    global ensemble_at_location
    doys = line_axs[-1].x
    pix_lai = line_axs[-1].y
    
#     step = 5
#     bin_doys = np.arange(doys.min(), doys.max(), step)
#     pix_lai = np.interp(bin_doys, doys, pix_lai)
#     doys = bin_doys
    
    bin_lais = []
    bin_doys = []
    step = 10
    for i in np.arange(doys.min(), doys.max()-step, step):
        mm = (doys >= i) * (doys <=i+step)
        bin_lai = np.nanmean(pix_lai[mm])
        if not np.isnan(bin_lai):
            bin_lais.append(bin_lai)
            bin_doys.append(int(i + step / 2))
    pix_lai = np.array(bin_lais)
    doys = np.array(bin_doys)
    print(pix_lai)
    lat, lon = my_map.center
    lat, lon = (lat // 0.1) * 0.1, (lon // 0.1) * 0.1
    year = 2021
    
    
    
    # Read ensemble
    wofost_status_info.description = 'Getting Wofost ensembles...'
    
    if '%.02f_%.02f'%(lat, lon) not in ensemble_at_location.keys():
        ensemble_at_location = {}
        param_array, sim_times, sim_lai, sim_yields, sim_doys = read_wofost_data(lat, lon, year)
        ensemble_at_location['%.02f_%.02f'%(lat, lon)] = param_array, sim_times, sim_lai, sim_yields, sim_doys
    else:
        wofost_status_info.description = 'Using cached ensembles'
        param_array, sim_times, sim_lai, sim_yields, sim_doys = ensemble_at_location['%.02f_%.02f'%(lat, lon)]
    
    print(ensemble_at_location.keys())
    
    t_axis = np.array([datetime.datetime.strptime(f"{year}/{x}", "%Y/%j").date()
              for x in doys])
    
    wofost_status_info.description = 'Fitting to Planet LAI'
    est_yield, est_yield_sd, parameters, _, _ , ensemble_lai_time, lai_fitted_ensembles = ensemble_assimilation(
        param_array, sim_times, sim_lai, sim_yields, pix_lai, t_axis, sel_n_best=5)
    
    for i, para in enumerate(paras):
        wofost_sliders_dict[para].value = np.mean(parameters[i])
    
    ensemble_lai_time = [int(i.strftime('%j')) for i in ensemble_lai_time]
    
    wofost_out_dict['LAI'].marks[6].x = ensemble_lai_time
    wofost_out_dict['LAI'].marks[6].y = lai_fitted_ensembles
    wofost_out_dict['LAI'].marks[6].display_legend = False
    print(est_yield)
    # est_yield: mean estimated yield for this LAI set of observations
    # est_yield_sd: standard deviation for the yield estimate
    # parameters: list with parameters of selected ensemble members
    # lai_fitted_ensembles: list with selected LAI simulations
    # TOOD:
    # 1. Update sliders with ?mean/median parameters?
    # 2. Probably run wofost with solution parameters (or just plot ensemble?)
    # 3. Update plots of LAI and TWSO
    
# Link assimilate button to function
assimilate_me_button.on_click(assimilate_me)    



# k_slider1 = IntSlider(min=181, max=224, value=200,        # Opacity is valid in [0,1] range
#                orientation='horizontal',       # Vertical slider is what we want
#                readout=True,                # No need to show exact value
#                layout=Layout(width='80%'),
#                description='Doy of sowing: ', 
#                style={'description_width': 'initial'}) 

# k_slider2 = FloatSlider(min=0.05, max=0.55, value=0.35,       # Opacity is valid in [0,1] range
#                step = 0.0025,
#                orientation='horizontal',       # Vertical slider is what we want
#                readout=True,                # No need to show exact value
#                layout=Layout(width='80%'),
#                description='Early stress level: ', 
#                style={'description_width': 'initial'}) 

# k_slider3 = FloatSlider(min=0.05, max=0.55,  value=0.35,       # Opacity is valid in [0,1] range
#                step = 0.0025,
#                orientation='horizontal',       # Vertical slider is what we want
#                readout=True,                # No need to show exact value
#                layout=Layout(width='80%'),
#                description='Late stress level: ', 
#                style={'description_width': 'initial'}) 

# def on_change_k_sliders(change):

#     if (change['name'] == 'value') & (change['type'] == 'change'):
#         value = change["new"]
#         old = change['old']
        
#         k1 = k_slider1.value
#         k2 = k_slider2.value
#         k3 = k_slider3.value
        
        
#         diff = abs(parameters - np.array([[k1, k2, k3]])).sum(axis=1)
#         ind = np.argmin(diff)
        
#         simu_lai = lais[ind]
        
        
#         # var_line = line_axs[-1]
#         # var_line.scales = line_axs[5][0].scales
#         # field_lai_boxes.scales = var_line.scales
#         # lai_dot.scales = var_line.scales

#         wofost_lai.x = simu_doys
#         wofost_lai.y = simu_lai
#         wofost_lai.scales = lai_fig.marks[1].scales
#         wofost_lai.colors = ['red']
#         print(k1, k2, k3)
#         print(parameters[ind])
        
        

# k_slider1.observe(on_change_k_sliders)
# k_slider2.observe(on_change_k_sliders)
# k_slider3.observe(on_change_k_sliders)



# paras = ['TDWI', 'TSUM1', 'TSUM2', 'RGRLAI', 'SDOY', 'SPAN', 'AMAXTB_150']

# paras, para_mins, para_maxs = np.array(df.loc[:, ['#PARAM_CODE', 'Min', 'Max']]).T
# wofost_sliders = []

# for i in range(round(len(paras) / 2)):
#     horizon_sliders = []
#     for j in range(2):
#         if i*2 + j <  len(paras):
#             para_name = paras[i*2+j]
#             para_min = para_mins[i*2+j]
#             para_max = para_maxs[i*2+j]
#             step = (para_max - para_min) / 10
#             initial = (para_min + para_max) / 2
#             wofost_slider = FloatSlider(min=para_min, max=para_max, value=initial,       # Opacity is valid in [0,1] range
#                            step = step,
#                            orientation='horizontal',       # Vertical slider is what we want
#                            readout=True,                # No need to show exact value
#                            layout=Layout(width='80%'),
#                            description='%s: '%para_name, 
#                            style={'description_width': 'initial'}) 
#             horizon_sliders.append(wofost_slider)
#     wofost_sliders.append(HBox(horizon_sliders))

# wofost_sliders.append(lai_fig)
    
# wofost_box = VBox(wofost_sliders, layout = box_layout)

lon = -2.7
lat = 8.20
year = 2021


@debounce(0.2)
def on_change_wofost_slider(change):
    
    global wofost_out_dict
    if (change['type'] == 'change'):
        
        lat, lon = my_map.center
        lat, lon = (lat // 0.1) * 0.1, (lon // 0.1) * 0.1
        print(lat, lon, year)
        wofost_status_info.description = 'Reading ERA5 weather data from local or GEE'
        meteo_file = get_era5_gee(year, lat, lon, dest_folder="data/ERA5_weather/")
        print(meteo_file)
        
        ens_parameters = {}
        paras_to_overwrite = [i for i in paras if 'AMAX_' not in i]
        for para in paras_to_overwrite:    
            ens_parameters[para] = wofost_sliders_dict[para].value
        # ens_parameters['AMAXTB'] = [0, 55.0, 1.5, wofost_sliders_dict['AMAXTB_150'].value]
        scalar =  wofost_sliders_dict["AMAX_SCALAR"].value
        ens_parameters["AMAXTB"]  = [0.0, 70.0*scalar,
                                    1.25, 70.0*scalar,
                                    1.50, 63.0*scalar,
                                    1.75, 49.0*scalar,
                                    2.0, 0.0,
                                    ]
        
        #ens_parameters['AMAXTB'] = [0, wofost_sliders_dict['AMAXTB_000'].value,
        #                           1.25, wofost_sliders_dict['AMAXTB_125'].value,
        #                           1.50, wofost_sliders_dict['AMAXTB_150'].value,
        #                           2.0, 2
        #                           ]
        wofost_status_info.description = 'Running model...'
        df = wofost_parameter_sweep_func(year, ens_parameters = ens_parameters.copy(), meteo=meteo_file)
#         print(df)
        
        dates = df.index
        doys = [int(i.strftime('%j')) for i in dates]
        
        # wofost_lai_fig.x = doys
        # wofost_lai_fig.y = np.array(df.LAI)
        wofost_out_paras = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']
        for wofost_out_para in wofost_out_paras:
            if wofost_out_para != 'TWSO':
                wofost_out_dict[wofost_out_para].marks[0].x = doys
                wofost_out_dict[wofost_out_para].marks[0].y = np.array(df.loc[:, wofost_out_para])
            else:
                twso = np.array(df.loc[:, wofost_out_para])
                real_twso = 1.4553 * np.nanmax(twso) - 1341.81
                real_twso = np.max([0, real_twso])
                twso = real_twso / np.nanmax(twso) * twso
                wofost_out_dict[wofost_out_para].marks[1].x = doys
                wofost_out_dict[wofost_out_para].marks[1].y = np.array(twso)
        wofost_out_dict['LAI'].marks[5].x = line_axs[-1].x
        wofost_out_dict['LAI'].marks[5].y = line_axs[-1].y
        colored_dvs_line.x = doys
        colored_dvs_line.y = wofost_out_dict['DVS'].marks[0].y
        colored_dvs_line.color = wofost_out_dict['DVS'].marks[0].y
        wofost_status_info.description = 'Done'
        
        dvs = np.array(df.loc[:, 'DVS'])
        dvs2_ind = np.nanargmax(dvs)
        dvs1_ind = np.nanargmin(abs(dvs-1))
        print(dvs1_ind)  
        print(dvs2_ind)
        
        

        
        for wofost_out_para in wofost_out_paras:
            wofost_fig_dsv1_vlines[wofost_out_para].x = [doys[dvs1_ind], doys[dvs1_ind]]
            wofost_fig_dsv2_vlines[wofost_out_para].x = [doys[dvs2_ind], doys[dvs2_ind]]
        
        for wofost_out_para in wofost_out_paras:
            scales =  wofost_out_dict[wofost_out_para].marks[1].scales
            ymin = scales['y'].min
            ymax = scales['y'].max
            y = ymin + (ymax - ymin) * 0.1
            dvs_labels_dict[wofost_out_para].scales = scales
            
            dvs_labels_dict[wofost_out_para].x = [doys[dvs1_ind], doys[dvs2_ind]]
            dvs_labels_dict[wofost_out_para].y = [y, y]
            dvs_labels_dict[wofost_out_para].default_size= 8
            
        
        scales = line_axs[-1].scales
        lai_fig_dvs_labels.scales = scales
        lai_fig_dvs1_vline.scales = scales
        lai_fig_dvs2_vline.scales = scales
        
        lai_fig_dvs1_vline.y = [0, 5]
        lai_fig_dvs2_vline.y = [0, 5]
        lai_fig_dvs1_vline.x = [doys[dvs1_ind], doys[dvs1_ind]]
        lai_fig_dvs2_vline.x = [doys[dvs2_ind], doys[dvs2_ind]]
        
        
        ymin = scales['y'].min
        ymax = scales['y'].max
        y = ymin + (ymax - ymin) * 0.1
        lai_fig_dvs_labels.x = [doys[dvs1_ind], doys[dvs2_ind]]
        lai_fig_dvs_labels.y = [y, y]
        lai_fig_dvs_labels.default_size= 8
        
        # wofost_out_dict['TWSO'].marks[0].x = doys
        # wofost_out_dict['TWSO'].marks[0].y = np.array(df.TWSO)
        
        # lai_fig.marks[1].scales = {'x': LinearScale(max=365.0, min=180.0), 'y': LinearScale(max=3.0, min=0.0)} 
        # lai_fig.axes[1].scale = LinearScale(max=3.0, min=0.0)
        # lai_fig.axes[0].scale = LinearScale(max=365.0, min=180.0)

prior_df = pd.read_csv('data/par_prior_maize_tropical-C.csv')
#paras = ['TDWI', 'SDOY', 'SPAN', 'CVO', 'AMAXTB_000', 'AMAXTB_125', 'AMAXTB_150']
paras = ['TDWI', 'SDOY', 'SPAN',  'AMAX_SCALAR']
all_paras, para_mins, para_maxs = np.array(prior_df.loc[:, ['#PARAM_CODE', 'Min', 'Max']]).T
para_inds = [all_paras.tolist().index(i) for i in paras]
wofost_sliders = []

para_meaning = {'TDWI': 'TDWI: Initial total crop dry weight [kg ha-1]',
                'SDOY': 'SDOY: Sowing day of year [days]',
                'SPAN': 'SPAN: Life span of leaves growing at 35 Celsius [d]',
                #'CVO' : 'Efficiency of conversion into storage org. [kg kg-1]',
                #'AMAXTB_000': 'Max. leaf CO2 assim. rate at development stage of 0',
                #'AMAXTB_125': 'Max. leaf CO2 assim. rate at development stage of 1.25',
                #'AMAXTB_150': 'Max. leaf CO2 assim. rate at development stage of 1.5',
                "AMAX_SCALAR": "AMAX_SCALAR: Scalar on Max. leaf CO2 assim. rate"
               }


para_simple_name = {'TDWI': 'Seed mass',
                    'SDOY': 'Sowing date',
                    'SPAN': 'Leaf life span',
                    'AMAX_SCALAR': 'Assimilation rate'
                   }

para_labels = []
for i in range(len(paras)):
    para_ind = para_inds[i]
    para_name = all_paras[para_ind]
    para_min = para_mins[para_ind]
    para_max = para_maxs[para_ind]
    step = (para_max - para_min) / 50
    initial = (para_min + para_max) / 2
    
    para_label = Label('%s: '%para_simple_name[para_name])
    
    wofost_slider = FloatSlider(min=para_min, max=para_max, value=initial,       # Opacity is valid in [0,1] range
                   step = step,
                   orientation='horizontal',       # Vertical slider is what we want
                   readout=True,                # No need to show exact value
                   layout=Layout(width='280px'),
                   # description='%s: '%para_simple_name[para_name], 
                   description_tooltip= para_meaning[para_name],
                   style={'description_width': 'initial'}) 
    wofost_sliders.append(wofost_slider)
    para_labels.append(para_label)
    
    wofost_slider.observe(on_change_wofost_slider)
    
    
wofost_labels_sliders = [HBox([VBox(para_labels), VBox(wofost_sliders)])]
    
wofost_sliders_dict = dict(zip(paras, wofost_sliders))


def create_tooltip_inputs(para_name):
    para_event = Event(source=wofost_sliders_dict[para_name], watched_events=['mouseenter', 'mouseleave'])
    def figure_tooltip_event_handler(event):
        old_decription = wofost_status_info.description
        if event['event'] == 'mouseenter':
            wofost_status_info.description = para_meaning[para_name]
        if event['event'] == 'mouseleave':
            wofost_status_info.description = ''
    para_event.on_dom_event(figure_tooltip_event_handler)

for para_name in paras:
    create_tooltip_inputs(para_name)
    

wofost_fig_vlines = {}
wofost_fig_dsv1_vlines = {}
wofost_fig_dsv2_vlines = {}
dvs_labels_dict = {}

def get_para_plot(para_name, x, y, xmin = 180, xmax = 330):
    global wofost_fig_vlines
    x = np.array(x)
    y = np.array(y)
    
    para_min_maxs = {'DVS': [0, 2],
                     'LAI': [0, 3],
                     'TAGP': [0, 15000],
                     'TWSO': [0, 5500],
                     'TWLV': [0, 2000],
                     'TWST': [0, 10000],
                     'TWRT': [0, 2000],
                     'TRA':  [0, 0.5],
                     'RD':   [0, 100],
                     'SM':   [0, 0.8],
                     'WWLOW': [0, 100]
                    }

    ymin, ymax = para_min_maxs[para_name]
    
    mm = (x >= xmin) & (x <= xmax)
    x_scale = LinearScale(min = xmin, max = xmax)
    # ymin = np.maximum(np.nanpercentile(y[mm], 2.5) * 0.9, 0)
    # ymax = np.nanpercentile(y[mm], 97.5) * 1.1
    y_scale = LinearScale(min = ymin  , max = ymax)
    
    line = Lines(x=x, y=y, scales={"x": x_scale, "y": y_scale}, labels = 'Wofost %s'%para_name, display_legend=True)
    tick_style = {'font-size': 8}
    tick_values = np.linspace(ymin, ymax, 4)
    tick_values
    
    vline = Lines(x=[xmin, xmin], y=[0, ymax], scales={"x": x_scale, "y": y_scale},
                       line_style='solid', colors=['gray'], stroke_width=1)
    
    dvs1_vline = Lines(x=[xmin*np.nan, xmin*np.nan], y=[0, ymax], scales={"x": x_scale, "y": y_scale},
                       line_style='solid', colors=['#31a354'], stroke_width=1)
    
    dvs2_vline = Lines(x=[xmin*np.nan, xmin*np.nan], y=[0, ymax], scales={"x": x_scale, "y": y_scale},
                       line_style='solid', colors=['#feb24c'], stroke_width=1)
    
    

    dvs_labels = bqLabel(x = [0, 0], 
                         y = [0, 0],  
                         text=["DVS=1", "DVS=2"], 
                         default_size=0,
                         scales={"x": x_scale, "y": y_scale},
                         colors = ['#31a354','#feb24c'])
    dvs_labels_dict[wofost_out_para] = dvs_labels
    
    wofost_fig_vlines[wofost_out_para] = vline
    wofost_fig_dsv1_vlines[wofost_out_para] = dvs1_vline
    wofost_fig_dsv2_vlines[wofost_out_para] = dvs2_vline
    
    ax_x = Axis(label="DOY", scale=x_scale,  num_ticks=5, tick_style=tick_style)
    ax_y = Axis(label=wofost_out_para_simple_name_dict[para_name], scale=y_scale, orientation="vertical", side="left", tick_values=tick_values, tick_style=tick_style)
    
    fig_layout = Layout(width='400px', height='160px', max_height='160px', max_width='400px')
    
    para_fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, vline, dvs1_vline, dvs2_vline, dvs_labels], 
                       title=para_name, 
                       animation_duration=500, 
                       title_style = {'font-size': '8'},
                       legend_text={'font-size': 7},
                       legend_location = 'top-left',
                       legend_style={'width': '40%', 'height': '30%', 'stroke-width':0},
                       fig_margin = dict(top=16, bottom=16, left=46, right=46))

    return para_fig


def update_wofost_fig_val(wofost_out_para, x, y, xmin = 180, xmax = 330):
    x = np.array(x)
    y = np.array(y)
    mm = (x >= xmin) & (x <= xmax)
    y = y[mm]
    x = x[mm]
    wofost_out_dict[wofost_out_para].marks[0].x = x
    wofost_out_dict[wofost_out_para].marks[0].y = y
    
wofost_out_paras = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']

wofost_out_para_simple_name = ['Development stage',
                               'LAI',
                               'Above ground biomass',
                               'Yield',
                               'Leaves weight',
                               'Stems weight',
                               'Roots weight',
                               'Transpiration',
                               'Root depth',
                               'Soil moisture',
                               'Available Water'
                              ]

wofost_out_para_simple_name_dict = dict(zip(wofost_out_paras, wofost_out_para_simple_name))

wofost_out_para_meaning = ['DVS: Wofost Development stage', 
                           'LAI: Leaf area index of the crop (m2/m2)', 
                           'TAGP: Total dry above-ground biomass (dry weight kg/ha)',
                           'TWSO: Total dry weight of storage organs (the yield) (kg/ha)',
                           'TWLV: Total dry weight of leaves (kg/ha)',
                           'TWST: Total dry weight of stems (kg/ha)',
                           'TWRT: Total dry weight of roots (kg/ha)',
                           'TRA: Crop transpiration (excluding soil evaporation) (cm/day)',
                           'RD: Crop rooting depth (cm)',
                           'SM: Root zone soil moisture as a volumetric fraction',
                           'WWLOW: Amount of available water in the rooted and unrooted zone (cm)'
                          ]

wofost_out_para_meaning = dict(zip(wofost_out_paras, wofost_out_para_meaning))



para_figs = []
x = np.arange(180, 330)
y = np.zeros_like(x)
for wofost_out_para in wofost_out_paras:
    para_fig = get_para_plot(wofost_out_para, x, y)
    para_figs.append(para_fig)
wofost_out_dict = dict(zip(wofost_out_paras, para_figs))



def create_tooltip(wofost_out_para):
    para_event = Event(source=wofost_out_dict[wofost_out_para], watched_events=['mouseenter', 'mouseleave'])
    def figure_tooltip_event_handler(event):
        old_decription = wofost_status_info.description
        if event['event'] == 'mouseenter':
            wofost_status_info.description = wofost_out_para_meaning[wofost_out_para]
        if event['event'] == 'mouseleave':
            wofost_status_info.description = ''
    para_event.on_dom_event(figure_tooltip_event_handler)



for wofost_out_para in wofost_out_paras:
    create_tooltip(wofost_out_para)
    
# wofost_fig_vlines = {}
# for wofost_out_para in wofost_out_paras:
#     vline = Lines(x=[180, 180], y=[0, wofost_out_dict[wofost_out_para].marks[0].scales['x'].max], scales=wofost_out_dict[wofost_out_para].marks[0], 
#                        line_style='solid', colors=['gray'], stroke_width=1)
#     wofost_fig_vlines[wofost_out_para] = vline
    
# obs_lai_line = Lines(x=line_axs[-1].x, y=line_axs[-1].y, scales=wofost_out_dict['LAI'].marks[0].scales, colors = ['red'])
obs_lai_line  = Scatter(x=line_axs[-1].x, y=line_axs[-1].y, scales=wofost_out_dict['LAI'].marks[0].scales, 
                        default_size=4, colors = ['green'], display_legend=True, labels=['Planet LAI'])    
ens_lai_line = Lines(x=line_axs[-1].x, y=line_axs[-1].y, scales=wofost_out_dict['LAI'].marks[0].scales, 
                     colors = ['#cccccc'], display_legend=False, labels=['Ensemble LAI'], opacities = [0.6,])

ens_lai_line_temp = Lines(x=line_axs[-1].x, y=line_axs[-1].y*np.nan, scales=wofost_out_dict['LAI'].marks[0].scales, 
                     colors = ['#cccccc'], display_legend=True, labels=['Ensemble LAI'])


wofost_out_dict['LAI'].marks[0].display_legend=True

lai_dot_wofost = Lines(x=[180,], y=[0,], scales=wofost_out_dict['LAI'].marks[0].scales,line_style='dotted', marker='circle', marker_size=45, colors = ['red'])
lai_vline = Lines(x=[180, 180], y=[0, 3], scales=wofost_out_dict['LAI'].marks[0].scales, 
                   line_style='solid', colors=['gray'], stroke_width=1)


twso_vline = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='dashed', colors=['gray'], fill='between')

twso_hline = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='solid', colors=['#ff0000'], fill='between', display_legend=True, labels = ['Avg. Field Yield'])

twso_shade = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='solid', colors=['#cccccc'], fill='between', opacities=[1, 1], display_legend=False, labels = ['1σ'])

twso_shade_temp = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='solid', colors=['#cccccc'], fill='between', opacities=[1, 1], display_legend=True, labels = ['1σ'])

dvs_line = wofost_out_dict['DVS'].marks[0]

colors = ['#e5f5e0', '#a1d99b', '#31a354', '#31a354', '#31a354','#a1d99b', '#ffeda0', '#feb24c']
col_line = ColorScale(colors=colors)

scales = dvs_line.scales
scales['color'] = col_line

dvs_yticks = wofost_out_dict['DVS'].axes[1].tick_values
dvs_yax = Axis(
    label=wofost_out_para_simple_name_dict["DVS"],
    scale=scales['y'],
    orientation="vertical",
    side="right",
    tick_format="0.1f",
    grid_lines = 'none',
    tick_values = dvs_yticks,
    tick_style = {'font-size': 8}
)

colored_dvs_line = FlexLine(x=dvs_line.x, y=dvs_line.y, color=dvs_line.y,
                             scales=scales, labels = ['Wofost DVS'], display_legend=False,)

wofost_out_dict['TWSO'].marks = [twso_shade] +  wofost_out_dict['TWSO'].marks[:5] + [twso_hline, colored_dvs_line,twso_shade_temp]
wofost_out_dict['TWSO'].legend_location = 'bottom-left'
wofost_out_dict['TWSO'].marks[1].display_legend=True

for wofost_out_para in wofost_out_paras:
    wofost_out_dict[wofost_out_para].axes = wofost_out_dict[wofost_out_para].axes + [dvs_yax,]

for wofost_out_para in ['TAGP', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']:
    wofost_out_dict[wofost_out_para].marks = wofost_out_dict[wofost_out_para].marks + [colored_dvs_line,]

    
wofost_out_dict['LAI'].marks = wofost_out_dict['LAI'].marks[:5] + [obs_lai_line, ens_lai_line, lai_dot_wofost, colored_dvs_line, ens_lai_line_temp]
    
    
wofost_output_dropdown1 = Dropdown(
    options=wofost_out_paras,
    value=wofost_out_paras[1],
    layout={'width': 'max-content'}
    
    # description="Field ID:",
)
wofost_output_dropdown2 = Dropdown(
    options=wofost_out_paras,
    value=wofost_out_paras[3],
    layout={'width': 'max-content'}
    # description="Field ID:",
)



wofost_output_dropdowns = HBox([wofost_output_dropdown1, wofost_output_dropdown2], layout = Layout(display='flex',
                                  flex_flow='horizontal',
                                  align_items='flex-start',
                                  width='100%'))


left_output = VBox([wofost_output_dropdown1, wofost_out_dict[wofost_output_dropdown1.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))

right_output = VBox([wofost_output_dropdown2, wofost_out_dict[wofost_output_dropdown2.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))



def on_change_dropdown1(change):
    global wofost_widgets
    
    left_output = VBox([wofost_output_dropdown1, wofost_out_dict[wofost_output_dropdown1.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    right_output = VBox([wofost_output_dropdown2, wofost_out_dict[wofost_output_dropdown2.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    
    wofost_widgets[-1] = VBox([left_output, right_output])
    wofost_box = VBox(wofost_widgets, 
                  layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))
    tab.children = [panel_box, wofost_box]

def on_change_dropdown2(change):
    global wofost_widgets
    
    left_output = VBox([wofost_output_dropdown1, wofost_out_dict[wofost_output_dropdown1.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    right_output = VBox([wofost_output_dropdown2, wofost_out_dict[wofost_output_dropdown2.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    wofost_widgets[-1] = VBox([left_output, right_output])

    wofost_box = VBox(wofost_widgets, 
                  layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))
    tab.children = [panel_box, wofost_box]

wofost_status_info = Button(description = '', button_style='info', layout=Layout(width='100%'), disabled=True)
wofost_status_info.style.button_color = '#999999'
wofost_out_panel = VBox([left_output, right_output])
wofost_widgets = [wofost_status_info, ] + wofost_labels_sliders + [assimilate_me_button, wofost_out_panel]
wofost_output_dropdown1.observe(on_change_dropdown1, 'value')
wofost_output_dropdown2.observe(on_change_dropdown2, 'value')



s2_bio_names = ['n', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown']
s2_bios_min_max = [[1, 2.5], [10, 80], [0, 0.02], [0, 0.04], [0, 3], [30, 80], [0, 1]]
s2_bios_min_max_dict = dict(zip(s2_bio_names, s2_bios_min_max))

s2_bio_simple_name = ['Leaf layers [-]', 'Chlorophyll a+b [ug/cm2]', 'Leaf dry matter [g/cm2]', 
                      'Leaf water content [g/cm2]', 'Leaf area index [m2/m2]', 'Leaf angle distribution [d]', 'Leaf senescence [-]']
s2_bio_simple_name_dict = dict(zip(s2_bio_names, s2_bio_simple_name))

s2_bio_colors = ['#407294', '#008080', '#b86a4b', '#1874cd', '#9acd32', '#800080', '#ffa500']
colors_dict = dict(zip(s2_bio_names, s2_bio_colors))


s2_bios_to_plot = ['n', 'cab', 'cm', 'cw', 'lai',  'cbrown']

fig_layout = Layout(width='auto', height='auto', max_height='150px', max_width='200px')
tick_style = {'font-size': 8, }
x_scale = LinearScale(min = 200, max = 365)

s2_bio_plot_dict = {}
s2_bio_plot_line_dict = {}
s2_bio_plot_fied_avg_line_dict = {}

    
s2_bio_vline = Lines(x=[0, 1], y=[0, 1], scales={"x": x_scale, "y": LinearScale(min = 0, max = 1)},
                   line_style='solid', colors=['gray'], stroke_width=1)


for s2_bio_name in s2_bios_to_plot:
    y_scale = LinearScale(min = s2_bios_min_max_dict[s2_bio_name][0], max = s2_bios_min_max_dict[s2_bio_name][1])
    
    y_ticks = np.linspace(y_scale.min, y_scale.max, 5)
    ax_x = Axis(label="DOY", scale=x_scale, num_ticks=5, tick_style=tick_style)
    
    ax_y = Axis(label=s2_bio_simple_name_dict[s2_bio_name], scale=y_scale, orientation="vertical", 
                tick_format='0.2f', side="left", tick_values=y_ticks, tick_style=tick_style)
    
    ax_y.min = s2_bios_min_max_dict[s2_bio_name][0]
    ax_y.max = s2_bios_min_max_dict[s2_bio_name][1]
    
    line = Lines(x=np.arange(200, 365), y=np.arange(200, 365)*np.nan, scales={"x": x_scale, "y": y_scale})
    line.colors = [colors_dict[s2_bio_name]]
    
    field_avg_box = Boxplot(x=np.arange(200, 365), y=np.arange(200, 365)[None]*np.nan, 
                          scales=line.scales, box_fill_color='gray')
    field_avg_box.auto_detect_outliers=False
    field_avg_box.stroke = 'red'
    field_avg_box.box_fill_color = colors_dict[s2_bio_name]
    field_avg_box.opacities = [0.4]
    field_avg_box.box_width=5
    
    bio_dot = Lines(x=[np.nan,], y=[np.nan,], scales=lai_fig.marks[1].scales,line_style='dotted', marker='circle', marker_size=45, colors = ['red'])

    line.stroke_width = 3
    fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, field_avg_box, s2_bio_vline], 
                       title=s2_bio_simple_name_dict[s2_bio_name], 
                       animation_duration=500, 
                       title_style = {'font-size': '8'},
                       fig_margin = dict(top=16, bottom=17, left=26, right=16))
    
    s2_bio_plot_line_dict[s2_bio_name] = line
    s2_bio_plot_dict[s2_bio_name] = fig
    s2_bio_plot_fied_avg_line_dict[s2_bio_name] = field_avg_box

left_box  = VBox([s2_bio_plot_dict['lai'], s2_bio_plot_dict['cm'], s2_bio_plot_dict['n']])
right_box = VBox([s2_bio_plot_dict['cab'], s2_bio_plot_dict['cw'], s2_bio_plot_dict['cbrown']])
s2_bio_box = HBox([left_box, right_box])


wofost_box = VBox(wofost_widgets, 
                  layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))


k_slider = FloatSlider(min=0, max=6, value=2,        # Opacity is valid in [0,1] range
               orientation='horizontal',       # Vertical slider is what we want
               readout=True,                # No need to show exact value
               layout=Layout(width='80%'),
               description='Outliers cutoff: ', 
               style={'description_width': 'initial'}) 

panel_box = VBox([fig_box, k_slider],
                 layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))



names = ['Planet Ref. fitting', 'Wofost simulation', 'Sentinel 2 bios']
tab = Tab()
tab.children = [panel_box, wofost_box, s2_bio_box]
[tab.set_title(i, title) for i, title in enumerate(names)]


widget_control1 = WidgetControl(widget=tab, position='topright')
my_map.add_control(widget_control1)



slider = FloatSlider(min=0, max=1, value=1,        # Opacity is valid in [0,1] range
               orientation='horizontal',       # Vertical slider is what we want
               readout=False,                # No need to show exact value
               layout=Layout(height='2em', width='200px')) # Fine tune display layout: make it thinner


tile = Tile.for_latitude_longitude(*my_map.center, zoom)
x, y = tile.tms

# for i in range(x-3, x + 4):
#     for j in range(y - 3, y + 4):
#         tile = Tile.from_tms(i, j, zoom)
#         url = "http://ecn.t3.tiles.virtualearth.net/tiles/a%s.png?g=1"%tile.quad_tree
#         ul, br = tile.bounds
#         image = ImageOverlay(
#             url=url,
#             bounds = tile.bounds,
#             name = '' #'bing_basemap_%d'%zoom
#         )
#         my_map.add_layer(image)  


def on_change_zoom(change):
    if change['type'] == 'change' and change['name'] == 'zoom':
        zoom = int(my_map.zoom)
        tile = Tile.for_latitude_longitude(*my_map.center, zoom)
        x, y = tile.tms


        for i in range(x- 3, x + 4):
            for j in range(y - 2, y + 3):
                tile = Tile.from_tms(i, j, zoom)
                url = "http://ecn.t3.tiles.virtualearth.net/tiles/a%s.png?g=1"%tile.quad_tree

                image = ImageOverlay(
                    url=url,
                    bounds = tile.bounds,
                    #name = 'bing_basemap_%d'%zoom
                )
                my_map.add_layer(image)   

        for layer in my_map.layers:
            if layer.name == 'bing_basemap_%d'%int(change['old']):
                my_map.remove_layer(layer)

# my_map.observe(on_change_zoom)


def create_field_image_tab(urls, dates):
    field_image_widgets = []
    for i, url in enumerate(urls):
        r = requests.get(url)
        if r.ok:
            field_image = widgetIMG(
              value=r.content,
              format='png',

              # width = 450,
            )
            field_image.layout.object_fit = 'cover'
            field_image.layout.height = '310px'

            date_str = dates[i]
            date = datetime.datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            date_str = date.strftime('%Y-%m-%dT%H:%M:%S  (DOY: %j)')

            date = Label(date_str)

            field_box_layout = Layout(min_width = '250px', overflow='hidden', align_items='center', border='1px solid white',)
            field_box = VBox([date, field_image], layout = field_box_layout)
            field_image_widgets.append(field_box)

    box_layout = Layout(overflow='scroll hidden',
                        border='0px solid black',
                        width='400px',
                        height='350px',
                        # align_items='flex-end',
                        flex_flow='row',
                        display='flex')
    carousel = Box(children=field_image_widgets, layout=box_layout)
    # tab = Tab(field_image_widgets)
    # [tab.set_title(i, '%02d'%(i+1)) for i in range(len(field_image_widgets))]
    return carousel



with open('./data/Ghana_field_photos.json', 'r') as f:
    Ghana_field_photo_dict = json.load(f)
    
field_movie = None
yield_img = None
yield_control = None
daily_img = None
maize_markers = []
wofost_yield_img = None
wofost_yield_unc_img = None
lai_colorbar_f = get_lai_color_bar()
image = lai_colorbar_f.getvalue()
lai_colorbar_output = widgetIMG(value=image, format='png',)
# lai_colorbar_output.layout.object_fit = 'contain'
lai_colorbar_label = Label('$$LAI [m^2/m^2]$$')
# lai_box = VBox([lai_label, output], align_content = 'stretch', layout=Layout(width='100%', height='50%'))
loading_bar_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/loading.gif'
loading_bar = requests.get(loading_bar_url)

colorbar_dropdown = Dropdown(
    options=['Wofost yield colorbar', 'Wofost yield unc. colorbar', 'Empirical yield colorbar', 'LAI colorbar'],
    value = 'LAI colorbar',
    layout={'width': 'max-content'}
)


colorbar_box = VBox([lai_colorbar_label, lai_colorbar_output], 
                    layout=Layout(display='flex',flex_flow='column',align_items='center')
                   )
colorbar_dropdown_box = VBox([colorbar_dropdown, colorbar_box], 
                    layout=Layout(display='flex',flex_flow='column',align_items='center')
                   )


def on_change_colorbar_dropdown(change):
    global colorbar_box
    colorbar_box.children = colorbar_box_dict[colorbar_dropdown.value]
    
colorbar_dropdown.observe(on_change_colorbar_dropdown, 'value')

colorbar_box_dict = {'LAI colorbar': [lai_colorbar_label, lai_colorbar_output]}

bing_images = []
def on_click(change):
    global field_id
    global field_bounds
    global doys
    global colorbar_box_dict
    global field_movie, field_lais, field_doys, field_cabs, yield_control, daily_img, maize_markers
    global wofost_out_dict, colorbar_box, bing_images
    global s2_projectionRef, s2_geo_trans, s2_bounds, s2_bios, s2_bio_doys
    field_id = change["new"]


    ind = field_ids.index(field_id)

    feature = data['features'][ind]
    poly = geometry.Polygon(feature['geometry']['coordinates'][0])
    lon, lat = poly.centroid.coords[0]
    my_map.center = lat, lon
    zoom = 17
    my_map.zoom = zoom
    tile = Tile.for_latitude_longitude(*my_map.center, zoom)
    x, y = tile.tms
    
    if len(bing_images)!= 0:
        for image in bing_images:
            my_map.remove_layer(image)
        bing_images = []
    for i in range(x- 3, x + 4):
        for j in range(y - 2, y + 3):
            tile = Tile.from_tms(i, j, zoom)
            url = "http://ecn.t3.tiles.virtualearth.net/tiles/a%s.png?g=1"%tile.quad_tree
            image = ImageOverlay(
                url=url,
                bounds = tile.bounds,
                #name = 'bing_basemap_%d'%zoom
            )
            my_map.add_layer(image)   
            bing_images.append(image)
    
    print(field_id)
    play_label.value = 'Click to play LAI movie over field: %s'%field_id
    home = os.getcwd()
    cwd = '/files/' + '/'.join(home.split('/')[3:])
    base_url = my_map.window_url.split('/lab/')[0] + cwd + '/'
    
    i_lab, i_tree = my_map.window_url.find('/lab/'),  my_map.window_url.find('/tree/')
    to_replace = my_map.window_url[i_lab:i_tree+6]
    base_url = '/'.join(my_map.window_url.replace(to_replace, '/files/').split('/')[:-1]) + '/'
    base_url

    if yield_control is not None:
        
        my_map.remove_control(yield_control)
        
    loading_bar_img = widgetIMG(value=loading_bar.content,
      format='gif', 
      width=30,
      height=40,
    )
    loading_label = Label('Loading data over field %s'%field_id)
    loading_info = HBox([loading_bar_img, loading_label])
    yield_control = WidgetControl(widget=loading_info, position='bottomleft')
    
    my_map.add_control(yield_control)

    
    s2_bios, s2_bio_doys = load_s2_bios(field_id)
    s2_projectionRef, s2_geo_trans = get_field_geo_transform_S2_30PYR(field_id)
    s2_bounds = get_s2_bounds(s2_projectionRef, s2_geo_trans, s2_bios.shape[2:])
    

    url, bounds, doys, yield_colorbar_f, med_lai, empirical_yield_min, empirical_yield_max = get_lai_gif(field_id)
    field_med_lai_line.x = doys
    field_med_lai_line.y = med_lai

    # daily_img = None
#     if daily_img in my_map.layers:
#         my_map.remove_control(daily_img)



    image = yield_colorbar_f.getvalue()
    empirical_colorbar = widgetIMG(value=image, format='png',)
    field_pics_base_url = 'https://github.com/UCL-EO/Ghana_field_images/raw/main/fields/'
    if field_id in Ghana_field_photo_dict.keys():
        urls = [field_pics_base_url + '/%s/%s'%(field_id, i) for i in Ghana_field_photo_dict[field_id]['files']]
        dates = Ghana_field_photo_dict[field_id]['dates']
    else:
        urls = []
        dates = []
    field_image_tab = create_field_image_tab(urls, dates)
    
#     yield_colorbar = WidgetControl(widget=output, position='bottomleft', transparent_bg=True)
#     yield_colorbar.widget = output
#     my_map.add_control(yield_colorbar)

    ylds = np.array(field_yields[field_id])
    
    yld_mean, yld_std = ylds.mean(), ylds.std()
    twso_vline.x = [365, 365]
    twso_vline.y = [0, yld_mean]

    twso_hline.x = [0, 365]
    twso_hline.y = [yld_mean, yld_mean]

    twso_shade.x = [0, 365]
    twso_shade.y = [[yld_mean - yld_std, yld_mean - yld_std], [yld_mean + yld_std, yld_mean + yld_std]]

    xmax = wofost_out_dict['LAI'].marks[0].x.max()

    twso_vline.x = [xmax, xmax]
    twso_vline.y = [0, yld_mean]

    twso_hline.x = [0, xmax]
    twso_hline.y = [yld_mean, yld_mean]

    twso_shade.x = [0, xmax]
    twso_shade.y = [[yld_mean - yld_std, yld_mean - yld_std], [yld_mean + yld_std, yld_mean + yld_std]]
    
    ymax = (yld_mean + 3 * yld_std)
    ymax = empirical_yield_max
    scales = twso_vline.scales
    scales['y'] = LinearScale(max=ymax, min=0)
    
    twso_vline.scales = scales
    twso_hline.scales = scales
    twso_shade.scales = scales
    
    # twso_vline.scales['y'] = LinearScale(max=yld_mean*1.5, min=0)
    wofost_out_dict['TWSO'].axes[1].scale  = LinearScale(max=ymax, min=0)
    wofost_out_dict['TWSO'].axes[1].tick_values = np.linspace(0, ymax, 5)
    wofost_out_dict['TWSO'].marks[1].scales = scales
    
    # for wofost_out_para in wofost_out_paras:
    #     scales =  wofost_out_dict[wofost_out_para].marks[1].scales
    #     dvs_labels_dict[wofost_out_para].scales = scales

    for wofost_out_para in wofost_out_paras:
        wofost_fig_vlines[wofost_out_para].x = [scales['x'].min, scales['x'].min]

    import scipy.stats as st

    cl = st.t.interval(0.95, len(ylds)-1, loc=np.mean(ylds), scale=st.sem(ylds))
    cl = (cl[1] + cl[0]) / 2
    
    
    district_name = df[df.CODE==field_id].District.iloc[0]
    
    wofost_yield_img_fname, wofost_yield_colorbar_f = get_wofost_yield(field_id, empirical_yield_min, empirical_yield_max )
    
    wofost_yield_unc_img_fname, wofost_yield_unc_colorbar_f = get_wofost_yield_unc(field_id)
    
    wofost_yield_label = Label('$$Yield [kg/ha]$$')

    image = wofost_yield_colorbar_f.getvalue()
    wofost_yield_colorbar = widgetIMG(value=image, format='png',)
    
    image = wofost_yield_unc_colorbar_f.getvalue()
    wofost_yield_unc_colorbar = widgetIMG(value=image, format='png',)
    
#     wofost_colorbar_box = VBox(, 
#                     layout=Layout(display='flex',flex_flow='column',align_items='center')
#                    )
    
    colorbar_box_dict['Wofost yield colorbar'] = [wofost_yield_label, wofost_yield_colorbar]
    colorbar_box_dict['Wofost yield unc. colorbar'] = [wofost_yield_label, wofost_yield_unc_colorbar]
    
    empirical_yield_label = Label('$$Yield [kg/ha]$$')
    # empirical_colorbar_box = VBox([empirical_yield_label, empirical_colorbar], 
    #                 layout=Layout(display='flex',flex_flow='column',align_items='center')
    #                )
    colorbar_box_dict['Empirical yield colorbar'] = [empirical_yield_label, empirical_colorbar]
    
    
    colorbar_dropdown.value = 'Wofost yield colorbar'
    
    encoded = base64.b64encode(open(wofost_yield_img_fname, 'rb').read())
    wofost_yield_img_url = "data:image/png;base64,%s"%encoded.decode()
    
    encoded = base64.b64encode(open(wofost_yield_unc_img_fname, 'rb').read())
    wofost_yield_unc_img_url = "data:image/png;base64,%s"%encoded.decode()
    
    yield_label1 = Label('%s (%s district)'%(field_id, district_name))
    yield_label2 = Label('Lat, Lon: %.05f, %.05f'%(lat, lon))
    
    yield_label3 = Label('Yield: %.02f [%.02f, %.02f, %.02f]'%(np.mean(ylds), ylds[0], ylds[1], ylds[2]))
    yield_box = VBox([yield_label1, yield_label2, yield_label3, colorbar_dropdown_box],
                     layout=Layout(object_fit='contain'))
    
    # label_box = HBox([play_label, maize_img], align_content = 'stretch', layout=Layout(width='100%', height='50%'))
    # yield_box = VBox([yield_label1, yield_label2, yield_label3, output, 
    #                   wofost_yield_label, wofost_yield_colorbar, 
    #                   lai_colorbar_label, lai_colorbar_output],
    #                  layout=Layout(width='100%', height='50%', object_fit='contain'))
    # yield_lai_box = VBox([yield_box, lai_box])
    
    field_df = df[df.CODE==field_id].dropna(subset=['COMMENTS'])

    comment_labels = []
    for i in range(len(field_df)):
        row = field_df.iloc[i]
        date = datetime.datetime(2021, int(row.DATE.split('/')[1]), int(row.DATE.split('/')[0])).strftime('%Y-%m-%d (DOY: %j)')
        commment = row.COMMENTS
        comment_str = '%s: %s'%(date, commment)
        comment_label = Label(value = comment_str)
        comment_labels.append(comment_label)
    field_comments_tab = VBox(comment_labels)

  
    comment_labels = []
    for i in range(len(field_df)):
        row = field_df.iloc[i]
        date = datetime.datetime(2021, int(row.DATE.split('/')[1]), int(row.DATE.split('/')[0])).strftime('%Y-%m-%d (DOY: %j)')
        commment = row.COMMENTS
        comment_str = '%s: %s'%(date, commment)
        comment_label = Label(value = comment_str)
        comment_labels.append(comment_label)
    field_comments_tab = VBox(comment_labels, layout=Layout(min_height='250px'))

    field_df = df[df.CODE==field_id].dropna(subset=['Phynology Data'])

    pheo_labels = []
    for i in range(len(field_df)):
        row = field_df.iloc[i]
        date = datetime.datetime(2021, int(row.DATE.split('/')[1]), int(row.DATE.split('/')[0])).strftime('%Y-%m-%d (DOY: %j)')
        pheo = row.loc['Phynology Data']
        pheo_str = '%s: %s'%(date, pheo)
        pheo_label = Label(value = pheo_str)
        pheo_labels.append(pheo_label)
    field_pheos_tab = VBox(pheo_labels, layout=Layout(min_height='250px'))


    yield_field_photo_tab = Tab([yield_box, field_image_tab, field_pheos_tab, field_comments_tab])
    yield_field_photo_tab.set_title(0, 'Field Yield')
    yield_field_photo_tab.set_title(1, 'Field Photos')
    yield_field_photo_tab.set_title(2, 'Phenology')
    yield_field_photo_tab.set_title(3, 'Comments')
    
    if yield_control is not None:        
        my_map.remove_control(yield_control)
        
    yield_control = WidgetControl(widget=yield_field_photo_tab, position='bottomleft')

    my_map.add_control(yield_control)

    url = 'data/S2_thumbs/S2_%s_yield.png'%(field_id)
    print(url)
    # dates = [(datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(i-1))).strftime('%Y-%m-%d') for i in doys]
    dates = [(datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(i-1))).strftime('%Y-%m-%d (DOY: %j)') for i in doys]
    slider2.options = dates
    field_bounds = bounds
    
    encoded = base64.b64encode(open(url, 'rb').read())
    url = "data:image/png;base64,%s"%encoded.decode()
    
    # url = base_url + url
    # print(url)
    field_mask = df.CODE == field_id

    field_doys = [int(datetime.datetime(2021, int(i.split('/')[1]), int(i.split('/')[0])).strftime('%j')) for i in df[field_mask].DATE]
    field_lais = np.array(df[field_mask].loc[:, 'LAI 1': 'LAI 4'])
    field_cabs = np.array(df[field_mask].loc[:, 'PLT. CHLRO 1': 'PLT. CHLOR 5'])

    # filed_cab_line = OHLC(x=dates2, y=np.array(prices2) / 60, marker='candle', 
    #                       scales={'x': sc_x, 'y': sc_y}, colors=['dodgerblue','orange'])


    # print(url)
#     if field_movie is not None:
#         my_map.remove_layer(field_movie)

#     field_movie = ImageOverlay(
#         url=url,
#         bounds = bounds,
#         name = 'S2_%s_lai_movie'%(field_id)
#     )
    # my_map.add_layer(field_movie)
    

    
    global yield_img, wofost_yield_img, wofost_yield_unc_img
    
    if yield_img is None:
        # print(url)
        # print(field_bounds)
        # print('S2_%s_lai_png'%(field_id))
        yield_img = ImageOverlay(
        url=url,
        bounds = field_bounds,
        name = 'Empirical yield map'#%(field_id)
        )
        my_map.add_layer(yield_img)
        yield_img.url = url
        yield_img.bounds = field_bounds
    else:
        my_map.remove_layer(yield_img)
        yield_img = ImageOverlay(
            url=url,
            bounds = field_bounds,
            name = 'Empirical yield map'#%(field_id)
        )
        my_map.add_layer(yield_img)
        # daily_img.url = url
        # daily_img.bounds = field_bounds
        
    if wofost_yield_unc_img is None:
        # print(url)
        # print(field_bounds)
        # print('S2_%s_lai_png'%(field_id))
        wofost_yield_unc_img = ImageOverlay(
            url=wofost_yield_unc_img_url,
            bounds = field_bounds,
            name = 'Wofost Yield uncertainty map'#%('#%(field_id)
        )
        my_map.add_layer(wofost_yield_unc_img)
        wofost_yield_unc_img.url = wofost_yield_unc_img_url
        wofost_yield_unc_img.bounds = field_bounds
    else:
        my_map.remove_layer(wofost_yield_unc_img)
        wofost_yield_unc_img = ImageOverlay(
            url=wofost_yield_unc_img_url,
            bounds = field_bounds,
            name = 'Wofost Yield uncertainty map'#%(field_id)
        )
        my_map.add_layer(wofost_yield_unc_img)
        # daily_img.url = url
        # daily_img.bounds = field_bounds
        
    if wofost_yield_img is None:
        # print(url)
        # print(field_bounds)
        # print('S2_%s_lai_png'%(field_id))
        wofost_yield_img = ImageOverlay(
            url=wofost_yield_img_url,
            bounds = field_bounds,
            name = 'Wofost Yield map'#%('#%(field_id)
        )
        my_map.add_layer(wofost_yield_img)
        wofost_yield_img.url = wofost_yield_img_url
        wofost_yield_img.bounds = field_bounds
    else:
        my_map.remove_layer(wofost_yield_img)
        wofost_yield_img = ImageOverlay(
            url=wofost_yield_img_url,
            bounds = field_bounds,
            name = 'Wofost Yield map'#%(field_id)
        )
        my_map.add_layer(wofost_yield_img)
        # daily_img.url = url
        # daily_img.bounds = field_bounds
        
        
    jslink((slider, 'value'), (yield_img, 'opacity') )
    
    try:
        my_map.remove_layer(daily_img)
    except:
        pass
    try:
        my_map.remove_layer(maize_marker)
    except:
        pass
    try:
        for this_marker, lai_ratio in maize_markers:
            my_map.remove_layer(this_marker)
    except:
        pass
    daily_img = None
    maize_marker = None
    maize_markers = []
    #for layer in my_map.layers:
        #if layer.name == 'S2_%s_lai_movie'%field_id:
            # Connect slider value to opacity property of the Image Layer


dropdown.observe(on_click, "value")
widget_control3 = WidgetControl(widget=dropdown, position="bottomleft")
my_map.add_control(widget_control3)


def on_change_slider2(change):

    if (change['name'] == 'value') & (change['type'] == 'change'):
        value = change["new"]
        old = change['old']
        ind = change['owner'].index

        global daily_img, lai_dot

        value = doys[ind]
        home = os.getcwd()
        cwd = '/files/' + '/'.join(home.split('/')[3:])
        base_url = my_map.window_url.split('/lab/')[0] + cwd + '/'
        
        i_lab, i_tree = my_map.window_url.find('/lab/'),  my_map.window_url.find('/tree/')
        to_replace = my_map.window_url[i_lab:i_tree+6]
        base_url = '/'.join(my_map.window_url.replace(to_replace, '/files/').split('/')[:-1]) + '/'
        base_url

        url = 'data/S2_thumbs/S2_%s_lai_%03d.png'%(field_id, value)
        
        encoded = base64.b64encode(open(url, 'rb').read())
        url = "data:image/png;base64,%s"%encoded.decode()

        # url = base_url + url
        field_bounds, _ = get_field_bounds(field_id)

#         if daily_img is not None:
#             my_map.remove_layer(daily_img)

#         daily_img = ImageOverlay(
#             url=url,
#             bounds = field_bounds,
#             name = 'S2_%s_lai_png'%(field_id)
#         )

#         my_map.add_layer(daily_img)
        # my_map.remove_layer(maize_marker) 

        # maize_marker.icon.icon_size = [(38*lai_ratio), int(95*lai_ratio)]
        if maize_marker is not None:
            lai_ratio = (pix_lai[ind] - 0) / (field_max - 0)
            
            
            lai_ratio = pix_lai[ind] / pix_lai.max()
            
            max_ind = np.argmax(pix_lai)
            max_ratio = pix_lai.max() / field_max
            
            if ind < max_ind:
                
                img_ind = int(lai_ratio * 5) + 1
                stage_ratio = lai_ratio / (img_ind / 5)
                
                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png', 
                            icon_size=[36.5*lai_ratio, 98.5*lai_ratio], 
                            icon_anchor=[36.5/2*lai_ratio, 98.5*lai_ratio])

                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                            icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                            icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])
                
                for (this_marker, max_ratio) in maize_markers:
 
                    
                    maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                                icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                                icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])

                    this_marker.icon = maize_icon
                    
            elif (lai_ratio > 0.8) & (ind >= max_ind):
                img_ind = 6
                stage_ratio = 1
                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                            icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                            icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])
            
                maize_marker.icon = maize_icon
                
                    
                for (this_marker, max_ratio) in maize_markers:
 
                    
                    maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                                icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                                icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])

                    this_marker.icon = maize_icon
            
                
            else:
                img_ind = np.min([int((1-lai_ratio / 0.8) * 6) + 7, 11])
                
                stage_ratio = (1-lai_ratio) / ((img_ind - 7) / 4)
                
                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png', 
                            icon_size=[36.5*lai_ratio, 98.5*lai_ratio], 
                            icon_anchor=[36.5/2*lai_ratio, 98.5*lai_ratio])

                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                            icon_size=[28.4 * max_ratio , 82.6 * max_ratio], 
                            icon_anchor=[28.4/2 * max_ratio , 82.6 * max_ratio])
            
                maize_marker.icon = maize_icon
                
                stage_ratio = 1
                for (this_marker, max_ratio) in maize_markers:
 
                    
                    maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                                icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                                icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])

                    this_marker.icon = maize_icon
                
            lai_dot.x = doys[[ind]]
            lai_dot.y = pix_lai[[ind]]
            lai_dot_wofost.x = doys[[ind]]
            lai_dot_wofost.y = pix_lai[[ind]]
#             lai_vline.x = doys[[ind]]
#             lai_vline.y = [3,]
            
            for wofost_out_para in wofost_out_paras:
                wofost_fig_vlines[wofost_out_para].x = [doys[ind], doys[ind]]
                
            s2_bio_vline.x = [doys[ind], doys[ind]]
            # lai_vline.x = 
            # lai_vline.y = [0, 3]
    
        # maize_marker.icon.icon_size=[38*lai_ratio, 95*lai_ratio] 
        # maize_marker.icon.icon_anchor=[22*lai_ratio,94*lai_ratio]
        # my_map.add_layer(maize_marker) 
        # print(maize_icon)   
        # print(lai_ratio)

        if daily_img is None:
            daily_img = ImageOverlay(
            url=url,
            bounds = field_bounds,
            name = 'LAI map'#%(field_id)
            )
            my_map.add_layer(daily_img)
            daily_img.url = url
            daily_img.bounds = field_bounds
        else:
            daily_img.url = url
            daily_img.bounds = field_bounds
            daily_img.name = 'LAI map'#%(field_id)
        # print(url)
        # jslink((slider, 'value'), (daily_img, 'opacity') )

play = Play(
    value=0,
    min=0,
    max=len(doys),
    step=1,
    interval=200,
    description="Press play",
    disabled=False
)

dates = [(datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(i-1))).strftime('%Y-%m-%d (DOY: %j)') for i in doys]
slider2 = SelectionSlider(options = dates, description='Date: ', style={'description_width': 'initial'}, layout = Layout(width='340px')) 
slider2.observe(on_change_slider2)
# widget_control2 = WidgetControl(widget=slider2, position="bottomright")
jslink((play, 'value'), (slider2, 'index'))
# label = 
# display(label)



play_label = Label('Click to play LAI movie over field: %s'%field_id)
# label_box = HBox([play_label, maize_img], align_content = 'stretch', layout=Layout(width='100%', height='50%'))
play_box = HBox([play, slider2])

play_box = VBox([play_label, play_box])

widget_control2 = WidgetControl(widget=play_box, position="bottomright")
my_map.add_control(widget_control2)


transparency_label = Label('Transparency:')
transparency_box = HBox([transparency_label, slider])
# my_map.add_control(WidgetControl(widget=transparency_box, position="bottomright"))

# lai_control = WidgetControl(widget=lai_box, position="bottomright")
# my_map.add_control(lai_control)

def random_color(feature):
    return {
        'color': 'black',
        'fillColor': np.random.choice(['red', 'yellow', 'green', 'orange']),
    }


fields = GeoJSON(
    data=data,
    style={
        'opacity': 1, 'dashArray': '0', 'fillOpacity': 0, 'weight': 1
    },
    hover_style={
        'color': 'white', 'dashArray': '0', 'fillOpacity': 0
    },
    name = 'Field boundaries',
    style_callback=random_color
)

def mouse_click_field(**kwarg):
    dropdown.value = kwarg['feature']['properties']['Field_ID']
fields.on_click(mouse_click_field)

with open('./data/Biophysical_Data_Collection_Points_V1.geojson', 'r') as f:
    data2 = json.load(f)
features_with_yield = []
for feature in data2['features']:
    if feature['properties']['Code_1'][:7] in codes:
        features_with_yield.append(feature)
data2['features'] = features_with_yield

points = GeoJSON(
    data=data2,
    point_style={'radius': 5, 'color': 'blue', 'fillOpacity': 0.5, 'fillColor': 'blue', 'weight': 0.1},
    # style={
    #     'opacity': 1, 'dashArray': '0', 'fillOpacity': 0.2, 'weight': 0.01
    # },
    hover_style={
        'color': 'white', 'dashArray': '0', 'fillOpacity': 1
    },
    name = 'Field measurement points',
    # style_callback=random_color
)



label = Label()
display(label)

maize_marker = None
pix_lai = None
maize_icon = None
marker_lon = None
marker_lat = None

def handle_interaction(**kwargs):
    
    # if kwargs.get('type') == 'mousemove':
    #     label.value = str(kwargs.get('coordinates'))
    if kwargs.get('type') == 'click':

        location=kwargs.get('coordinates')
        # print(location)
        
        point = geometry.Point(location[1], location[0])

        ind = field_ids.index(field_id)
        feature = data['features'][ind]
        poly = geometry.Polygon(feature['geometry']['coordinates'][0])
        field_mask = df.CODE == field_id

        field_doys = [int(datetime.datetime(2021, int(i.split('/')[1]), int(i.split('/')[0])).strftime('%j')) for i in df[field_mask].DATE]

        if poly.contains(point):    
            global sels, planet_sur, doys
            global maize_icon, pix_lai, field_max, field_min, field_lai_boxes, lai_dot, maize_markers, marker_lon, marker_lat, maize_marker
            
            label.value = 'Maize planted in field: %s'%field_id
            # global 
            # if maize_marker is not None:
            try:
                my_map.remove_layer(maize_marker)
                maize_markers = []
            except:
                pass
            # marker = Marker(location=location)

            marker_lon, marker_lat = location
            
            doys, pix_lai, pix_cab, sels, lais, planet_sur = get_pixel(location, field_id)

            mean_ref, mean_bios, std_ref, std_bios, sel_inds, u_mask = da_pix(sels, planet_sur, u_thresh = k_slider.value)

            show_inds = np.random.choice(range(sels.shape[2]), 50)
            sels_to_show = sels[:, :, show_inds]

            pix_cab, pix_lai = mean_bios

            max_ind = np.argmax(pix_lai)

            field_max = lais[:, max_ind].max()
            field_min = lais[:, max_ind].min() 

            lai_ratio = (pix_lai.max() - field_min) / (field_max - field_min)
            lai_ratio = (pix_lai.max() - 0) / (field_max - 0)
            # print(field_max, field_min)
            # print(lai_ratio)

#             icon = Icon(icon_url='https://leafletjs.com/examples/custom-icons/leaf-green.png', icon_size=[(38*lai_ratio), int(95*lai_ratio)], icon_anchor=[22,94])
            maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png', 
                        icon_size=[36.5*lai_ratio, 98.5*lai_ratio], 
                        icon_anchor=[36.5/2*lai_ratio, 98.5*lai_ratio])

            maize_marker = Marker(location=location, icon=maize_icon, rotation_angle=0, rotation_origin='22px 94px', draggable=False, name='Maize marker')
            maize_markers.append([maize_marker, lai_ratio])
            
            my_map.add_layer(maize_marker)                        



            sels_to_show = sels[:, :, sel_inds]
            for ii in range(6):
                line, ax_x, ax_y = line_axs[ii]
                y_scale = LinearScale(min = sels_to_show[ii].T.min(), max = sels_to_show[ii].T.max())
                ax_y.scale = y_scale
                line.x = doys
                line.y = sels_to_show[ii].T
                line.scales = {"x": x_scale, "y": y_scale}

            # cab_fig = figy[2].children[0]
            # lai_fig = figy[2].children[1]

            # field_cab_boxes = Boxplot(x=field_doys[:-1], y=field_cabs[:-1], 
            #                           scales=cab_fig.marks[0].scales, box_fill_color='gray')
            # field_cab_boxes.auto_detect_outliers=False
            # # field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
            # #                       scales=cab_fig.marks[0].scales, colors=['orange'])
            # field_cab_boxes.stroke = 'red'
            # field_cab_boxes.box_fill_color = 'blue'
            # field_cab_boxes.opacities = [0.5]
            # field_cab_boxes.box_width=5
            # cab_fig.marks = cab_fig.marks[:2] + [field_cab_boxes,]


            # field_lai_boxes = Boxplot(x=field_doys[:-1], y=field_lais[:-1], 
            #                           scales=lai_fig.marks[1].scales, box_fill_color='gray')
            # field_lai_boxes.auto_detect_outliers=False
            # # field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
            # #                       scales=cab_fig.marks[0].scales, colors=['orange'])
            # field_lai_boxes.stroke = 'red'
            # field_lai_boxes.box_fill_color = 'blue'
            # field_lai_boxes.opacities = [0.4]
            # field_lai_boxes.box_width=5
            # lai_fig.marks = lai_fig.marks[:2] + [field_lai_boxes,]

            # var_line = line_axs[-2]
            # var_line.x = doys
            # var_line.y = pix_cab
            # var_line.scales = line_axs[4][0].scales
            # field_cab_boxes.scales = var_line.scales

            var_line = line_axs[-1]
            var_line.x = doys
            var_line.y = pix_lai
            var_line.scales = line_axs[5][0].scales
            field_lai_boxes.scales = var_line.scales
            field_lai_boxes.x = field_doys
            field_lai_boxes.y = field_lais
            
            s2_bio_plot_fied_avg_line_dict['lai'].x = field_doys
            s2_bio_plot_fied_avg_line_dict['lai'].y = field_lais
            

            s2_bio_plot_fied_avg_line_dict['cab'].x = field_doys
            s2_bio_plot_fied_avg_line_dict['cab'].y = field_cabs
            
            lai_dot.scales = var_line.scales
            field_med_lai_line.scales = var_line.scales
                
            for ii in range(4):
                line, ax_x, ax_y = line_axs[ii]
                ref_line = ref_lines[ii]
                ref_line.scales = line.scales
                ref_line.x = doys[~u_mask]
                ref_line.y = planet_sur[ii][~u_mask]
            
            ndvi = (planet_sur[3] - planet_sur[2]) / (planet_sur[3] + planet_sur[2])
            
            line, ax_x, ax_y = line_axs[4]
            ref_line = ref_lines[4]
            ref_line.scales = line.scales
            ref_line.x = doys[~u_mask]
            ref_line.y = ndvi[~u_mask]

            # print(planet_sur.shape, u_mask.shape)
            for ii in range(4):
                line, ax_x, ax_y = line_axs[ii]
                good_ref_line = good_ref_lines[ii]
                good_ref_line.scales = line.scales
                good_ref_line.x = doys[u_mask]
                good_ref_line.y = planet_sur[ii][u_mask]
                
            line, ax_x, ax_y = line_axs[4]
            good_ref_line = good_ref_lines[4]
            good_ref_line.scales = line.scales
            good_ref_line.x = doys[u_mask]
            good_ref_line.y = ndvi[u_mask]
            
            wofost_out_dict['LAI'].marks[5].x = line_axs[-1].x
            wofost_out_dict['LAI'].marks[5].y = line_axs[-1].y

            s2_pix_bio_dict = get_pixel_s2_bios(marker_lon, marker_lat, s2_projectionRef, s2_geo_trans, s2_bios)

            for s2_bio_name in s2_bios_to_plot:
                s2_bio_plot_line_dict[s2_bio_name].x = s2_bio_doys
                s2_bio_plot_line_dict[s2_bio_name].y = s2_pix_bio_dict[s2_bio_name]


        else:
            label.value = 'Not in field: %s'%field_id
            # print('Not in field: %s'%field_id)


def on_change_k_slider(change):

    if (change['name'] == 'value') & (change['type'] == 'change'):
        value = change["new"]
        old = change['old']
        mean_ref, mean_bios, std_ref, std_bios, sel_inds, u_mask = da_pix(sels.copy(), planet_sur.copy(), u_thresh = value)

        show_inds = np.random.choice(range(200), 50)
        sels_to_show = sels[:, :, show_inds]

        global field_lai_boxes

        sels_to_show = sels[:, :, sel_inds]
        for ii in range(6):
            line, ax_x, ax_y = line_axs[ii]
            y_scale = LinearScale(min = sels_to_show[ii].T.min(), max = sels_to_show[ii].T.max())
            # ax_y.scale = y_scale
            ax_y.scale = ax_y.scale
            line.x = doys
            line.y = sels_to_show[ii].T
            line.scales = {"x": x_scale, "y": y_scale}

        pix_cab, pix_lai = mean_bios

        # var_line = line_axs[-2]
        # var_line.x = doys
        # var_line.y = pix_cab
        # var_line.scales = line_axs[4][0].scales


        var_line = line_axs[-1]
        var_line.x = doys
        var_line.y = pix_lai
        var_line.scales = line_axs[5][0].scales

        # var_line = line_axs[-2]
        # var_line.x = doys
        # var_line.y = pix_cab
        # var_line.scales = line_axs[4][0].scales
        # field_cab_boxes.scales = var_line.scales

        var_line = line_axs[-1]
        var_line.x = doys
        var_line.y = pix_lai
        var_line.scales = line_axs[5][0].scales
        field_lai_boxes.scales = var_line.scales
        field_lai_boxes.x = field_doys
        field_lai_boxes.y = field_lais

        for ii in range(4):
            line, ax_x, ax_y = line_axs[ii]
            ref_line = ref_lines[ii]
            ref_line.scales = line.scales
            ref_line.x = doys[~u_mask]
            ref_line.y = planet_sur[ii][~u_mask]
            
        ndvi = (planet_sur[3] - planet_sur[2]) / (planet_sur[3] + planet_sur[2])

        line, ax_x, ax_y = line_axs[4]
        ref_line = ref_lines[4]
        ref_line.scales = line.scales
        ref_line.x = doys[~u_mask]
        ref_line.y = ndvi[~u_mask]

        # print(planet_sur.shape, u_mask.shape)
        for ii in range(4):
            line, ax_x, ax_y = line_axs[ii]
            good_ref_line = good_ref_lines[ii]
            good_ref_line.scales = line.scales
            good_ref_line.x = doys[u_mask]
            good_ref_line.y = planet_sur[ii][u_mask]
        line, ax_x, ax_y = line_axs[4]
        good_ref_line = good_ref_lines[4]
        good_ref_line.scales = line.scales
        good_ref_line.x = doys[u_mask]
        good_ref_line.y = ndvi[u_mask]
        lai_dot.scales = var_line.scales
        field_med_lai_line.scales = var_line.scales
        
        wofost_out_dict['LAI'].marks[5].x = line_axs[-1].x
        wofost_out_dict['LAI'].marks[5].y = line_axs[-1].y

k_slider.observe(on_change_k_slider)
my_map.on_interaction(handle_interaction)
my_map.add_layer(fields)
my_map.add_layer(points)



info_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/info.png'
info = requests.get(info_url)

usage_text_box = VBox([HTML(value = f"<b><font color='red' font weight='bold'>Usage</b>"),
                       Label('1. Choose a field ID from the dropdown list to load data'),
                       Label('2. Scroll left to see the field photos in the Field Photos tab'),
                       Label('3. Using the Date slider to load the LAI over the field or click to play LAI movie'),
                       Label('4. Click the map within the field to check the pixel values'),
                       Label('5. Using sliders in the WOFOST simulation tab to fit the field LAI'),
                       Label('6. Auto fit can automatically fit the Planet LAI'),
                       ])

info_img = widgetIMG(value=info.content,
  format='png', 
  width=30,
  align="center")
info_box = VBox([info_img, usage_text_box])
info_control = WidgetControl(widget=info_box, position='topleft')

info_image_event = Event(source=info_img, watched_events=['mouseenter', 'mouseleave'])
def info_image_event_handler(event):
    if event['event'] == 'mouseenter':
        info_box.children = [info_img, usage_text_box]
    if event['event'] == 'mouseleave':
        info_box.children = [info_img]
info_image_event.on_dom_event(info_image_event_handler)


usage_text_box_event = Event(source=usage_text_box, watched_events=['mouseleave'])
def usage_text_box_event_handler(event):
    if event['event'] == 'mouseleave':
        info_box.children = [info_img]
usage_text_box_event.on_dom_event(usage_text_box_event_handler)


layer_control = LayersControl(position='topleft')
my_map.add_control(layer_control)

my_map.add_control(info_control)
# my_map


font_size = '12'
font_color = 'white'
font_family = 'arial'
font_weight = 'normal'
labels = []
for i in fields.data['features']:
    field_name = i['properties']['Field_ID']
    coords = i['geometry']['coordinates'][0]
    x,y = geometry.Polygon(coords).centroid.xy
    # from https://github.com/giswqs/geemap/blob/master/geemap/geemap.py#L6583
    html = f'<div style="font-size: {font_size};color:{font_color};font-family:{font_family};font-weight: {font_weight}">{field_name}</div>'
    marker = Marker(
                location=[y[0], x[0]],
                icon=DivIcon(
                    icon_size=(1, 1),
                    icon_anchor=(10, 10),
                    html=html,
                ),
                draggable=False
            )
    
    labels.append(marker)
    
marker_cluster = MarkerCluster(
    markers=labels,
    # disable_clustering_at_zoom=15,
    # max_cluster_radius=100,
    
)

my_map.add_layer(marker_cluster)