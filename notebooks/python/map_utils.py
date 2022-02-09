import os
import datetime
import requests
from io import BytesIO
import matplotlib.cm as cm
import numpy as np
import pyproj
import imageio
from pyproj import Proj, Transformer
from PIL import Image, ImageFont, ImageDraw 

    

def get_pixel(location, field_name):
    lat, lon = location
    npz_name = '%s_bios_planet_only_v5.npz'%field_name
    if not os.path.exists('./data/' + npz_name):
        tsen_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Tsen/'
        url = tsen_url + npz_name
        r = requests.get(url)
        if r.status_code != 200:
            r.raise_for_status()
        with open('./data/' + npz_name, 'wb') as f:
            f.write(r.content)

    f = np.load('./data/' + npz_name, allow_pickle=True)
    projectionRef = str(f.f.projectionRef)
    geo_trans = f.f.geotransform
    doys = f.f.doys
    valid_mask = f.f.valid_mask
    alpha = (valid_mask * 255.).astype(np.uint8)
    lai = f.f.mean_bios_all[:, 4]

    pj1 = Proj(projectionRef)
    transformer = Transformer.from_crs('EPSG:4326', pj1.crs)
    x, y = transformer.transform(lat, lon)
    inds = np.where(valid_mask)
    pix_x = int((x - geo_trans[0]) / geo_trans[1])
    pix_y = int((y - geo_trans[3]) / geo_trans[5])
    mm = (inds[0] == pix_y) & (inds[1] == pix_x)
    return doys, lai[mm].ravel()

def get_lai_gif(field_name):
    npz_name = '%s_bios_planet_only_v5.npz'%field_name
    if not os.path.exists('./data/' + npz_name):
        tsen_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Tsen/'
        url = tsen_url + npz_name
        r = requests.get(url)
        if r.status_code != 200:
            r.raise_for_status()
        with open('./data/' + npz_name, 'wb') as f:
            f.write(r.content)
    
    f = np.load('./data/' + npz_name, allow_pickle=True)
    projectionRef = str(f.f.projectionRef)
    geo_trans = f.f.geotransform
    doys = f.f.doys
    valid_mask = f.f.valid_mask
    alpha = (valid_mask * 255.).astype(np.uint8)
    lai = f.f.mean_bios_all[:, 4]



    pj1 = pyproj.Proj(projectionRef)
    transformer = Transformer.from_crs(pj1.crs, 'EPSG:4326')

    x_min = geo_trans[0]
    y_min = geo_trans[3]
    x_max = geo_trans[0] + valid_mask.shape[1] * geo_trans[1]
    y_max = geo_trans[3] + valid_mask.shape[0] * geo_trans[5]

    coords = np.array([[x_min, y_min], [x_max, y_max]])

    x_coords = [x_min, x_max]
    y_coords = [y_min, y_max]

    (x_min, x_max), (y_min, y_max) = transformer.transform(x_coords,y_coords)

    bounds = ((x_min, y_max), (x_max, y_min))
    print(bounds)


    cmap = cm.Greens
    frames = []
    for i in range(len(doys)):
        lai_map = np.zeros(valid_mask.shape)
        lai_map[valid_mask] = lai[:, i]
        greyscale = cmap(lai_map / 2, bytes=True)
        greyscale[:, :, -1] = alpha
        # greyscale = np.concatenate([greyscale, alpha[:, :, None]])
        date = datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(doys[i]) -1)

        img = Image.fromarray(greyscale, mode='RGBA')

        scale = 256 / img.height
        new_height = int(scale * img.height)
        new_width = int(scale * img.width)

        img = img.resize(( new_width, new_height), resample = Image.NEAREST)
        draw = ImageDraw.Draw(img, "RGBA")
        #strptimet = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        font = font = ImageFont.load_default()
        text = date.strftime('%Y-%m-%d')
        w, h = font.getsize(text)
        x, y = 5, 5  


        TINT_COLOR = (0, 0, 0)  # Black
        TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
        OPACITY = int(255 * TRANSPARENCY)
        draw.rectangle((x, y, x + w, y + h), fill=TINT_COLOR + (OPACITY,), )
        draw.rectangle((x-2, y-2, x + w+2, y + h+2), outline=(0, 0, 0, 127), width=1)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        fname = './data/S2_thumbs/S2_%s_lai_%03d.png'%(field_name, doys[i])
        img.save(fname)
        frames.append(np.asarray(img))
    fp_out = './data/S2_thumbs/S2_%s_lai.gif'%(field_name)
    imageio.mimsave(fp_out, frames, 'GIF', duration=0.2)
    # url = base_url + '/output.gif'
    # print(url)
    return 'data/S2_thumbs/S2_%s_lai.gif'%field_name, bounds
    # print(bounds)
    