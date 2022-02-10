import os
import datetime
import requests
from io import BytesIO
import matplotlib.cm as cm
import numpy as np
import pyproj
# import imageio
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
    cab = f.f.mean_bios_all[:, 1]

    pj1 = Proj(projectionRef)
    transformer = Transformer.from_crs('EPSG:4326', pj1.crs)
    x, y = transformer.transform(lat, lon)
    inds = np.where(valid_mask)
    pix_x = int((x - geo_trans[0]) / geo_trans[1])
    pix_y = int((y - geo_trans[3]) / geo_trans[5])
    mm = (inds[0] == pix_y) & (inds[1] == pix_x)
    
    inds = f.f.unique_neighbour_inverse_inds[mm][0][:200]
    sels = f.f.unique_neighbour_s2_refs[:, :, inds][[0, 1, 2, 7]]
    bios = f.f.unique_neighbour_orig_bios[:, :, inds][[1, 4]]
    sels = np.concatenate([sels, bios])
    planet_sur = f.f.s2_sur_all[:, :, mm].squeeze()
             
    return doys, lai[mm].ravel(), cab[mm].ravel(), sels, lai, planet_sur

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
        TRANSPARENCY = 0.75  # Degree of transparency, 0-100%
        
        OPACITY = int(255 * TRANSPARENCY)
        draw.rectangle((x, y, x + w, y + h), fill=TINT_COLOR + (OPACITY,), )
        draw.rectangle((x-2, y-2, x + w+2, y + h+2), outline=(0, 0, 0, 127), width=1)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        this_alpha = img.getchannel('A')
        img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
        mask = Image.eval(this_alpha, lambda a: 255 if a <= 5 else 0)

        # Paste the color of index 255 and use alpha as a mask
        img.paste(255, mask)
        img.info['transparency'] = 255
        
        fname = './data/S2_thumbs/S2_%s_lai_%03d.png'%(field_name, doys[i])
        img.save(fname)
        
        frames.append(img)
        
        # frames.append(np.asarray(img))
        
        
    fp_out = './data/S2_thumbs/S2_%s_lai.gif'%(field_name)
    # imageio.mimsave(fp_out, frames, 'GIF', duration=0.2)
    # url = base_url + '/output.gif'
    # print(url)
    
    frames[0].save(fp_out, save_all=True, append_images=frames[1:], loop=0, duration=200, optimize=False)
    
    return 'data/S2_thumbs/S2_%s_lai.gif'%field_name, bounds, doys
    # print(bounds)
    
def get_field_bounds(field_name):
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

    return bounds, doys

# from PIL import Image

# def gen_frame(path):
#     im = Image.open(path)
#     alpha = im.getchannel('A')

#     # Convert the image into P mode but only use 255 colors in the palette out of 256
#     im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)

#     # Set all pixel values below 128 to 255 , and the rest to 0
#     mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)

#     # Paste the color of index 255 and use alpha as a mask
#     im.paste(255, mask)

#     # The transparency index is 255
#     im.info['transparency'] = 255

#     return im


# im1 = gen_frame('frame1.png')
# im2 = gen_frame('frame2.png')        
# im1.save('GIF.gif', save_all=True, append_images=[im2], loop=5, duration=200)