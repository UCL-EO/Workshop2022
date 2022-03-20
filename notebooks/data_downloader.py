import os
import requests
import numpy as np
from multiprocessing import Pool
def downloader(url):
    fname = 'data/' + url.split('/')[-1]
    if not os.path.exists(fname):
        r = requests.get(url)
        with open(fname, 'wb') as f:
            f.write(r.content)
        
bio_urls = np.loadtxt('data/bio_urls.txt', dtype='str')
p = Pool(4)
p.map(downloader, bio_urls)
p.close()
p.join()

ens_urls = np.loadtxt('data/ens_urls.txt', dtype='str')
p = Pool(4)
p.map(downloader, ens_urls)
p.close()
p.join()


s2_bios_urls = np.loadtxt('data/s2_bios_urls.txt', dtype='str')
p = Pool(4)
p.map(downloader, s2_bios_urls)
p.close()
p.join()