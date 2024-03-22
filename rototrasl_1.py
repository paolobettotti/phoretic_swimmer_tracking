import numpy as np
import pims as pm
import av
from skimage.filters import threshold_minimum, threshold_otsu
from skimage.measure import label, regionprops
import skimage.transform as skt
from skimage import exposure
from skimage.util import img_as_ubyte
import sys
import os
import pickle
from tqdm import tqdm

####################################################################################
# FUNCTIONS DEF
def find_pool(image):
# search pool threshold in channel 2
    img = img_as_ubyte(image)
    p5, p95 = np.percentile(img, (5, 95))
    img = exposure.rescale_intensity(img, in_range=(p5, p95))
    threshold = threshold_otsu(img)
    binary = img > threshold
    label_img = label(binary, connectivity=binary.ndim)
    props = regionprops(label_img)
    try:
        pool_idx = np.argmax([props[tt].area for tt in range(len(props))])
        minr, minc, maxr, maxc = props[pool_idx].bbox
#check that pool is large (80% horizontal frame size) and circularly shaped        
        if (np.abs(maxc - minc) - np.abs(maxr - minr) < 70 and  binary.shape[0] - np.abs((maxr - minr)) < 0.2 * binary.shape[0]):
            np.abs((maxc - minc) - (maxr - minr)) < 0.05*np.min([(maxc - minc),(maxr - minr)])
            return {'pool frame idx': pool_idx, 'threshold value':threshold, 'minrow': minr, 'mincol':minc, 'maxrow':maxr, 'maxcol':maxc}
        else:
            print(f'Differences: {(maxc - minc)} {(maxr - minr)}')
            print(f'Pool NOT found!')
    except ValueError:
        print('Error: Pool NOT found!')
        return 'error'

###################################################################################
#INPUT FILE AND FRAMESET CREATION
filename = sys.argv[1]
root, extension = os.path.splitext(filename)
onlyname = os.path.basename(root)
print(f'Elaborating: {onlyname}') 

frameset = pm.PyAVReaderTimed(filename, cache_size= 100)

###################################################################################
# variables initialization
thresh = []
binary = np.zeros((frameset.shape[1],frameset.shape[2]))
cen_swim = []
ang = []
angular_range = list(range(-30,30))
pool_pos = []
missed = []

###################################################################################
# elaborate 1st frame of the chunk
try:
    pool_pos.append(find_pool(frameset[0][:,:,2]))
except:
    pass
    
thresh.append(threshold_minimum(frameset[0][:,:,0]))
binary = frameset[0][:,:,0] < thresh[0]
label_img = label(binary, connectivity=2)
props = regionprops(label_img)
larg_idx = np.argmax([tt.area for tt in props])
cen_swim.append(props[larg_idx].centroid)
cen_prev = cen_swim[0]
#find object diagonal in px
minr, minc, maxr, maxc = props[0].bbox
diag = np.rint(np.sqrt( (maxr - minr)**2 + (maxc - minc)**2)).astype(np.uint16)

# define array subset to rotate for the 1st frame
pad = 10
# cen_shift is the center of the centroid in the cropped and binarized images
cen_shift = diag/2 + pad

prev_inf_vert_lim = np.rint(cen_swim[0][0] - cen_shift).astype(int) # inf means towards the origin (= top row of the image!)
prev_sup_vert_lim = np.rint(cen_swim[0][0] + cen_shift).astype(int)
prev_inf_horz_lim = np.rint(cen_swim[0][1] - cen_shift).astype(int)
prev_sup_horz_lim = np.rint(cen_swim[0][1] + cen_shift).astype(int)
red_prev = binary[prev_inf_vert_lim:prev_sup_vert_lim, prev_inf_horz_lim:prev_sup_horz_lim]

###################################################################################
# RUN OVER ALL FRAMES
for count, item in enumerate(tqdm(frameset[1:])):
# find the pool position
    if count % 500 == 0:
        try:
            pool_pos.append(find_pool(item[:,:,2]))
        except:
            pass

# this speed up significantly the execution (around 20%)
    if count % 100 == 0:
        try:
            thresh.append(threshold_minimum(item[:,:,0]))
        except:
            print(f'Error at frame: {count}')
            missed.append(count)
        
    binary = item[:,:,0] < thresh[-1]
    label_img = label(binary, connectivity=2)
    props = regionprops(label_img)
    larg_idx = np.argmax([tt.area for tt in props])
    cen_swim.append(props[larg_idx].centroid)
    # cen_next serves to shift this frame with respect the previous one
    cen_next = props[larg_idx].centroid
    prev_inf_vert_lim = np.rint(cen_swim[-1][0] - cen_shift).astype(int) # inf means towards the origin (= top row of the image!)
    prev_sup_vert_lim = np.rint(cen_swim[-1][0] + cen_shift).astype(int)
    prev_inf_horz_lim = np.rint(cen_swim[-1][1] - cen_shift).astype(int)
    prev_sup_horz_lim = np.rint(cen_swim[-1][1] + cen_shift).astype(int)
    red_next = binary[prev_inf_vert_lim:prev_sup_vert_lim, prev_inf_horz_lim:prev_sup_horz_lim]
    
    res = []
    for ee in angular_range:
        try:

            res.append( np.cumsum( red_next + skt.rotate(red_prev,ee) ) [-1] )
        except ValueError:
            print('Missed frame centroid!')
            res.append(0)
            break

    ang.append(angular_range[np.argmin(res)])

    red_prev = red_next
    cen_prev = cen_next

sim_output = [np.asarray(cen_swim), np.asarray(thresh), np.asarray(missed), np.asarray(pool_pos)]
with open(onlyname +'.pkl', 'wb') as handle:
        pickle.dump(sim_output, handle)

