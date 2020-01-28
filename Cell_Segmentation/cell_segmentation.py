import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import copy
from scipy import ndimage

import skimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage import exposure

img = cv2.imread('hw2-9e6aa6f4.png')
#img_orig = cv2.imread('hw2-9e6aa6f4.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




###############################################################################
'''
An algorithm for local contrast enhancement, that uses histograms computed over 
different tile regions of the image. Local details can therefore be enhanced 
even in regions that are darker or lighter than most of the image.
'''
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
###############################################################################

###############################################################################
'''
Turn image into [0,255]-valued array
'''
kuva = img_adapteq * 255
kuva = kuva.astype('uint8')
gray = cv2.cvtColor(kuva, cv2.COLOR_BGR2GRAY)
###############################################################################

###############################################################################
'''
Use bilateral filter to smooth the image
'''
filtered = cv2.bilateralFilter(gray,9,80,0)
###############################################################################

###############################################################################
'''
'''
ret,thresh = cv2.threshold(filtered,120,255,cv2.THRESH_BINARY_INV)
###############################################################################

###############################################################################
'''
'''
# Implement Otsu's method

# Intensities are in range [0,255]
bins = np.arange(0,256,1)

P = plt.hist(gray.ravel(),bins,density=True)[0]

low_var = 100000
low_t = 1

for t in range (1,255):
    # Compute q1 and q2 for a given threshold (Eq. 2)
    q1 = np.sum(P[:t])
    q2 = np.sum(P[t:])

    # Mean values (Eq. 3)
    i1 = np.arange(0,t,1)
    i2 = np.arange(t,255,1)
    mean1 = np.sum(i1*P[:t])/q1
    mean2 = np.sum(i2*P[t:])/q2

    # Variances (Eq. 4)
    var1 = np.sum(((i1 - mean1)**2)*P[:t])/q1
    var2 = np.sum(((i2 - mean2)**2)*P[t:])/q2

    # Weighted within-group variance (Eq. 1)
    varw = q1*var1 + q2*var2
    if varw < low_var:
        low_var = varw
        low_t = t

#print("Lowest threshold: ",low_t)
#print("Lowest variance:  ",low_var)

t = low_t

# Compute q1 and q2 for a given threshold (Eq. 2)
q1 = np.sum(P[:t])
q2 = np.sum(P[t:])

# Mean values (Eq. 3)
i1 = np.arange(0,t,1)
i2 = np.arange(t,255,1)
mean1 = np.sum(i1*P[:t])/q1
mean2 = np.sum(i2*P[t:])/q2

# Variances (Eq. 4)
var1 = np.sum(((i1 - mean1)**2)*P[:t])/q1
var2 = np.sum(((i2 - mean2)**2)*P[t:])/q2

# Weighted within-group variance (Eq. 1)
varw = q1*var1 + q2*var2

ret,thresh2 = cv2.threshold(filtered , low_t, 255, cv2.THRESH_BINARY_INV)


# Compare to OpenCV's implementation. Notice that 'gray' is the original 
# image. You should change the input to the filtered image when comparing.
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
###############################################################################

###############################################################################
'''

'''
sel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

morphed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, sel1)


#sel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

#morphed2 = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, sel2)

###############################################################################

###############################################################################
'''
Switch 0's to 255's and 255's to 0's. Then take contours of that image.
'''
morphed[morphed == 255] = 100
morphed[morphed == 0] = 255
morphed[morphed == 100] = 0

cnt,_ = cv2.findContours(morphed, 1, 1)
###############################################################################

###############################################################################
'''
Draw contours of the image.
'''
final = np.zeros_like(otsu)

for i in range(len(cnt)):
    area = cv2.contourArea(cnt[i])

    if area > 200:# and area < 20000:
        cv2.drawContours(final, cnt, i, (255,255,255), thickness=-1)
###############################################################################

###############################################################################
'''
'''
distance = ndimage.distance_transform_edt(final)

local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=final)

markers = skimage.morphology.label(local_maxi)

labels_ws = watershed(-distance, markers, mask=final)

###############################################################################

###############################################################################
'''

'''
cnt_2,_ = cv2.findContours(labels_ws.astype('uint8'), 1, 1)

laabelit = np.unique(labels_ws)
for i in laabelit:
    temp_kuva = copy.deepcopy(labels_ws.astype('uint8'))
    temp_kuva[temp_kuva != i] = 0
    if i > 1 and np.sum(temp_kuva) > 0:

        cnt_temp,_ = cv2.findContours(temp_kuva, 1, 1)

        cnt_2.append(cnt_temp[0])
###############################################################################

###############################################################################
'''
Plot the new contours on the image
'''
plt.close() 

for i in range(len(cnt_2)):
    area = cv2.contourArea(cnt_2[i]) 

    if area > 200:# and area < 20000:
        cv2.drawContours(img_adapteq, cnt_2, i, (255,255,255), thickness=-1)
        plt.plot(cnt_2[i][:,0,0],cnt_2[i][:,0,1])
        
plt.gca().invert_yaxis()
plt.title('Contours')
plt.imshow(img)
plt.show()
###############################################################################

