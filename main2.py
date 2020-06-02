
import numpy as np
import cv2
import skimage.segmentation as seg
import matplotlib.pyplot as plt
from skimage import io, segmentation, morphology
from scipy import ndimage
import os

import levelset_2 as levelset

def det_mouth(im):

    face_classifier = cv2.CascadeClassifier('H:/My Documents/F19/F19-University/ENGG6090/project/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('H:/My Documents/F19/F19-University/ENGG6090/project/haarcascade_smile.xml')
    mouth_cascade2 = cv2.CascadeClassifier('H:/My Documents/F19/F19-University/ENGG6090/project/haarcascade_mcs_mouth.xml')

    X = 177
    Y = 88

    faces = face_classifier.detectMultiScale(im, 1.3, 5)

    if faces is ():
        print('No face found')
        return None, 0, 0
        
    for (x,y,w,h) in faces:
        roi_im = im[y:y+h, x:x+w]
        mouth = mouth_cascade.detectMultiScale(roi_im, 1.7, 11)
        break

    tempx = x
    tempy = y

    if mouth is ():
        mouth = mouth_cascade2.detectMultiScale(roi_im, 1.7, 11)
        if mouth is ():
            print('No mouth found')
            return None, 0, 0
    
    for (x,y,w,h) in mouth:
        #y1 = int(y1 - 0.15*h1)
        #out = roi_im[y:y+h, x:x+w]
        out = roi_im[int((y + h/2) - Y/2):int((y + h/2) - Y/2)+Y, int((x + w/2) - X/2):int((x + w/2) - X/2)+X]
        break

    y = tempy + int((y + h/2) - Y/2)
    x = tempx + int((x + w/2) - X/2)

    return out, x, y

# Unused
def snake(im, alpha=0.015, beta=1, gamma=0.005):

    y,x = im.shape

    # initialization
    s = np.linspace(0, 2*np.pi, 100)
    r = x/2 + 70*np.cos(s)
    c = y/2 + 30*np.sin(s)
    init = np.array([r, c]).T

    snake = segmentation.active_contour(
        ndimage.filters.gaussian_filter(im, 0.25),
        init, alpha=alpha, beta=beta, gamma=gamma, coordinates='rc',
        w_edge=2.5, boundary_condition='periodic')

    print(snake)
    
    plt.imshow(im, cmap='gray')
    plt.plot(init[:,0], init[:,1], '--r', lw=3)
    plt.plot(snake[:,0], snake[:,1], '-b', lw=3)
    plt.show()

'''
im = cv2.imread('//north.cfs.uoguelph.ca/soe-other-home$/twong05/My Documents/F19/F19-University/ENGG6090/project/s1.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

roi = det_mouth(im)

snake(roi)
'''
#print(det_mouth(im))


ms = 1
if ms == 0:
    prefpath = 'C:/Users/twong05/final/sad/'
    fchar = 'm'
elif ms == 1:
    prefpath = 'C:/Users/twong05/final/happy/'
    fchar = 's'
    
ftype = '.png'
outpath = 'C:/Users/twong05/final/output/'

#saliency = cv2.saliency.StaticSaliencyFineGrained_create()
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
# shape prior
prior = cv2.imread('H:/My Documents/F19/F19-University/ENGG6090/project/shape_prior.png')
prior = np.clip(cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY), 0, 1)

selem3 = morphology.disk(3)
selem5 = morphology.disk(5)
selem7 = morphology.disk(7)
selem9 = morphology.disk(9)
selem11 = morphology.disk(11)

dice = []

os.chdir('C:/Users/twong05/final/seg_eval')
for f in os.listdir():

    # load file
    fname = f
    file_name, file_ext = os.path.splitext(f)
    file_head, file_tail = os.path.split(f)
    if file_ext != '.png':
        continue
    
    try:
        img = cv2.imread(fname, 0)
        print('file ' + fname + ' loaded')
    except FileNotFoundError:
        print('File could not be loaded, skipping')
        continue

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # object detection
    roi, x, y = det_mouth(img)
    if roi is None:
        print('no face found')
        continue
    temp1,temp2 = roi.shape
    if temp1 <= 0 or temp2 <= 0:
        print('no face found')
        continue
    print('mouth detected')
    
    out = roi
    
    #roi = img

    # segmentation
    # Saliency
    (success, saliencyMap) = saliency.computeSaliency(roi)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    binout = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    
    binout = morphology.binary_dilation(binout, selem11)
    binout = morphology.binary_erosion(binout, selem5)
    binout = morphology.area_closing(binout.astype('uint8'), 128)
    '''
    #out = saliencyMap
    # level set
    
    print('segmenting by level set')
    #salinecyMap = cv2.equalizeHist(saliencyMap)
    #im1 = (saliencyMap/np.max(saliencyMap)) ** 0.5 * np.max(saliencyMap)
    #im1 = morphology.area_opening(cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    #im1 = morphology.area_closing(cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1], 1024)
    
    phi0 = levelset.default_phi(roi)
    phi = levelset.levelset(roi, phi0, 200, prior, dt=1, v=0.1, mu=500, lamb1=1, lamb2=1, tau=0.05)

    binout = np.heaviside(phi, 1)
    
    binout = morphology.binary_closing(binout, selem5)    
    binout = morphology.binary_opening(binout, selem5)
    '''
    #out = roi * binout
    print('level set complete')
    
    # compare to reference
    # load reference segmentation
    
    try:
        ref = cv2.imread(file_head + 'out2/out2-' + file_tail, 0)
    except:
        print('could not read reference seg file')
    # match segmentation to original image
    bw_final = np.zeros(img.shape)
    bw_final[y:y+binout.shape[0], x:x+binout.shape[1]] = np.copy(binout)
    # compute dice
    intersection = np.logical_and(bw_final.astype(bool), ref.astype(bool))
    plt.imshow(ref)
    plt.show()
    dice.append(2. * np.sum(intersection) / (bw_final.astype(bool).sum() + ref.astype(bool).sum()))
    #print(dice)
    
    
    # save file
    fname = file_head + 'output/' + file_tail
    #try:
    io.imsave(fname, out)
    print('file saved')
    #except:
        #print('error saving file ' + fname)


dice = np.array(dice)
print(dice)
print(np.mean(dice))
print(np.mean(dice[dice > 0]))

print('end of code')

