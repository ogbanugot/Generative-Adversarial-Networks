#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:25:16 2018

@author: ogban ugot
"""
# =============================================================================

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


class ORBmatcher:
    
    def __init__(self, template, image, min_match_count):
        self.template = template
        self.image = image
        self.min_match_count = min_match_count
    
    def objectMatcher(self, i,t, img2, matches, kp1, img1, kp2, des2):
        img1= img1
        img2 = img2
        good = []
        for m in matches:
            #write distance of close matches
            file = '/home/ugot/Documents/CASIA-FingerprintV5 /good match distance.txt'
            writeMatch(t,matches,file)
            if m.distance < 60:
                good.append(m)
        print(len(good))
        #write the distance of the good features 
        file = '/home/ugot/Documents/CASIA-FingerprintV5 /close match metrics.txt'
        writeMatch(t,good,file)
        #write the number of good match metrics
        file = '/home/ugot/Documents/CASIA-FingerprintV5 /number of good match metrics.txt'
        writelenGood(t,len(good),file)
        
        if len(good) >= self.min_match_count:
            try: 
                src_pts = np.float32([ kp1[good[m].queryIdx].pt for m in range(len(good))]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[good[m].trainIdx].pt for m in range(len(good))]).reshape(-1,1,2)
            
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
            
                h,w, _ = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,(255,255,255),3, cv2.LINE_AA)
                
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)
                                        
                img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
                img3 = img3
                plt.imsave('/home/ugot/Documents/CASIA-FingerprintV5 /matches/match %d%d.png'% (i,t),img3, 'gray')

                
            except IndexError:
                print("out of index range")
                
        
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),self.min_match_count))

        
    def bestMatch(self, filename1, filename2):                
        templates = self.template # queryImage
        image = self.image # trainImage
        print(len(templates))
        print(len(image))
        # Initiate SIFT detector
        Matches = {}
        avgDistance= {}
        orb = cv2.ORB_create()
        for i in range(len(image)):
            img = image[i]
            kp2, des2 = orb.detectAndCompute(image[i],None)                             
            print("checking image %d, %s" % (i, filename2[i]))
            for t in range(len(templates)):
                print("matching againts template %d, %s" % (t, filename1[t]))
                try:                    
                    # find the keypoints and descriptors with SIFT
                    kp1, des1 = orb.detectAndCompute(templates[t],None)
                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                
                    # Match descriptors.
                    match = bf.match(des1, des2) 
                    if len(match) == 0:
                        print("no single match")
                    else:                    
                        print(len(match))                
                        # Sort them in the order of their distance.
                        match = sorted(match, key = lambda x:x.distance)
                        #general matching metrics
                        file = '/home/ugot/Documents/CASIA-FingerprintV5 /general match metrics.txt'
                        writeMatch(i,match,file)
                        Matches[t] = {'matches':match, 'kp':kp1, 'template':templates[t]}
                        #find the sum of the first matches
                        for m in range(len(match)):                        
                            summ = (match[m].distance)   
                            avg = summ/len(match)
                        avgDistance[t] = avg
                        
                        ky = min(avgDistance, key=avgDistance.get)
                        matches = Matches[ky]['matches']
                        kp = Matches[ky]['kp'] 
                        template = Matches[ky]['template']                        
                        ORBmatcher.objectMatcher(self,i, t, img, matches, kp, template, kp2, des2)
                except:
                    print("distance mismatch")
                    
#function to write matches to file on disk
def writeMatch(index, match, file):
    for m in match:
        my_file=open(file,"a")
        my_file.write('%d; %d;'  % (index, m.distance) +'\n')
        my_file.close()               

def writelenGood(index, lengood, file):
    my_file=open(file,"a")
    my_file.write('%d; %d;'  % (index, lengood) +'\n')
    my_file.close()               



#function to load images from disk        
def getimage(path):
    images = []
    filename = []
    ### Loop through all the images and save them to an array
    for file in os.listdir(path):
        pth =  (path+'/'+file)
        img = cv2.imread(pth)
        print(img)
        images.append(img)
        filename.append(file)      
    print ("found %d images and %d files" % (len(images), len(filename)))
    return (images, filename)

    
        