import cv2
import numpy as np

class keypoint_match:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def get_keypoint(self,imageA,imageB):
        '''1，对两幅图像分别进行关键点检测'''
        gray_imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        gray_imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (kpsA,featuresA) = self.sift.detectAndCompute(gray_imageA,None)
        (kpsB,featuresB) = self.sift.detectAndCompute(gray_imageB,None)
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])
        return kpsA,featuresA,kpsB,featuresB
    def match(self,kpsA,featuresA,kpsB,featuresB):
        '''2. 匹配关键点'''
        matchs = self.matcher.knnMatch(featuresA,featuresB,k = 2)
        print(len(matchs))
        best_matchs = []
        for m in matchs:
            if len(m)==2 and m[0].distance<0.75*m[1].distance:
                best_matchs.append((m[0].trainIdx,m[0].queryIdx))
        print(len(best_matchs))
        if len(best_matchs)>4:
            posA = np.float32([kpsA[i] for (_,i) in best_matchs])
            posB = np.float32([kpsB[i] for (i,_) in best_matchs])
            H,_ = cv2.findHomography(posA,posB,cv2.RANSAC,4)
            return H
        else:
            return None


class image_stitcher:

    def __init__(self):
        pass
    def image_transform(self,imageA,imageB,H):
        '''将第二幅图像按照单应性关系进行变换 '''
        image_shape = (imageA.shape[1]+imageB.shape[1],imageA.shape[0])
        result = cv2.warpPerspective(imageA,H,image_shape)
        return result
    def stitcher_imageAB(self,result,imageA):
        '''拼接图片'''
        result[0:imageA.shape[0],0:imageA.shape[1]] = imageA
        return result




