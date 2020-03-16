import numpy as np
import cv2

def show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
image1 = cv2.imread('2.png',0)
image2 = cv2.imread('1.png',0)
show('1',image1)
show('2',image2)
sift = cv2.xfeatures2d.SIFT_create()
(kps1,features1) = sift.detectAndCompute(image1,None)
(kps2,features2) = sift.detectAndCompute(image2,None)
kps1 = [kp.pt for kp in kps1]
kps2 = [kp.pt for  kp in kps2]
kps1 = np.float32(kps1)
kps2 = np.float32(kps2)
matcher = cv2.BFMatcher()
raw_match = matcher.knnMatch(features1,features2,k=2)
print(len(raw_match))
matchs = []
for m in raw_match:
    if len(m)==2 and m[0].distance<0.75*m[1].distance:
        matchs.append((m[0].trainIdx,m[0].queryIdx))
if len(matchs)>4:
    pst1 = np.float32([kps1[i] for (_,i) in matchs])
    pst2 = np.float32([kps2[i] for (i,_) in matchs])

    H,_ = cv2.findHomography(pst1,pst2,cv2.RANSAC,4.0)
result = cv2.warpPerspective(image1,H,(image1.shape[1]+image2.shape[1],image1.shape[0]))
show('3', result)
result[0:image2.shape[0]:,0:image2.shape[1]] = image2
show('result',result)