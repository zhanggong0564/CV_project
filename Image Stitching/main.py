import utils
import cv2


if __name__ == '__main__':
    imageA = cv2.imread('1.png')
    imageB = cv2.imread('2.png')
    stitcher_match = utils.keypoint_match()
    kpsA,featuresA,kpsB,featuresB = stitcher_match.get_keypoint(imageA,imageB)
    H = stitcher_match.match(kpsA,featuresA,kpsB,featuresB)
    image_transform_stitcher = utils.image_stitcher()
    result1 = image_transform_stitcher.image_transform(imageA,imageB,H)
    cv2.imshow('result1', result1)
    result2 = image_transform_stitcher.stitcher_imageAB(result1,imageA)
    cv2.imwrite('/home/zhanggong-study/CV_Project/Image Stitching/sample/result1.jpg',result1)
    cv2.imwrite('/home/zhanggong-study/CV_Project/Image Stitching/sample/result2.jpg', result2)
    cv2.imshow('imageA',imageA)
    cv2.imshow('imageB',imageB)
    cv2.imshow('result2',result2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()