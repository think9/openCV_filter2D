import cv2
import numpy as np

#Read image with bgr and grayscale
src  = cv2.imread('./school.png')
src_gray  = cv2.imread('./school.png', 0)

#1 Mean Filter
mean = [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]
mean = np.array(mean) / 9
mean_result = cv2.filter2D(src, cv2.CV_32F, mean)
mean_result = cv2.normalize(mean_result, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

#2 Weighted Mean Filter
wMean = [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]
wMean = np.array(wMean) / 16
wMean_result = cv2.filter2D(src, cv2.CV_32F, wMean)
wMean_result = cv2.normalize(wMean_result, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

#3 Sobel Filter
sobelX = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
sobelX = np.array(sobelX)
sobelX_result = cv2.filter2D(src_gray, cv2.CV_32F, sobelX)

sobelY = [[-1, -2, -1],
          [0, 0, 0,],
          [1, 2, 1]]
sobelY = np.array(sobelY)
sobelY_result = cv2.filter2D(src_gray, cv2.CV_32F, sobelY)

mag = cv2.magnitude(sobelX_result, sobelY_result)
sobel_result = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

#4 Laplacian Filter
def getSign(number):
    if number >= 0:
        return 1
    else:
        return -1

def zeroCrossing(mat):
    height, width = mat.shape
    result = np.zeros(mat.shape, dtype = np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighbors = [mat[y-1,x], mat[y+1,x], mat[y,x-1], mat[y,x+1],
                       mat[y-1,x-1], mat[y-1,x+1], mat[y+1,x-1], mat[y+1,x+1]]                       
            minimum = min(neighbors)
            if getSign(mat[y,x]) != getSign(minimum):
                result[y, x] = 255
    return result

laplacian = [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
laplacian = np.array(laplacian)

laplacian2 = [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]]
laplacian2 = np.array(laplacian2)

blur = cv2.GaussianBlur(src_gray, ksize = (7, 7), sigmaX = 0.0)
laplacian_apply = cv2.filter2D(blur, cv2.CV_32F, laplacian)
laplacian_zeroCrossing = zeroCrossing(laplacian_apply)
laplacian_apply = cv2.normalize(laplacian_apply, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

laplacian_result = cv2.filter2D(src_gray, cv2.CV_32F, laplacian2)
laplacian_result = cv2.convertScaleAbs(laplacian_result)
laplacian_result = cv2.normalize(laplacian_result, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)


#Show result
##cv2.imshow('Original', src)
##cv2.imshow('Mean Filter',  mean_result)
##cv2.imshow('Weighted Mean Filter',  wMean_result)
##cv2.imshow('Original with Grayscale', src_gray)
##cv2.imshow('Sobel Filter',  sobel_result)
cv2.imshow('Apply Laplacian Filter', laplacian_apply)
cv2.imshow('Laplacian Zero Crossing', laplacian_zeroCrossing)
cv2.imshow('Enhanced Image with Laplacian Filter', laplacian_result)

###Save the result
##cv2.imwrite('./1_Original.png', src)
##cv2.imwrite('./2_Mean Filter.png',  mean_result)
##cv2.imwrite('./3_Weighted Mean Filter.png',  wMean_result)
##cv2.imwrite('./4_Original with Grayscale.png', src_gray)
##cv2.imwrite('./5_Sobel Filter.png',  sobel_result)
##cv2.imwrite('./6_Apply Laplacian Filter.png', laplacian_apply)
cv2.imwrite('./7_2_Laplacian Zero Crossing.png', laplacian_zeroCrossing)
##cv2.imwrite('./8_Enhanced Image with Laplacian Filter.png', laplacian_result)

cv2.waitKey()    
cv2.destroyAllWindows()
