import copy
import datetime
import math
import time

import cv2
import numpy as np
import license_locate
window_h = 20
window_w = 180
binary_global = []
def get_upper_and_lower_boundaries(list):
        upper = 0
        lower = 0
        if len(list) != 0:
            for i in range(len(list)):
                sum += list[i]
            avg = math.floor(sum / len(list))
            # 从中间往上遍历，遇到比均值小的，则是车牌细定位的上边界，并跳出循环
            flag = False
            for i in range(math.floor(len(list) / 2), 0, -1):
                if list[i] < avg:
                    upper = i + 1
                    flag = True
                if flag is False:
                    continue
                break

            # 从中间往下遍历，遇到比均值小的，则是车牌细定位的下边界，并跳出循环
            flag = False
            for i in range(math.floor(len(list) / 2), len(list)):
                if list[i] < avg:
                    lower = i - 1
                    flag = True
                if flag is False:
                    continue
                break
        return lower, upper

def get_horizontal_projection( shadow_image):
    h, w = shadow_image.shape[:2]
    hor_list = []
    for x in range(h):
        pixel_sum = 0
        for y in range(w):
            pixel_sum += shadow_image[x][y]
        hor_list.append(pixel_sum)
    return hor_list

if __name__ == '__main__':
    i = 27
    instance = license_locate.Locate()
    #for i in range(22, 97):
    begin = time.time()
    filename = 'picture_for_train/'+str(i)+'.jpg'
    #filename = filename + str(i) + ".jpg"
    image = cv2.imread(filename)
        #instance.showPicture(image)
    image = cv2.resize(image, (640, 480))
        #instance.showPicture(image)
    sobel_image,sobel_image_y, diff, binary = instance.preprocess_image(filename)
    # cv2.imshow("image", sobel_image)
    # cv2.waitKey(0)
    # cv2.imshow("image", diff)
    # cv2.waitKey(0)
    # cv2.imshow("image", binary)
    # cv2.waitKey(0)
    integral_array, candiate_list = instance.rough_locate(sobel_image, binary, diff)
    print(len(candiate_list))
    # for each in candiate_list:
    #     instance.showLine(each[0],each[1],binary,1,"")
    # cv2.imshow("image", binary)
    # cv2.waitKey(0)
    result, position = instance.detail_locate_and_confirm(candiate_list, sobel_image, sobel_image_y,integral_array)
    x1 = position[0]
    x2 = position[1]
    y1 = position[2]
    y2 = position[3]
    ExpendH = math.floor((x2 - x1) / 2)
    image_need = image[x1:x2, y1: y2]

    end = time.time()
    print("时间:"+str(end - begin)+"s")
    image_need = cv2.resize(image_need,(210,40))
    cv2.imwrite('picture_for_tilt/'+str(i)+'.jpg',image_need)
    # cv2.imshow("image", image_need)
    # cv2.waitKey(0)
