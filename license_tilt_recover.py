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
def showLine(startPoint,filename):
    # image = cv2.imread(filename)
    # image = cv2.resize(image, (640, 480))
    color = 0  # 置为hei点
    image = copy.deepcopy(filename)
    for topLineElement in range(startPoint[0],startPoint[0]+180):
        image[startPoint[1]][topLineElement]=color
        image[startPoint[1]+20][topLineElement]=color
    for leftColumnElement in range(startPoint[1], startPoint[1] + 20):
        image[leftColumnElement][startPoint[0]]=color
        image[leftColumnElement][startPoint[0] + 180]=color

    cv2.imshow("image", image)
    cv2.waitKey(500)
if __name__ == '__main__':
    instance = license_locate.Locate()
    for i in range(10, 97):
        begin = time.time()
        filename = 'picture_for_train/'
        filename = filename + str(i) + ".jpg"
        image = cv2.imread(filename)
        #instance.showPicture(image)
        image = cv2.resize(image, (640, 480))
        #instance.showPicture(image)
        sobel_image, diff, binary = instance.preprocess_image(filename)
        integral_array, candiate_list = instance.rough_locate(sobel_image, binary, diff)
        print(len(candiate_list))
        binary_global = binary
        for each in candiate_list:
            showLine(each,binary)
        result, position = instance.detail_locate_and_confirm(candiate_list, sobel_image, integral_array)
        x1 = position[0]
        x2 = position[1]
        y1 = position[2]
        y2 = position[3]
        ExpendH = math.floor((x2 - x1) / 2)
        image_need = image[x1:x2, y1: y2]
        # image_need_correct = sobel_image[position[0] + lower - window_h - ExpendH:position[0] - window_h + upper + ExpendH,
        #                       image_left: image_right]
        # if position[1] - math.floor(3 * w_1 / 2) < 0:
        #     image_need_correct = image[
        #                          position[0] - window_h + lower - ExpendH: position[0] - window_h + upper + ExpendH,
        #                          left:right]
        # elif position[1] + window_w + math.floor(3 * w_1 / 2) > 640:
        #     image_need_correct = image[
        #                          position[0] - window_h + lower - ExpendH: position[0] - window_h + upper + ExpendH,
        #                          position[1] - math.floor(3 * w_1 / 2) + left:640]
        # print(image_left)
        # print(image_right)
        print(i)
        # h, w = image_need_correct.shape[:2]
        # min = 100
        # for i in range(-10,11):
        #     M = cv2.getRotationMatrix2D((w / 2, h / 2), i, 1)
        #     result = cv2.warpAffine(image_need_correct, M, (w, h))
        #     hor_list = get_horizontal_projection(result)
        #     if len(hor_list) < min:
        #         min = len(hor_list)
        # print(hor_list)
        end = time.time()
        print("时间:"+str(end - begin)+"s")
        image_need = cv2.resize(image_need,(210,40))
        cv2.imwrite('picture_for_tilt/'+str(i)+'.jpg',image_need)
        # cv2.imshow("image_need_correct", result)
        # cv2.waitKey(0)
