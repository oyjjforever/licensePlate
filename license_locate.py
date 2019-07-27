import datetime
import math
from collections import Counter

import cv2
import numpy as np

#  确定最小搜索窗口为180*30，后续进行细定位
window_h = 20
window_w = 180


class Locate:
    # 预处理图片，以获取灰度跳变图
    def preprocess_image(self, file_name):
        # 加权平均法 + cv2 进行灰度化
        image = cv2.imread(file_name)
        gray_image = self.get_gray_image_by_weight_avg(image)

        # 进行Sobel的垂直边缘检测
        x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
        scale_abs_x = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        sobel_image = scale_abs_x

        # 二值化处理和中值滤波平滑
        diff = np.array((640, 480), np.uint8)
        gray_image = cv2.resize(gray_image, (640, 480))
        sobel_image = cv2.resize(sobel_image, (640, 480))  # 调整图片尺寸，以便后期处理
        diff = cv2.absdiff(gray_image, sobel_image)  # 做灰度图像和经过sobel边缘检测后的图片的水平差分，以去除背景影响
        avg = self.get_pixel_avg(diff)
        binary = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,
                                       math.floor(avg / 10))
        # cv2.imshow("binary", binary)
        # cv2.waitKey(0)
        return sobel_image, diff, binary

    def rough_locate(self, sobel_image, binary_image, diff):
        h, w = binary_image.shape[:2]  # w = 640,h = 480
        # 获取到二值化图像后，获取灰度跳变数组
        image_jump_array = self.get_2d_gray_jump(binary_image)

        # 通过灰度跳变数组获取灰度跳变的积分图，用于找到灰度跳变点最多的区域
        integral_array = self.get_gray_scale_jump_integral(h, w, image_jump_array)

        # 通过灰度跳变积分图，进行窗口搜索，粗定位车牌位置,
        # 以上得到粗定位的车牌候选区域，之后进行细定位车牌位置

        candiate_list = self.rough_search_by_window(h, w, integral_array, sobel_image)
        return integral_array, candiate_list

    def detail_locate_and_confirm(self, candiate_list, sobel_image, integral_array):
        # position : 粗定位的车牌左上角坐标
        # 第一步扩展车牌的上下区域获得最佳上下边界
        # 具体为获取扩展区域的水平投影并获取其平均值，从水平中央开始向两侧扫描，发现小于均值的则是上下边界
        # 第二步扩展车牌的左右区域获得最佳左右边界
        # 扩展后通过数据结构的最大子段和获取左右边界，因为在灰度图像中，白色为255，黑色为0

        candiate_list_2 = []  # 用作第二轮筛选
        for i in range(len(candiate_list)):

            flag = True  # 标志表示步骤正常运行 ,标志要放在循环内，以便每次刷新
            position = candiate_list[i]
            h, w = sobel_image.shape[:2]  # w = 640,h = 480
            # print(position)
            step_1 = sobel_image[position[0] - window_h:position[0] + 2 * window_h, position[1]:position[1] + window_w]
            # cv2.imshow("step_1", step_1)
            # cv2.waitKey(0)
            if position[0] - window_h < 0:
                step_1 = sobel_image[0:position[0] + 2 * window_h, position[1]:position[1] + window_w]
            elif position[0] + 2 * window_h > h:
                step_1 = sobel_image[position[0] - window_h:h, position[1]:position[1] + window_w]
            # 获取水平投影数组
            hor_list = self.get_horizontal_projection(step_1)

            # 通过水平投影数组获取车牌的重定位的上下边界
            upper, lower = self.detail_position_the_upper_and_lower_boundaries(hor_list)
            if upper != lower:
                step_1 = step_1[lower:upper, 0:w]
                # cv2.imshow("step_1", step_1)
                # cv2.waitKey(0)
                flag = True
            else:
                flag = False
            #  ##########上下边界定位结束########################

            #  ##########下面开始进行左右边界的定位###################
            if flag is True:  # 如果第一步上下边界正常运行
                image_step_1 = step_1
                h_1, w_1 = image_step_1.shape[:2]
                image_left = position[1] - math.floor(w_1 / 2)
                image_right = position[1] + window_w + math.floor(w_1 / 2)
                if image_left < 0:
                    flag = False
                elif image_right > 640:
                    flag = False

            if flag is True:
                image_step_2 = sobel_image[position[0] - window_h + lower: position[0] - window_h + upper,
                               image_left:image_right]

                ver_list = self.get_vertical_projection(image_step_2)
                sum = 0
                for i in range(len(ver_list)):
                    sum += ver_list[i]
                avg = math.floor(sum / len(ver_list))
                # 求均值后，将ver_list每个值都减去avg，然后用最大字段和来获取左右边界，在灰度图中，白色为255，黑色为0
                for i in range(len(ver_list)):
                    ver_list[i] = ver_list[i] - avg
                left, right = self.max_sequence(ver_list)
                if left == 0 and right == 0:
                    flag = False

            if flag is True:
                h, w = image_step_2.shape[:2]
                image_step_2 = image_step_2[0:h, left:right]

                # cv2.imshow("image_step_2", image_step_2)
                # cv2.waitKey(0)
                x1 = position[0] - window_h + lower
                x2 = position[0] - window_h + upper
                y1 = image_left + left
                y2 = image_left + right
                # print("x2 - x1 = " + str(x2 - x1))
                # print("y2 - y1 = " + str(y2 - y1))

                # 筛选掉边缘区域
                if x1 < 100 or y1 < 100 or x2 > 480 or y2 > 640:
                    flag = False

                # 筛选车牌长宽比
                ratio = (y2 - y1) / (x2 - x1)
                # print(ratio)
                if ratio > 12 or ratio < 5:
                    flag = False

                if flag is True:
                    area_gray_jump = int(integral_array[x2][y2]) + int(
                        integral_array[x1][y1]) - int(integral_array[x2][y1]) - int(
                        integral_array[x1][y2])
                    if area_gray_jump > math.floor(window_w * window_h / 8):  # 灰度跳变数要大于搜索窗口面积的八分之一
                        if len(candiate_list_2) != 0:
                            repeated_index = len(candiate_list_2) - 1
                            if not (abs(x1 - candiate_list_2[repeated_index][0]) <= 5 or abs(x2 - candiate_list_2[repeated_index][0]) <= 5 or abs(y1 - candiate_list_2[repeated_index][2]) <= 5 or abs(y2 - candiate_list_2[repeated_index][3]) <= 5):
                                candiate_list_2.append((x1, x2, y1, y2))
                        else:
                            candiate_list_2.append((x1, x2, y1, y2))
                        image_need = sobel_image[x1:x2, y1:y2]
                        # cv2.imshow("image_need", image_need)
                        # cv2.waitKey(0)
        # 融合候选区域中坐标相近的位置


        # print(candiate_list_2)
        # print(Counter(candiate_list_2).most_common(3))
        result_locate = Counter(candiate_list_2).most_common(5)
        max_ratio = 0  # 不能起名为关键字，会冲突
        max_index = 0
        max_width = 0
        for i in range(len(result_locate)):  #

            x1 = result_locate[i][0][0]
            x2 = result_locate[i][0][1]
            y1 = result_locate[i][0][2]
            y2 = result_locate[i][0][3]
            w = y2 - y1
            image = sobel_image[x1:x2,y1:y2]
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            ratio = math.floor((y2 - y1) / (x2 - x1))
            print(ratio)
            if ratio >= max_ratio:
                    max_ratio = ratio
                    max_index = i
                    max_width = x2 - x1

            print(max_ratio)
        # print(result_locate[0][1]) 出现最多的数出现的次数 [((238, 255, 202, 362), 36)]
        result_x1 = result_locate[max_index][0][0]
        result_x2 = result_locate[max_index][0][1]
        result_y1 = result_locate[max_index][0][2]
        result_y2 = result_locate[max_index][0][3]
        position = (result_x1, result_x2, result_y1, result_y2)
        result = sobel_image[result_x1:result_x2, result_y1:result_y2]
        return result, position

    def rough_search_by_window(self, h, w, integral_array, ori_image):
        max_gray_jump = 0
        candiate_list = []
        max_locate = (0, 0)
        for x in range(0, h - window_h, 5):
            for y in range(0, w - window_w, 5):
                # (x4,y4) + (x1,y1) - (x2,y2) - (x3,y3)
                area_jump_level = 0
                if ori_image[x + math.floor(window_h / 2)][y + math.floor(window_w / 2)] > 127:
                    area_jump_level = int(integral_array[x + window_h][y + window_w]) + int(
                        integral_array[x][y]) - int(integral_array[x + window_h][y]) - int(
                        integral_array[x][y + window_w])
                if area_jump_level > (window_h * window_w) / 8 and 100 < x < 480 - 100 and 100 < y < 640 - 100:
                    candiate_list.append((x, y))
                    # print(area_jump_level)
        # print(candiate_list_1)
        candiate_list.reverse()
        return candiate_list

    def get_gray_image_by_weight_avg(self, ori_image):
        h, w = ori_image.shape[:2]
        gray_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                # Y = 0．3R + 0．59G + 0．11B
                # 通过cv格式打开的图片，像素格式为 BGR
                gray_img[i, j] = 0.3 * ori_image[i, j][2] + 0.11 * ori_image[i, j][0] + 0.59 * ori_image[i, j][1]
        return gray_img

    def get_pixel_avg(self, image):
        sum = 0
        h, w = image.shape[:2]
        for x in range(h):
            for y in range(w):
                sum += image[x][y]
        avg = sum / (h * w)
        return avg

    def get_character_density(self, image):  # 计算图片的字符密度
        sum = 0
        h, w = image.shape[:2]
        for x in range(h):
            for y in range(w):
                sum += image[x][y]
        avg = sum / (h * w)
        return sum

    # 获取图像灰度跳变点的积分图
    def get_2d_gray_jump(self, image):
        jump_list_2d = np.zeros((480, 640), np.uint8)
        h, w = image.shape[:2]
        for x in range(h):
            for y in range(w):
                if abs(int(image[x][y]) - image[x][y - 1]) > 230:  # 将数值调低后，就能找到亮度较低的车牌图像的定位
                    jump_list_2d[x][y] = 1
                else:
                    jump_list_2d[x][y] = 0
        # np.set_printoptions(threshold=1e6)
        # print(jump_list_2d)
        return jump_list_2d

    # 获取灰度跳变的积分图
    def get_gray_scale_jump_integral(self, h, w, jump_array):
        h, w = jump_array.shape[:2]  # 把图片2像素的行数，列数以及通道数返回给rows，cols，channels
        sum = np.zeros((h + 1, w + 1), dtype=np.float32)  # 创建指定大小的数组，数组元素以 0 来填充：
        image_integral = cv2.integral(jump_array, sum, cv2.CV_32SC1)  # 计算积分图,输出是sum
        gray_jump_integral = np.zeros((h + 1, w + 1), dtype=np.uint16)
        cv2.normalize(image_integral, gray_jump_integral, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16UC1)  # 归一化处理
        np.set_printoptions(threshold=1e6)
        # print(gray_jump_integral)
        return gray_jump_integral

    def get_horizontal_projection(self, shadow_image):
        h, w = shadow_image.shape[:2]
        hor_list = []
        for x in range(h):
            pixel_sum = 0
            for y in range(w):
                pixel_sum += shadow_image[x][y]
            hor_list.append(pixel_sum)
        return hor_list

    def get_vertical_projection(self, shadow_image):
        h, w = shadow_image.shape[:2]
        ver_list = []
        for x in range(w):
            pixel_sum = 0
            for y in range(h):
                pixel_sum += shadow_image[y][x]
            ver_list.append(pixel_sum)
        return ver_list

    def detail_position_the_upper_and_lower_boundaries(self, list):
        sum = 0
        upper = 0
        lower = 0

        for i in range(len(list)):
            sum += list[i]
        avg = math.floor(sum / len(list))
        # 从中间往上遍历，遇到比均值小的，则是车牌细定位的上边界，并跳出循环
        flag = False
        for i in range(math.floor(len(list) / 2), 0, -1):
            if list[i] < avg:
                upper = i
                flag = True
            if flag is False:
                continue
            break

        # 从中间往下遍历，遇到比均值小的，则是车牌细定位的下边界，并跳出循环
        flag = False
        for i in range(math.floor(len(list) / 2), len(list)):
            if list[i] < avg:
                lower = i
                flag = True
            if flag is False:
                continue
            break
        return lower, upper

    def max_sequence(self, array):
        sum = 0
        max = 0
        bestI = 0
        bestJ = 0
        i = 0
        for j in range(len(array)):
            sum += array[j]
            if sum > 0:
                sum += array[j]
            else:
                sum = array[j]
                i = j
            if sum > max:
                max = sum
                bestI = i
                bestJ = j

        l = bestI
        r = bestJ


        return l, r

    def license_locate(self, filename):  # 这个是获取垂直跳变点图的位置
        sobel_image, diff, binary = self.preprocess_image(filename)
        integral_array, candiate_list = self.rough_locate(sobel_image, binary, diff)
        result, position = self.detail_locate_and_confirm(candiate_list, sobel_image, integral_array)
        return result


if __name__ == '__main__':
    begin = datetime.datetime.now()
    instance = Locate()

    for i in range(9, 58):
        filename = 'H:/PyCharm/workspace/licensePlate/picture_for_train/'
        filename = filename + str(i) + ".jpg"
        print(filename)
        # ###############行扫描法 #################
        # image = cv2.imread(filename)
        # gray_img = instance.get_gray_image_by_weight_avg(image)
        # out = instance.grey_scale(gray_img)
        # cv2.imshow("out", out)
        # result = instance.mark_row_area(out)
        #
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        ##################################################

        result = instance.license_locate(filename)
        # # cv2.imshow(filename[-8:], result)
        # # cv2.waitKey(0)
        # h, w = result.shape[:2]
        #
        # result = cv2.resize(result, (200, 40))
        # cv2.imwrite('D:/Car_Identify/picture_locate/' + str(i) + ".jpg", result)

    end = datetime.datetime.now()
    print(str((end - begin).seconds) + "s")
