# coding=utf-8
import os
from multiprocessing import Pool

import cv2
import numpy as np
from gdalconst import *
from osgeo import gdal


def findAllFiles(root_dir, filter):
    """
    在指定目录查找指定类型文件

    :param root_dir: 查找目录
    :param filter: 文件类型
    :return: 路径、名称、文件全路径

    """

    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


def findAllFilesReverse(root_dir, filter):
    """
    在指定目录查找指定类型文件，返回倒序list

    :param root_dir: 查找目录
    :param filter: 文件类型
    :return: 路径、名称、文件全路径

    """

    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    paths.reverse()
    names.reverse()
    files.reverse()
    return paths, names, files


def isDirExist(path='output'):
    """
    判断指定目录是否存在，如果存在返回True，否则返回False并新建目录

    :param path: 指定目录
    :return: 判断结果

    """

    if not os.path.exists(path):
        os.mkdir(path)
        return False
    else:
        return True


def reverseRGB(img):
    """
    反转RGB波段顺序

    :param img: RGB波段影像
    :return: 波段顺序反转的波段影像

    """

    img2 = np.zeros(img.shape, np.uint8)
    img2[:, :, 0] = img[:, :, 2]
    img2[:, :, 1] = img[:, :, 1]
    img2[:, :, 2] = img[:, :, 0]
    return img2


def splitRGB(img):
    """
    用于对读取的RGB影像进行波段拆分，返回单独的每个波段

    :param img: RGB影像
    :return: 拆分后的R、G、B波段数据

    """

    band_r = img[:, :, 0]
    band_g = img[:, :, 1]
    band_b = img[:, :, 2]
    return band_r, band_g, band_b


def mergeRGB(band_r, band_g, band_b):
    """
    合并独立的R、G、B波段数据为一个彩色图像

    :param band_r: R波段
    :param band_g: G波段
    :param band_b: B波段
    :return: 彩色图像

    """

    h = min(band_r.shape[0], band_g.shape[0], band_b.shape[0])
    w = min(band_r.shape[1], band_g.shape[1], band_b.shape[1])
    img = np.zeros([h, w, 3], np.uint8)
    img[:, :, 0] = band_r[:h, :w]
    img[:, :, 1] = band_g[:h, :w]
    img[:, :, 2] = band_b[:h, :w]
    return img


def getSurfKps(img, hessianTh=1500):
    """
    获取SURF特征点和描述子

    :param img: 读取的输入影像
    :param hessianTh: 海塞矩阵阈值，默认为1500
    :return: 特征点和对应的描述子

    """

    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=hessianTh)
    kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
    return kp, des


def getSiftKps(img, numKps=2000):
    """
    获取SIFT特征点和描述子

    :param img: 读取的输入影像
    :param numKps: 期望提取的特征点个数，默认2000
    :return: 特征点和对应的描述子

    """

    sift = cv2.xfeatures2d_SIFT.create(nfeatures=numKps)
    kp, des = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img, None)
    return kp, des


def getOrbKps(img, numKps=2000):
    """
    获取ORB特征点和描述子

    :param img: 读取的输入影像
    :param numKps: 期望提取的特征点个数，默认2000
    :return: 特征点和对应的描述子

    """

    orb = cv2.ORB_create(nfeatures=numKps)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def drawKeypoints(img, kps, color=[0, 0, 255], rad=3, thickness=1):
    """
    在影像上绘制特征点

    :param img: 待绘制的影像
    :param kps: 待绘制的特征点列表
    :param color: 特征点颜色，默认红色，-1表示随机
    :param rad: 特征点大小，默认为3
    :param thickness: 特征点轮廓粗细，默认为1
    :return: 带有特征点的影像

    """

    if img.shape.__len__() == 2:
        img_pro = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_pro = img.copy()
    if color[0] == -1 and color[1] == -1 and color[2] == -1:
        if type(kps[0]) is cv2.KeyPoint:
            for point in kps:
                pt = (int(point.pt[0]), int(point.pt[1]))
                color[0] = getRandomNum(0, 255)
                color[1] = getRandomNum(0, 255)
                color[2] = getRandomNum(0, 255)
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
        else:
            for point in kps:
                pt = (int(point[0]), int(point[1]))
                color[0] = getRandomNum(0, 255)
                color[1] = getRandomNum(0, 255)
                color[2] = getRandomNum(0, 255)
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
    else:
        if type(kps[0]) is cv2.KeyPoint:
            for point in kps:
                pt = (int(point.pt[0]), int(point.pt[1]))
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
        else:
            for point in kps:
                pt = (int(point[0]), int(point[1]))
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
    return img_pro


def getRandomNum(start=0, end=100):
    """
    获取指定范围内的随机整数，默认范围为0-100

    :param start: 最小值
    :param end: 最大值
    :return: 随机数

    """

    return np.random.randint(start, end + 1)


def logTransform(img, v=200, c=256):
    """
    影像的灰度对数拉伸变换，默认支持8bit灰度

    :param img: 待变换影像
    :param v: 变换系数v，越大效果越明显
    :param c: 灰度量化级数的最大值，默认为8bit(256)
    :return: 拉伸后的影像

    """

    img_normalize = img * 1.0 / c
    log_res = c * (np.log(1 + v * img_normalize) / np.log(v + 1))
    img_new = np.uint8(log_res)
    return img_new


def flannMatch(kp1, des1, kp2, des2):
    """
    基于FLANN算法的匹配

    :param kp1: 特征点列表1
    :param des1: 特征点描述列表1
    :param kp2: 特征点列表2
    :param des2: 特征点描述列表2
    :return: 匹配的特征点对

    """

    good_matches = []
    good_kps1 = []
    good_kps2 = []

    print("kp1 num:" + len(kp1).__str__() + "," + "kp2 num:" + len(kp2).__str__())

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    if type(kp1[0]) is cv2.KeyPoint:
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.5 * n.distance:
                good_matches.append(matches[i])
                good_kps1.append([kp1[matches[i][0].queryIdx].pt[0], kp1[matches[i][0].queryIdx].pt[1]])
                good_kps2.append([kp2[matches[i][0].trainIdx].pt[0], kp2[matches[i][0].trainIdx].pt[1]])
    else:
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.5 * n.distance:
                good_matches.append(matches[i])
                good_kps1.append([kp1[matches[i][0].queryIdx][0], kp1[matches[i][0].queryIdx][1]])
                good_kps2.append([kp2[matches[i][0].trainIdx][0], kp2[matches[i][0].trainIdx][1]])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        return good_kps1, good_kps2
    else:
        print("good matches:" + good_matches.__len__().__str__())
        return good_kps1, good_kps2


def bfMatch(kp1, des1, kp2, des2, disTh=15.0):
    """
    基于BF算法的匹配

    :param kp1: 特征点列表1
    :param des1: 特征点描述列表1
    :param kp2: 特征点列表2
    :param des2: 特征点描述列表2
    :return: 匹配的特征点对

    """

    good_kps1 = []
    good_kps2 = []
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    if matches.__len__() == 0:
        return good_kps1, good_kps2
    else:
        min_dis = 10000
        for item in matches:
            dis = item.distance
            if dis < min_dis:
                min_dis = dis

        g_matches = []
        for match in matches:
            if match.distance <= max(1.1 * min_dis, disTh):
                g_matches.append(match)

        print("matches:" + g_matches.__len__().__str__())

        if type(kp1[0]) is cv2.KeyPoint:
            for i in range(g_matches.__len__()):
                good_kps1.append([kp1[g_matches[i].queryIdx].pt[0], kp1[g_matches[i].queryIdx].pt[1]])
                good_kps2.append([kp2[g_matches[i].trainIdx].pt[0], kp2[g_matches[i].trainIdx].pt[1]])
        else:
            for i in range(g_matches.__len__()):
                good_kps1.append([kp1[g_matches[i].queryIdx][0], kp1[g_matches[i].queryIdx][1]])
                good_kps2.append([kp2[g_matches[i].trainIdx][0], kp2[g_matches[i].trainIdx][1]])
        return good_kps1, good_kps2


def drawMatches(img1, kps1, img2, kps2, color=[0, 0, 255], rad=5, thickness=1):
    """
    用于绘制两幅影像间的匹配点对

    :param img1: 影像1
    :param kps1: 影像1上匹配的特征点
    :param img2: 影像2
    :param kps2: 影像2上匹配的特征点
    :param color: 特征点及连线的颜色，默认为红色，-1表示随机
    :param rad: 特征点大小，默认为5
    :param thickness: 匹配连线的宽度，默认为1
    :return: 绘制好的特征点匹配影像

    """

    if img1.shape.__len__() == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.shape.__len__() == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
    img_out[:img1.shape[0], :img1.shape[1], :] = img1
    img_out[:img2.shape[0], img1.shape[1]:, :] = img2

    if color[0] == -1 and color[1] == -1 and color[2] == -1:
        if type(kps1[0]) == cv2.KeyPoint:
            for kp1, kp2 in zip(kps1, kps2):
                pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
                pt2 = (int(kp2.pt[0] + img1.shape[1]), int(kp2.pt[1]))
                color[0] = getRandomNum(0, 255)
                color[1] = getRandomNum(0, 255)
                color[2] = getRandomNum(0, 255)
                cv2.circle(img_out, pt1, rad, color, thickness, cv2.LINE_AA)
                cv2.circle(img_out, pt2, rad, color, thickness, cv2.LINE_AA)
                cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
        else:
            for kp1, kp2 in zip(kps1, kps2):
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0] + img1.shape[1]), int(kp2[1]))
                color[0] = getRandomNum(0, 255)
                color[1] = getRandomNum(0, 255)
                color[2] = getRandomNum(0, 255)
                cv2.circle(img_out, pt1, rad, color, thickness, cv2.LINE_AA)
                cv2.circle(img_out, pt2, rad, color, thickness, cv2.LINE_AA)
                cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
    else:
        if type(kps1[0]) == cv2.KeyPoint:
            for kp1, kp2 in zip(kps1, kps2):
                pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
                pt2 = (int(kp2.pt[0] + img1.shape[1]), int(kp2.pt[1]))
                cv2.circle(img_out, pt1, rad, color, thickness, cv2.LINE_AA)
                cv2.circle(img_out, pt2, rad, color, thickness, cv2.LINE_AA)
                cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
        else:
            for kp1, kp2 in zip(kps1, kps2):
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0] + img1.shape[1]), int(kp2[1]))
                cv2.circle(img_out, pt1, rad, color, thickness, cv2.LINE_AA)
                cv2.circle(img_out, pt2, rad, color, thickness, cv2.LINE_AA)
                cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_out


def loadSingleBandtoMem(img_path):
    """
    以只读方式打开遥感影像，用于加载单波段影像到内存中

    :param img_path: 需要加载的影像路径
    :return: 影像数据

    """

    dataset = gdal.Open(img_path, GA_ReadOnly)
    band_1 = dataset.GetRasterBand(1)
    data = band_1.ReadAsArray(0, 0, band_1.XSize, band_1.YSize)
    return data


def getBlockRange(img, row=2, col=2):
    """
    将较大影像分块，返回每块的坐标索引

    :param img: 原始影像
    :param row: 待分的行数，默认为2
    :param col: 待分的列数，默认为2
    :return: 包含每块坐标索引的列表

    """

    img_h = img.shape[1]
    img_w = img.shape[0]
    # print img_h, img_w
    w_per_block = img_w / row
    h_per_block = img_h / col
    # print h_per_block, w_per_block
    blocks = []
    for i in range(row):
        for j in range(col):
            w = i * w_per_block
            h = j * h_per_block
            rb_w = w + w_per_block
            rb_h = h + h_per_block
            # print w, '-', rb_w, h, '-', rb_h
            blocks.append([w, rb_w, h, rb_h])
    return blocks


def findAffine(kps1, kps2):
    """
    基于匹配的特征点对求解仿射关系

    :param kps1: 匹配的特征点1
    :param kps2: 匹配的特征点2
    :return: 仿射矩阵

    """

    if kps1.__len__() < 3 or kps2.__len__() < 3:
        affine = None
    else:
        if type(kps1[0]) is cv2.KeyPoint:
            tmp_kps1 = cvtCvKeypointToNormal(kps1)
            tmp_kps2 = cvtCvKeypointToNormal(kps2)
            affine, mask = cv2.estimateAffine2D(np.array(tmp_kps1), np.array(tmp_kps2))
        else:
            affine, mask = cv2.estimateAffine2D(np.array(kps1), np.array(kps2))
    return affine


def findHomography(kps1, kps2):
    """
    基于匹配的特征点对求解单应变换关系

    :param kps1: 匹配的特征点1
    :param kps2: 匹配的特征点2
    :return: 单应矩阵

    """

    if kps1.__len__() < 5 or kps2.__len__() < 5:
        homo = None
    else:
        if type(kps1[0]) is cv2.KeyPoint:
            tmp_kps1 = cvtCvKeypointToNormal(kps1)
            tmp_kps2 = cvtCvKeypointToNormal(kps2)
            homo, mask = cv2.findHomography(np.array(tmp_kps1), np.array(tmp_kps2))
        else:
            homo, mask = cv2.findHomography(np.array(kps1), np.array(kps2))
    return homo


def resampleImg(img, trans):
    """
    基于获得的变换关系对影像进行重采

    :param img: 待重采影像
    :param trans: 变换矩阵，仿射or单应
    :return: 重采后的影像

    """

    if trans is None:
        return img
    if trans.shape[0] == 2:
        resampled_img = cv2.warpAffine(img, trans,
                                       (img.shape[1],
                                        img.shape[0]))
    elif trans.shape[0] == 3:
        resampled_img = cv2.warpPerspective(img, trans,
                                            (img.shape[1],
                                             img.shape[0]))
    else:
        resampled_img = img
    return resampled_img


def getMinAndMaxGrayValue(img, ratio=0.0):
    """
    获得影像中的灰度最大和最小值，目前只支持8bit影像

    :param img: 待统计的影像
    :param ratio: 统计时首尾忽略像素的百分比占比，默认为0
    :return: 灰度最小、最大值

    """

    bins = np.arange(256)
    hist, bins = np.histogram(img, bins)
    total_pixels = img.shape[0] * img.shape[1]
    min_index = int(ratio * total_pixels)
    max_index = int((1 - ratio) * total_pixels)
    min_gray = 0
    max_gray = 0
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > min_index:
            min_gray = i
            break
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > max_index:
            max_gray = i
            break
    return min_gray, max_gray


def linearStretch(img, new_min, new_max, ratio=0.0):
    """
    灰度线性拉伸

    :param img: 待拉伸影像
    :param new_min: 期望最小值
    :param new_max: 期望最大值
    :param ratio: 拉伸百分比，默认为0，若为0.02则为2%线性拉伸
    :return: 拉伸后的影像

    """

    old_min, old_max = getMinAndMaxGrayValue(img, ratio)
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    print("=>linear stretch:")
    print('old min = %d,old max = %d new min = %d,new max = %d' % (old_min, old_max, new_min, new_max))
    img3 = np.uint8((new_max - new_min) / (old_max - old_min) * (img2 - old_min) + new_min)
    return img3


def resampleToBase(img_base, img_warp, flag='affine'):
    """
    依据获得的变换关系，将影像重采到基准影像上

    :param img_base: 基准影像路径
    :param img_warp: 待重采影像路径
    :param flag: 变换模型，'affine'(仿射)或'homo'(单应)
    :return: 重采后的影像

    """

    img1 = cv2.imread(img_base)
    img2 = cv2.imread(img_warp)
    kp1, des1 = getSiftKps(img1)
    kp2, des2 = getSiftKps(img2)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    if flag == 'affine':
        trans = findAffine(good_kp2, good_kp1)
    elif flag == 'homo':
        trans = findHomography(good_kp2, good_kp1)
    else:
        trans = None
    resample_img = resampleImg(img2, trans)
    return resample_img


def siftFlannMatch(img1, img2, numKps=2000):
    """
    包装的函数，直接用于sift匹配，方便使用

    :param img1: 输入影像1
    :param img2: 输入影像2
    :param numKps: 每张影像上期望提取的特征点数量，默认为2000
    :return: 匹配好的特征点列表

    """

    kp1, des1 = getSiftKps(img1, numKps=numKps)
    kp2, des2 = getSiftKps(img2, numKps=numKps)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2


def surfFlannMatch(img1, img2, thHessian=1500):
    """
    包装的函数，直接用于surf匹配，方便使用

    :param img1: 输入影像1
    :param img2: 输入影像2
    :param thHessian: 海塞矩阵阈值，默认为1500
    :return: 匹配好的特征点列表

    """

    kp1, des1 = getSurfKps(img1, hessianTh=thHessian)
    kp2, des2 = getSurfKps(img2, hessianTh=thHessian)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2


def orbBFMatch(img1, img2, numKps=2000, disTh=15.0):
    """
    包装的函数，直接用于orb匹配，方便使用

    :param img1: 输入影像1
    :param img2: 输入影像2
    :param numKps: 每张影像上期望提取的特征点数量，默认为2000
    :return: 匹配好的特征点列表

    """

    kp1, des1 = getOrbKps(img1, numKps=numKps)
    kp2, des2 = getOrbKps(img2, numKps=numKps)
    good_kp1, good_kp2 = bfMatch(kp1, des1, kp2, des2, disTh=disTh)
    return good_kp1, good_kp2


def cvtCvKeypointToNormal(keypoints):
    """
    将OpenCV中KeyPoint类型的特征点转换成(x,y)格式的普通数值

    :param keypoints: KeyPoint类型的特征点列表
    :return: 转换后的普通特征点列表

    """

    cvt_kps = []
    for i in range(keypoints.__len__()):
        cvt_kps.append((keypoints[i].pt[0], keypoints[i].pt[1]))
    return cvt_kps


def saveKeyPoints(keypoints, savePath, seprator='\t'):
    """
    将特征点输出到文本文件中

    :param keypoints: 特征点列表
    :param savePath: 输出文件路径和文件名
    :param seprator: 可选参数，数据分隔符，默认为tab
    :return: 空

    """

    if keypoints.__len__() != 0:
        save_file = open(savePath, 'w+')
        save_file.write(keypoints.__len__().__str__() + "\n")
        if type(keypoints[0]) is cv2.KeyPoint:
            kps = cvtCvKeypointToNormal(keypoints)
            for i in range(kps.__len__()):
                save_file.write(kps[i][0].__str__() + seprator + kps[i][1].__str__() + "\n")
        else:
            for i in range(keypoints.__len__()):
                save_file.write(keypoints[i][0].__str__() + seprator + keypoints[i][1].__str__() + "\n")
        save_file.close()
    else:
        print "keypoint list is empty,please check."


def saveMatchPoints(keypoints1, keypoints2, savePath, seprator='\t'):
    """
    将匹配的特征点输出到文本文件中

    :param keypoints1: 特征点列表1
    :param keypoints2: 特征点列表2
    :param savePath: 输出文件路径和文件名
    :param seprator: 可选参数，数据分隔符，默认为tab
    :return: 空

    """

    if keypoints1.__len__() != 0 and keypoints2.__len__() != 0:
        save_file = open(savePath, 'w+')
        save_file.write(keypoints1.__len__().__str__() + "\n")
        if type(keypoints1[0]) is cv2.KeyPoint:
            kps1 = cvtCvKeypointToNormal(keypoints1)
            kps2 = cvtCvKeypointToNormal(keypoints2)
            for i in range(kps1.__len__()):
                save_file.write(kps1[i][0].__str__() + seprator + kps1[i][1].__str__() + seprator +
                                kps2[i][0].__str__() + seprator + kps2[i][1].__str__() + "\n")
        else:
            for i in range(keypoints1.__len__()):
                save_file.write(keypoints1[i][0].__str__() + seprator + keypoints1[i][1].__str__() + seprator +
                                keypoints2[i][0].__str__() + seprator + keypoints2[i][1].__str__() + "\n")
        save_file.close()
    else:
        print "keypoint list is empty,please check."


def joinKps(kps_list, des_list, parts):
    """
    对于分块模式，用于将每一块提取出的特征点列表和描述子合并成一个list和描述子

    :param kps_list: 包含有多个特征点列表的列表
    :param des_list: 包含有多个描述子的列表
    :param parts: 分块索引列表，通过getBlockRange函数获得
    :return: 合并后的特征点列表和对应描述子

    """

    kps = []

    for i in range(kps_list.__len__()):
        if type(kps_list[i][0]) is cv2.KeyPoint:
            tmp_kps = cvtCvKeypointToNormal(kps_list[i])
            print parts[i][0], parts[i][2]
            for j in range(tmp_kps.__len__()):
                kps.append((tmp_kps[j][0] + parts[i][2], tmp_kps[j][1] + parts[i][0]))
        else:
            for j in range(kps_list[i].__len__()):
                kps.append((kps_list[i][j][0] + parts[i][2], kps_list[i][j][1] + parts[i][0]))
    des = np.vstack(tuple(des_list))
    return kps, des


def siftSpeedUp(input_data):
    """
    基于多进程并行的Sift特征提取加速函数，供内部函数调用

    :param input_data: 一个元组，包含(影像块,块范围索引,每块特征点个数)
    :return: 返回一个元组，包含特征点和描述子

    """

    img_block = input_data[0]
    block_range = input_data[1]
    num = input_data[2]
    kp, des = getSiftKps(img_block, numKps=num)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def surfSpeedUp(input_data):
    """
    基于多进程并行的Surf特征提取加速函数，供内部函数调用

    :param input_data: 一个元组，包含(影像块,块范围索引,海塞矩阵阈值)
    :return: 返回一个元组，包含特征点和描述子

    """

    img_block = input_data[0]
    block_range = input_data[1]
    thHessian = input_data[2]
    kp, des = getSurfKps(img_block, hessianTh=thHessian)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def orbSpeedUp(input_data):
    """
    基于多进程并行的Orb特征提取加速函数，供内部函数调用

    :param input_data: 一个元组，包含(影像块,块范围索引,每块特征点个数)
    :return: 返回一个元组，包含特征点和描述子

    """

    img_block = input_data[0]
    block_range = input_data[1]
    num = input_data[2]
    kp, des = getOrbKps(img_block, numKps=num)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def getSiftKpsWithBlockSpeedUp(img, row=2, col=2, kpsPerBlock=2000, processNum=4):
    """
    多进程并行的分块Sift提取加速函数

    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块提取的特征点数量，默认为2000
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子

    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], kpsPerBlock))
    if processNum > row * col:
        pool = Pool(processes=row * col)
    else:
        pool = Pool(processes=processNum)
    res = pool.map(siftSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getSurfKpsWithBlockSpeedUp(img, row=2, col=2, thHessian=1500, processNum=4):
    """
    多进程并行的分块Surf提取加速函数

    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param thHessian: Surf算子的海塞矩阵阈值，默认为1500
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子

    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], thHessian))
    if processNum > row * col:
        pool = Pool(processes=row * col)
    else:
        pool = Pool(processes=processNum)
    res = pool.map(surfSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getOrbKpsWithBlockSpeedUp(img, row=2, col=2, kpsPerBlock=2000, processNum=4):
    """
    多进程并行的分块Orb提取加速函数

    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点个数，默认为2000
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子

    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], kpsPerBlock))
    if processNum > row * col:
        pool = Pool(processes=row * col)
    else:
        pool = Pool(processes=processNum)
    res = pool.map(orbSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getSiftKpsWithBlock(img, row=2, col=2, kpsPerBlock=2000):
    """
    分块Sift特征提取函数，对于较大影像比较有效果

    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点个数，默认为2000
    :return: 提取的全图特征点和描述子

    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getSiftKps(img_part, numKps=kpsPerBlock)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def getSurfKpsWithBlock(img, row=2, col=2, thHessian=1500):
    """
    分块Surf特征提取函数，对于较大影像比较有效果

    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param thHessian: Surf的海塞矩阵阈值，默认为1500
    :return: 提取的全图特征点和描述子

    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getSurfKps(img_part, hessianTh=thHessian)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def getORBKpsWithBlock(img, row=2, col=2, kpsPerBlock=2000):
    """
    分块Orb特征提取函数，对于较大影像比较有效果

    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点数量，默认为2000
    :return: 提取的全图特征点和描述子

    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getOrbKps(img_part, numKps=kpsPerBlock)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def readKeypoints(filePath, separator):
    kps = []
    open_file = open(filePath, 'r')
    num = open_file.readline().strip()
    data_line = open_file.readline().strip()
    while data_line:
        split_parts = data_line.split(separator)
        kps.append((float(split_parts[0]), float(split_parts[1])))
        data_line = open_file.readline().strip()
    return kps


def readMatchPoints(filePath, separator):
    kps1 = []
    kps2 = []
    open_file = open(filePath, 'r')
    num = open_file.readline().strip()
    data_line = open_file.readline().strip()
    while data_line:
        split_parts = data_line.split(separator)
        kps1.append((float(split_parts[0]), float(split_parts[1])))
        kps2.append((float(split_parts[2]), float(split_parts[3])))
        data_line = open_file.readline().strip()
    return kps1, kps2


def checkAffineAccuarcy(kp1, kp2):
    affine = findAffine(kp1, kp2)
    T = np.mat(affine[:, 2].reshape(2, 1))
    R = np.mat(affine[:2, :2])

    kp2_ = []
    for i in range(kp1.__len__()):
        pt1 = np.mat(kp1[i]).reshape(2, 1)
        pt2_ = R * pt1 + T
        kp2_.append(pt2_)

    accuracy = []
    for i in range(kp2_.__len__()):
        dx = kp2_[i][0] - np.mat(kp2[i]).reshape(2, 1)[0]
        dy = kp2_[i][1] - np.mat(kp2[i]).reshape(2, 1)[1]
        d = np.sqrt(dx * dx + dy * dy)
        accuracy.append((float(dx), float(dy), float(d)))
    return accuracy


def drawAffineMap():
    pass
