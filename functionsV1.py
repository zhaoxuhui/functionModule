# coding=utf-8
import os
import numpy as np
import cv2
from osgeo import gdal
from gdalconst import *


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
    # 遍历
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
    # 遍历
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
    band_r = img[:, :, 0]
    band_g = img[:, :, 1]
    band_b = img[:, :, 2]
    return band_r, band_g, band_b


def mergeRGB(band_r, band_g, band_b):
    h = min(band_r.shape[0], band_g.shape[0], band_b.shape[0])
    w = min(band_r.shape[1], band_g.shape[1], band_b.shape[1])
    img = np.zeros([h, w, 3], np.uint8)
    img[:, :, 0] = band_r[:h, :w]
    img[:, :, 1] = band_g[:h, :w]
    img[:, :, 2] = band_b[:h, :w]
    return img


def getSurfKps(img, hessianTh=1500):
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=hessianTh)
    kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
    return kp, des


def getSiftKps(img, numKps=2000):
    sift = cv2.xfeatures2d_SIFT.create(nfeatures=numKps)
    kp, des = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img, None)
    return kp, des


def getOrbKps(img, numKps=2000):
    orb = cv2.ORB_create(nfeatures=numKps)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def drawKeypoints(img, kps, color=[0, 0, 255], rad=3, thickness=1):
    if img.shape.__len__() == 2:
        img_pro = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_pro = img.copy()
    if color[0] == -1 and color[1] == -1 and color[2] == -1:
        for point in kps:
            pt = (int(point.pt[0]), int(point.pt[1]))
            color[0] = getRandomNum(0, 255)
            color[1] = getRandomNum(0, 255)
            color[2] = getRandomNum(0, 255)
            cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
    else:
        for point in kps:
            pt = (int(point.pt[0]), int(point.pt[1]))
            cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
    return img_pro


def getRandomNum(start=0, end=100):
    return np.random.randint(start, end + 1)


def logTransform(img, v=200, c=256):
    img_normalize = img * 1.0 / c
    log_res = c * (np.log(1 + v * img_normalize) / np.log(v + 1))
    img_new = np.uint8(log_res)
    return img_new


def flannMatch(kp1, des1, kp2, des2):
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

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append([kp1[matches[i][0].queryIdx].pt[0], kp1[matches[i][0].queryIdx].pt[1]])
            good_kps2.append([kp2[matches[i][0].trainIdx].pt[0], kp2[matches[i][0].trainIdx].pt[1]])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        return good_kps1, good_kps2
    else:
        print("good matches:" + good_matches.__len__().__str__())
        return good_kps1, good_kps2


def bfMatch(kp1, des1, kp2, des2):
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
            if match.distance <= max(1.1 * min_dis, 15.0):
                g_matches.append(match)

        print("matches:" + g_matches.__len__().__str__())
        # 筛选
        for i in range(g_matches.__len__()):
            good_kps1.append([kp1[g_matches[i].queryIdx].pt[0], kp1[g_matches[i].queryIdx].pt[1]])
            good_kps2.append([kp2[g_matches[i].trainIdx].pt[0], kp2[g_matches[i].trainIdx].pt[1]])
        return good_kps1, good_kps2


def drawMatches(img1, kps1, img2, kps2, color=[0, 0, 255], rad=5, thickness=1):
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
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    band_1 = dataset.GetRasterBand(1)
    data = band_1.ReadAsArray(0, 0, band_1.XSize, band_1.YSize)
    return data


def getBlockRange(img, row=2, col=2):
    img_h = img.shape[1]
    img_w = img.shape[0]
    print img_h, img_w
    w_per_block = img_w / row
    h_per_block = img_h / col
    print h_per_block, w_per_block
    blocks = []
    for i in range(row):
        for j in range(col):
            w = i * w_per_block
            h = j * h_per_block
            rb_w = w + w_per_block
            rb_h = h + h_per_block
            print w, '-', rb_w, h, '-', rb_h
            blocks.append([w, rb_w, h, rb_h])
    return blocks


def findAffine(kps1, kps2):
    if kps1.__len__() < 3 or kps2.__len__() < 3:
        affine = None
    else:
        affine, mask = cv2.estimateAffine2D(np.array(kps1), np.array(kps2))
    return affine


def findHomography(kps1, kps2):
    if kps1.__len__() < 5 or kps2.__len__() < 5:
        homo = None
    else:
        homo, mask = cv2.findHomography(np.array(kps1), np.array(kps2))
    return homo


def resampleImg(img, trans):
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


def hist_calc(img, ratio):
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


def linearStretch(img, new_min, new_max, ratio):
    old_min, old_max = hist_calc(img, ratio)
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    print("=>2% linear stretch:")
    print('old min = %d,old max = %d new min = %d,new max = %d' % (old_min, old_max, new_min, new_max))
    img3 = np.uint8((new_max - new_min) / (old_max - old_min) * (img2 - old_min) + new_min)
    return img3


def resampleToBase(img_base, img_warp, flag='affine'):
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


def sift_flann(img1, img2):
    kp1, des1 = getSiftKps(img1)
    kp2, des2 = getSiftKps(img2)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2


def surf_flann(img1, img2):
    kp1, des1 = getSurfKps(img1)
    kp2, des2 = getSurfKps(img2)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2


def orb_bf(img1, img2):
    kp1, des1 = getOrbKps(img1)
    kp2, des2 = getOrbKps(img2)
    good_kp1, good_kp2 = bfMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2
