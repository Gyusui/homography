# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:49:49 2017

@author: 牛帥
"""

import numpy as np
import cv2
import screeninfo

def f(a):
    p = np.array([[0, 0, 0, a[0], a[1], 1, -a[3]*a[0], -a[3]*a[1], -a[3]],
                  [a[0], a[1], 1, 0, 0, 0, -a[2]*a[0], -a[2]*a[1], -a[2]]])          #行列を表す
    return p


def estimate_homography(points):                 #ホモグラフィ推定関数
    p =[]                                        #点を展開して、Aは8＊8の行列、Bは余った行列
    for a in points:
        p.append(f(a))

    P = np.vstack(p)
    A = P[:, :8]
    B = P[:, 8:9]

    invA = np.linalg.inv(A)
    x = np.dot(invA, -B)
    x = invA.dot(-B)                       #Ph = 0  Ax+b = 0  Ax = -b
    h = np.vstack((x,1))                   #x = (-b)/A
    H = h.reshape(3,3)                     #Hは３＊３を変更するg
    return H

def perspective_transform(H, point):        #Hを利用して、プロジェクタ座標を計算する
    point = np.hstack((point, 1)).reshape(3, 1)
    point = H.dot(point)
    point = point / point[2]
    point = point.reshape(3)
    return point[:2]


if __name__ == '__main__':
    screen_id = 1

    # get the size of the screeng
    screen = screeninfo.get_monitors()[screen_id]         #モニターの解像度を獲得する
    width, height = screen.width, screen.height

    p_width = 854
    p_height = 480
    pw_pairs = np.loadtxt("points.txt")

    H_wp = estimate_homography(pw_pairs)
    H_pw = np.linalg.inv(H_wp)

    points_w = np.loadtxt("targets.txt")              #四角形の上でスクリーン座標系を作る、これとプロジェクタ座標系のHを利用して計算する
    points_p = []
    for point_w in points_w:
        point_p = perspective_transform(H_pw, point_w)
        points_p.append(point_p)

    points_p = np.vstack(points_p)
    points_s = np.array([[0, 0],
                         [p_width, 0],
                         [p_width, p_height],
                         [0, p_height]])
    sp_pairs = np.hstack((points_s, points_p))
    H_ps = estimate_homography(sp_pairs)
    H_sp = np.linalg.inv(H_ps)

    img = cv2.imread('flower_p.jpg')
    dst = cv2.warpPerspective(img, H_ps, (p_width, p_height))
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x, screen.y)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('flower_s.png',dst)
    #cv.WarpPerspective(src, dst, mapMatrix, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))