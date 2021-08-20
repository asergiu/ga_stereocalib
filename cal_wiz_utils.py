import math
import os
from glob import glob

import cv2
import numpy as np

from cal_wiz_constants import Constants


def measure_rms(root: str,
                flip: bool = False,
                chess: (int, int) = (9 - 1, 8 - 1),
                corner: int = 11,
                sq_mm: float = 7.0) -> (float, float):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1E-1)

    obj_pts = np.zeros((chess[0] * chess[1], 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:chess[0], 0:chess[1]].T.reshape((-1, 2)) * sq_mm

    obj_points = []
    img_points = []

    images = [y for x in os.walk(root) for y in glob(os.path.join(x[0], '*.jpg'))]
    for img in images:
        chess_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if flip:
            chess_img = cv2.flip(src=chess_img, flipCode=1)

        ret, corners = cv2.findChessboardCorners(
            image=chess_img,
            patternSize=chess,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not ret:
            print('Could not find chessboard corners in image ' + img + '!')
            exit(-1)

        cv2.cornerSubPix(chess_img, corners, (corner, corner), (-1, -1), crit)

        obj_points.append(obj_pts)
        img_points.append(corners)

    mtx = np.eye(3, dtype=float)
    dst = np.zeros(5, dtype=float)
    flg = cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

    err_rms, mtx, dst, r_vecs, t_vecs = cv2.calibrateCamera(objectPoints=obj_points,
                                                            imagePoints=img_points,
                                                            cameraMatrix=mtx,
                                                            distCoeffs=dst,
                                                            imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                            criteria=crit,
                                                            flags=flg)

    cnt = 0
    err_prj = 0
    for idx in range(len(obj_points)):
        cnt += len(obj_points[idx])
        prj_pts, _ = cv2.projectPoints(objectPoints=obj_points[idx],
                                       rvec=r_vecs[idx],
                                       tvec=t_vecs[idx],
                                       cameraMatrix=mtx,
                                       distCoeffs=dst)
        err = cv2.norm(img_points[idx] - prj_pts, normType=cv2.NORM_L2)
        err_prj += err ** 2
    return err_rms, math.sqrt(err_prj / cnt)
