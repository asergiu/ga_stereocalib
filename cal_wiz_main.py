import math
import os
import threading
import time
from random import randrange, seed, random
from shutil import copyfile

import cv2
import numpy as np

from cal_wiz_chromosome import Chromosome
from cal_wiz_constants import Constants

seed()

GA_EPOCHS = 50
GA_POOL_SIZE = 25
GA_P_MUTATION = 0.25
GA_P_CROSSOVER = 0.75
GA_T_CATACLYSM = 0.01
GA_P_CATACLYSM = 0.50

ROOT_IN = 'in'
ROOT_OUT = 'out'

CRIT_MONO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1E-3)
CRIT_STEREO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 90, 1E-9)

OBJ_PTS = np.zeros((Constants.CHESS[0] * Constants.CHESS[1], 3), np.float32)
OBJ_PTS[:, :2] = np.mgrid[0:Constants.CHESS[0], 0:Constants.CHESS[1]].T.reshape((-1, 2)) * Constants.SQ_MM
OBJ_POINTS = []

FILE_NAMES = []
IMG_POINTS_L = []
IMG_POINTS_R = []


def init():
    print('Loading calibration images...')
    for g in range(Chromosome.N_GENES):
        g_str = str(Chromosome.GENES[g] + 1)

        chess_img_left = cv2.imread(ROOT_IN + '/left (' + g_str + ').jpg', cv2.IMREAD_GRAYSCALE)
        if Constants.FLIP:
            chess_img_left = cv2.flip(src=chess_img_left, flipCode=1)
        chess_img_left = cv2.resize(src=chess_img_left,
                                    dsize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)

        chess_img_right = cv2.imread(ROOT_IN + '/right (' + g_str + ').jpg', cv2.IMREAD_GRAYSCALE)
        if Constants.FLIP:
            chess_img_right = cv2.flip(src=chess_img_right, flipCode=1)
        chess_img_right = cv2.resize(src=chess_img_right,
                                     dsize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)

        flg = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK

        ret_l, corners_l = cv2.findChessboardCorners(
            image=chess_img_left,
            patternSize=Constants.CHESS,
            flags=flg)

        ret_r, corners_r = cv2.findChessboardCorners(
            image=chess_img_right,
            patternSize=Constants.CHESS,
            flags=flg)

        if not ret_l or not ret_r:
            print('Could not find chessboard corners in image pair ' + g_str + '!')
            continue
        else:
            print('Image pair ' + g_str + ' OK...')

        cv2.cornerSubPix(chess_img_left, corners_l, (Constants.CORNER, Constants.CORNER), (-1, -1), CRIT_MONO)
        cv2.cornerSubPix(chess_img_right, corners_r, (Constants.CORNER, Constants.CORNER), (-1, -1), CRIT_MONO)

        OBJ_POINTS.append(OBJ_PTS)
        FILE_NAMES.append(g_str)
        IMG_POINTS_L.append(corners_l)
        IMG_POINTS_R.append(corners_r)
    print('Loaded calibration images...')


def measure_board(stereo: {}, excluded: [int]) -> float:
    if excluded is None or len(excluded) == 0:
        count = Chromosome.N_GENES
        img_points_l = np.array(IMG_POINTS_L, dtype=float).reshape((count, Constants.CHESS[1], Constants.CHESS[0], 2))
        img_points_r = np.array(IMG_POINTS_R, dtype=float).reshape((count, Constants.CHESS[1], Constants.CHESS[0], 2))
    else:
        count = Chromosome.N_GENES - len(excluded)
        img_points_l = [IMG_POINTS_L[idx] for idx in range(Chromosome.N_GENES) if idx not in excluded]
        img_points_l = np.array(img_points_l, dtype=float).reshape((count, Constants.CHESS[1], Constants.CHESS[0], 2))
        img_points_r = [IMG_POINTS_R[idx] for idx in range(Chromosome.N_GENES) if idx not in excluded]
        img_points_r = np.array(img_points_r, dtype=float).reshape((count, Constants.CHESS[1], Constants.CHESS[0], 2))

    err_board = 0.0
    pts_board = 0
    for id_img in range(count):
        for id_x1 in range(0, Constants.CHESS[0] - 1):
            for id_y1 in range(0, Constants.CHESS[1] - 1):
                lxy1 = img_points_l[id_img][id_y1][id_x1]
                lxy1 = cv2.undistortPoints(src=lxy1,
                                           cameraMatrix=stereo['M1'],
                                           distCoeffs=stereo['D1'],
                                           R=None,
                                           P=stereo['M1']).flatten()
                rxy1 = img_points_r[id_img][id_y1][id_x1]
                rxy1 = cv2.undistortPoints(src=rxy1,
                                           cameraMatrix=stereo['M2'],
                                           distCoeffs=stereo['D2'],
                                           R=None,
                                           P=stereo['M2']).flatten()
                xy1 = cv2.triangulatePoints(projMatr1=stereo['P1'],
                                            projMatr2=stereo['P2'],
                                            projPoints1=lxy1.astype(dtype=float),
                                            projPoints2=rxy1.astype(dtype=float))
                xy1 = cv2.convertPointsFromHomogeneous(xy1.T).flatten()

                for id_x2 in range(id_x1 + 1, Constants.CHESS[0]):
                    for id_y2 in range(id_y1 + 1, Constants.CHESS[1]):
                        lxy2 = img_points_l[id_img][id_y2][id_x2]
                        lxy2 = cv2.undistortPoints(src=lxy2,
                                                   cameraMatrix=stereo['M1'],
                                                   distCoeffs=stereo['D1'],
                                                   R=None,
                                                   P=stereo['M1']).flatten()
                        rxy2 = img_points_r[id_img][id_y2][id_x2]
                        rxy2 = cv2.undistortPoints(src=rxy2,
                                                   cameraMatrix=stereo['M2'],
                                                   distCoeffs=stereo['D2'],
                                                   R=None,
                                                   P=stereo['M2']).flatten()
                        xy2 = cv2.triangulatePoints(projMatr1=stereo['P1'],
                                                    projMatr2=stereo['P2'],
                                                    projPoints1=lxy2.astype(dtype=float),
                                                    projPoints2=rxy2.astype(dtype=float))
                        xy2 = cv2.convertPointsFromHomogeneous(xy2.T).flatten()

                        d_computed = np.linalg.norm(xy1 - xy2)
                        d_actual = np.array([id_x1, id_y1], dtype=float) - np.array([id_x2, id_y2], dtype=float)
                        d_actual = np.linalg.norm(d_actual) * Constants.SQ_MM
                        err = abs(d_actual - d_computed)

                        pts_board += 1
                        err_board += err
    return err_board / pts_board


def measure_epi(stereo: {}, chromosome: Chromosome) -> float:
    count = chromosome.genes.count(True)
    img_points_l = [IMG_POINTS_L[idx] for idx in range(Chromosome.N_GENES) if chromosome.genes[idx]]
    img_points_l = np.array(img_points_l, dtype=float).reshape((count, Constants.CHESS[1] * Constants.CHESS[0], 2))
    img_points_r = [IMG_POINTS_R[idx] for idx in range(Chromosome.N_GENES) if chromosome.genes[idx]]
    img_points_r = np.array(img_points_r, dtype=float).reshape((count, Constants.CHESS[1] * Constants.CHESS[0], 2))

    err_epi = 0
    pts_epi = 0
    for idx in range(count):
        pts_l = np.copy(img_points_l[idx]).reshape((-1, 2))
        pts_l = cv2.undistortPoints(src=pts_l,
                                    cameraMatrix=stereo['M1'],
                                    distCoeffs=stereo['D1'],
                                    R=None,
                                    P=stereo['M1'])
        lines_l = cv2.computeCorrespondEpilines(points=pts_l, whichImage=1, F=stereo['F'])

        pts_r = np.copy(img_points_r[idx]).reshape((-1, 2))
        pts_r = cv2.undistortPoints(src=pts_r,
                                    cameraMatrix=stereo['M2'],
                                    distCoeffs=stereo['D2'],
                                    R=None,
                                    P=stereo['M2'])
        lines_r = cv2.computeCorrespondEpilines(points=pts_r, whichImage=2, F=stereo['F'])

        pts_l = pts_l.reshape((-1, 2))
        pts_r = pts_r.reshape((-1, 2))

        lines_l = lines_l.reshape((-1, 3))
        lines_r = lines_r.reshape((-1, 3))

        err = 0
        for idy in range(Constants.CHESS[1] * Constants.CHESS[0]):
            err += abs(pts_l[idy][0] * lines_r[idy][0] + pts_l[idy][1] * lines_r[idy][1] + lines_r[idy][2])
            err += abs(pts_r[idy][0] * lines_l[idy][0] + pts_r[idy][1] * lines_l[idy][1] + lines_l[idy][2])

        err_epi += err
        pts_epi += Constants.CHESS[1] * Constants.CHESS[0]
    return err_epi / pts_epi


def fitness_mono(chromosome: Chromosome) -> (str, float):
    obj_points = []
    img_points_l = []
    img_points_r = []
    for g in range(Chromosome.N_GENES):
        if chromosome.genes[g]:
            obj_points.append(OBJ_POINTS[g])
            img_points_l.append(IMG_POINTS_L[g])
            img_points_r.append(IMG_POINTS_R[g])

    mtx_l = cv2.initCameraMatrix2D(objectPoints=obj_points,
                                   imagePoints=img_points_l,
                                   imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT))
    mtx_r = cv2.initCameraMatrix2D(objectPoints=obj_points,
                                   imagePoints=img_points_r,
                                   imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT))

    dst_l = np.zeros(5, dtype=float)
    dst_r = np.zeros(5, dtype=float)

    flg = chromosome.get_flags() if Constants.FLAGS else 0
    flg |= cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

    err_rms_l, mtx_l, dst_l, rvec_l, tvec_l = cv2.calibrateCamera(objectPoints=obj_points,
                                                                  imagePoints=img_points_l,
                                                                  cameraMatrix=mtx_l,
                                                                  distCoeffs=dst_l,
                                                                  imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                                  criteria=CRIT_MONO,
                                                                  flags=flg)
    err_rms_r, mtx_r, dst_r, rvec_r, tvec_r = cv2.calibrateCamera(objectPoints=obj_points,
                                                                  imagePoints=img_points_r,
                                                                  cameraMatrix=mtx_r,
                                                                  distCoeffs=dst_r,
                                                                  imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                                  criteria=CRIT_MONO,
                                                                  flags=flg)

    cnt = 0
    err_prj_l = 0
    err_prj_r = 0

    for idx in range(len(obj_points)):
        cnt += len(obj_points[idx])

        prj_pts_l, _ = cv2.projectPoints(objectPoints=obj_points[idx],
                                         rvec=rvec_l[idx],
                                         tvec=tvec_l[idx],
                                         cameraMatrix=mtx_l,
                                         distCoeffs=dst_l)
        err = cv2.norm(img_points_l[idx] - prj_pts_l, normType=cv2.NORM_L2)
        err_prj_l += err ** 2

        prj_pts_r, _ = cv2.projectPoints(objectPoints=obj_points[idx],
                                         rvec=rvec_r[idx],
                                         tvec=tvec_r[idx],
                                         cameraMatrix=mtx_r,
                                         distCoeffs=dst_r)
        err = cv2.norm(img_points_r[idx] - prj_pts_r, normType=cv2.NORM_L2)
        err_prj_r += err ** 2

    err_prj_l = math.sqrt(err_prj_l / cnt)
    err_prj_r = math.sqrt(err_prj_r / cnt)
    err_prj = min(err_prj_l, err_prj_r)

    fs_path = Chromosome.to_string(chromosome) + '.yml'
    fs = cv2.FileStorage(os.path.join(ROOT_OUT, fs_path), cv2.FILE_STORAGE_WRITE)
    fs.write('M1', mtx_l)
    fs.write('D1', dst_l)
    fs.write('M2', mtx_r)
    fs.write('D2', dst_r)
    fs.write('SZ', (int(Constants.S_WIDTH), int(Constants.S_HEIGHT)))
    fs.write('ERR_PRJ_L', err_prj_l)
    fs.write('ERR_PRJ_R', err_prj_r)
    fs.write('ERR_PRJ', err_prj)
    fs.release()

    return fs_path, err_prj


def fitness_stereo(chromosome: Chromosome) -> (str, float, float, float):
    obj_points = []
    img_points_l = []
    img_points_r = []
    for g in range(Chromosome.N_GENES):
        if chromosome.genes[g]:
            obj_points.append(OBJ_POINTS[g])
            img_points_l.append(IMG_POINTS_L[g])
            img_points_r.append(IMG_POINTS_R[g])

    mtx_l = cv2.initCameraMatrix2D(objectPoints=obj_points,
                                   imagePoints=img_points_l,
                                   imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT))
    mtx_r = cv2.initCameraMatrix2D(objectPoints=obj_points,
                                   imagePoints=img_points_r,
                                   imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT))

    dst_l = np.zeros(5, dtype=float)
    dst_r = np.zeros(5, dtype=float)

    flg = chromosome.get_flags() if Constants.FLAGS else 0
    flg |= cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_SAME_FOCAL_LENGTH

    err_rms, mtx_l, dst_l, mtx_r, dst_r, r, t, e, f = cv2.stereoCalibrate(
        objectPoints=obj_points,
        imagePoints1=img_points_l,
        imagePoints2=img_points_r,
        cameraMatrix1=mtx_l,
        distCoeffs1=dst_l,
        cameraMatrix2=mtx_r,
        distCoeffs2=dst_r,
        imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
        criteria=CRIT_STEREO,
        flags=flg)

    r1, r2, p1, p2, q, roi_top, roi_bot = cv2.stereoRectify(
        cameraMatrix1=mtx_l,
        distCoeffs1=dst_l,
        cameraMatrix2=mtx_r,
        distCoeffs2=dst_r,
        imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
        R=r,
        T=t,
        alpha=1,
        flags=cv2.CALIB_ZERO_DISPARITY)

    valid_area_top = roi_top[3] * roi_top[2]
    valid_area_bot = roi_bot[3] * roi_bot[2]
    valid_area_min = min(valid_area_top, valid_area_bot)
    valid_ratio = valid_area_min / (Constants.S_WIDTH * Constants.S_HEIGHT)

    if valid_ratio < Constants.VALID_ROI_RATIO:
        return '', Constants.INF, Constants.INF, Constants.INF, Constants.INF, Constants.INF

    stereo = {'M1': mtx_l,
              'M2': mtx_r,
              'D1': dst_l,
              'D2': dst_r,
              'R1': r1,
              'R2': r2,
              'P1': p1,
              'P2': p2,
              'R': r,
              'T': t,
              'E': e,
              'F': f,
              'Q': q,
              'SZ': (int(Constants.S_WIDTH), int(Constants.S_HEIGHT))}

    err_epi = measure_epi(stereo, chromosome=chromosome)
    err_board = measure_board(stereo, excluded=[])

    fs_path = Chromosome.to_string(chromosome) + '.yml'
    fs = cv2.FileStorage(os.path.join(ROOT_OUT, fs_path), cv2.FILE_STORAGE_WRITE)
    fs.write('M1', mtx_l)
    fs.write('M2', mtx_r)
    fs.write('D1', dst_l)
    fs.write('D2', dst_r)
    fs.write('R1', r1)
    fs.write('R2', r2)
    fs.write('P1', p1)
    fs.write('P2', p2)
    fs.write('R', r)
    fs.write('T', t)
    fs.write('E', e)
    fs.write('F', f)
    fs.write('Q', q)
    fs.write('SZ', (int(Constants.S_WIDTH), int(Constants.S_HEIGHT)))
    fs.write('ERR_RMS', err_rms)
    fs.write('ERR_EPI', err_epi)
    fs.write('ERR_BOARD', err_board)
    fs.write('VALID_RATIO', valid_ratio)
    fs.release()

    return fs_path, err_rms, err_epi, err_board


REPO_THREAD = {}
REPO_FITNESS = {}


class Worker:
    @staticmethod
    def __size__():
        return len(REPO_THREAD)

    @staticmethod
    def __fitness__(chromosome: Chromosome):
        if Constants.MONO:
            REPO_FITNESS[Chromosome.to_string(chromosome)] = fitness_mono(chromosome)
        else:
            REPO_FITNESS[Chromosome.to_string(chromosome)] = fitness_stereo(chromosome)

    @staticmethod
    def enqueue_one(chromosome: Chromosome):
        chromo_str = Chromosome.to_string(chromosome)
        while chromo_str in REPO_THREAD:
            chromosome = Chromosome()
            chromo_str = Chromosome.to_string(chromosome)
        if Constants.PARALLEL:
            REPO_THREAD[chromo_str] = threading.Thread(target=Worker.__fitness__, args=(chromosome,))
        else:
            REPO_THREAD[chromo_str] = None
            Worker.__fitness__(chromosome)

    @staticmethod
    def dequeue_all() -> ([Chromosome], [float]):
        if Constants.PARALLEL:
            for key in REPO_THREAD:
                REPO_THREAD[key].start()
            for key in REPO_THREAD:
                REPO_THREAD[key].join()
        keys = []
        fitnesses = []
        for key in REPO_FITNESS:
            keys.append(Chromosome.from_string(key))
            if Constants.MONO:
                fitness = REPO_FITNESS[key][1]
            else:
                fitness = REPO_FITNESS[key][3]
            fitnesses.append(fitness)
        REPO_THREAD.clear()
        REPO_FITNESS.clear()
        return keys, fitnesses


def ga() -> (Chromosome, float):
    print('Parallel is ' + str(Constants.PARALLEL))
    print('Flip is ' + str(Constants.FLIP))
    print('Flags are ' + str(Constants.FLAGS))
    print('Mono is ' + str(Constants.MONO))

    print('Initializing...')
    for _ in range(GA_POOL_SIZE):
        Worker.enqueue_one(Chromosome())
    chromosome_pool, chromosome_fitness = Worker.dequeue_all()
    print('Initialized...')

    garden_of_eden_pool = []
    garden_of_eden_fitness = []
    timing = []

    for e in range(GA_EPOCHS):
        print('Epoch: ' + str(e + 1))
        start_time = time.time()

        for c in range(min(len(chromosome_pool), GA_POOL_SIZE)):
            r = random()
            if r < GA_P_CROSSOVER and c > 0:
                chromosome_1 = chromosome_pool[c]
                chromosome_2 = chromosome_pool[randrange(c)]
                offspring_1, offspring_2 = Chromosome.operator_crossover(chromosome_1, chromosome_2)
                if offspring_1.is_valid():
                    Worker.enqueue_one(offspring_1)
                if offspring_2.is_valid():
                    Worker.enqueue_one(offspring_2)

            r = random()
            if r < GA_P_MUTATION:
                chromosome = chromosome_pool[c]
                offspring = Chromosome.operator_mutation(chromosome)
                if offspring.is_valid():
                    Worker.enqueue_one(offspring)
        new_chromosomes, new_fitnesses = Worker.dequeue_all()

        chromosome_pool += new_chromosomes
        chromosome_fitness += new_fitnesses

        done = False
        while not done:
            done = True
            for c in range(len(chromosome_pool) - 1):
                if chromosome_fitness[c] > chromosome_fitness[c + 1]:
                    aux = chromosome_fitness[c]
                    chromosome_fitness[c] = chromosome_fitness[c + 1]
                    chromosome_fitness[c + 1] = aux
                    aux = Chromosome.clone(chromosome_pool[c])
                    chromosome_pool[c] = Chromosome.clone(chromosome_pool[c + 1])
                    chromosome_pool[c + 1] = Chromosome.clone(aux)
                    done = False

        chromosome_pool = chromosome_pool[:GA_POOL_SIZE]
        chromosome_fitness = chromosome_fitness[:GA_POOL_SIZE]

        garden_of_eden_pool.append(chromosome_pool[0])
        garden_of_eden_fitness.append(chromosome_fitness[0])

        minimum = chromosome_fitness[0]
        maximum = chromosome_fitness[-1]
        length = len(chromosome_pool)
        to_reset = int(GA_P_CATACLYSM * length)

        if abs(maximum - minimum) < GA_T_CATACLYSM:
            print('Type 1 Cataclysm!')
            for _ in range(to_reset):
                Worker.enqueue_one(Chromosome())
            new_chromosomes, new_fitnesses = Worker.dequeue_all()
            chromosome_pool[:to_reset] = new_chromosomes
            chromosome_fitness[:to_reset] = new_fitnesses

        if chromosome_fitness[to_reset] == Constants.INF:
            print('Type 2 Cataclysm!')
            for _ in range(length - to_reset):
                Worker.enqueue_one(Chromosome())
            new_chromosomes, new_fitnesses = Worker.dequeue_all()
            chromosome_pool[to_reset:] = new_chromosomes
            chromosome_fitness[to_reset:] = new_fitnesses

        print(chromosome_fitness[0], chromosome_fitness[-1])
        end_time = time.time()
        timing.append(end_time - start_time)
        timing_np = np.array(timing, dtype=float)
        timing_mean = np.mean(timing_np)
        timing_eta = (GA_EPOCHS - (e + 1)) * timing_mean / 60.0
        print('ETA: ' + str(int(timing_eta)) + ' minutes...')

    done = False
    while not done:
        done = True
        for c in range(len(garden_of_eden_pool) - 1):
            if garden_of_eden_fitness[c] > garden_of_eden_fitness[c + 1]:
                aux = garden_of_eden_fitness[c]
                garden_of_eden_fitness[c] = garden_of_eden_fitness[c + 1]
                garden_of_eden_fitness[c + 1] = aux
                aux = Chromosome.clone(garden_of_eden_pool[c])
                garden_of_eden_pool[c] = Chromosome.clone(garden_of_eden_pool[c + 1])
                garden_of_eden_pool[c + 1] = Chromosome.clone(aux)
                done = False

    return garden_of_eden_pool[0], garden_of_eden_fitness[0]


print('Splitting stereo images...')
for idx in range(1, Constants.TOTAL + 1):
    img_raw = cv2.imread(f'in/raw ({idx}).jpg')
    img_left = img_raw[:, 0: Constants.WIDTH]
    cv2.imwrite(f'in/left ({idx}).jpg', img_left)
    img_right = img_raw[:, Constants.WIDTH: Constants.WIDTH * 2]
    cv2.imwrite(f'in/right ({idx}).jpg', img_right)
print('Split stereo images...')

init()
sol, sol_fit = ga()

with open(os.path.join(ROOT_OUT, 'results.log'), 'w+') as log:
    log.write('Flip is ' + str(Constants.FLIP) + '\n')
    log.write('Flags are ' + str(Constants.FLAGS) + '\n')
    log.write('Mono is ' + str(Constants.MONO) + '\n')
    log.write('Best overall fitness is: ' + str(sol_fit) + '\n')
    log.write('Selected image pair indexes are:\n')
    log.write(str(sol) + '\n')
    if Constants.MONO:
        best_yml, best_prj = fitness_mono(sol)
        log.write('Path to YML file is: ' + best_yml + '\n')
        log.write('PRJ error is: ' + str(best_prj) + '\n')
    else:
        best_yml, best_rms, best_epi, best_board = fitness_stereo(sol)
        log.write('Path to YML file is: ' + best_yml + '\n')
        log.write('RMS error is: ' + str(best_rms) + '\n')
        log.write('EPI error is: ' + str(best_epi) + '\n')
        log.write('BOARD error is: ' + str(best_board) + '\n')
    log.write('Genetic algorithm parameters are...\n')
    log.write('Number of epochs: ' + str(GA_EPOCHS) + '\n')
    log.write('Chromosome pool size: ' + str(GA_POOL_SIZE) + '\n')
    log.write('Mutation probability: ' + str(GA_P_MUTATION) + '\n')
    log.write('Crossover probability: ' + str(GA_P_CROSSOVER) + '\n')
    log.write('Cataclysm pool weight threshold: ' + str(GA_T_CATACLYSM) + '\n')
    log.write('Percentage of fittest individuals destroyed in cataclysm is: ' + str(GA_P_CATACLYSM) + '\n')
    log.write('Minimum number of image pairs to be considered is ' + str(Chromosome.MIN_GENES) + '\n')
    best_len = 0
    for i in range(Chromosome.N_GENES):
        if sol.genes[i]:
            best_len += 1
    log.write('The solution has chosen ' + str(best_len) + ' image pairs...\n')
    copyfile(os.path.join(ROOT_OUT, best_yml), os.path.join(ROOT_OUT, 'sol.yml'))
