import numpy as np


class Constants:
    INF = 1E6
    WIDTH = 4056
    HEIGHT = 3040
    SCALE = 2
    S_WIDTH = WIDTH // SCALE
    S_HEIGHT = HEIGHT // SCALE
    K_MAT = np.array(
        [S_WIDTH, 0, S_WIDTH / 2,
         0, S_WIDTH, S_HEIGHT / 2,
         0, 0, 1], dtype=float).reshape((3, 3))
    D_VEC = np.array([0, 0, 0, 0], dtype=float)

    DEBUG = False
    PARALLEL = True
    FLIP = False
    FLAGS = False
    MONO = False

    TOTAL = 30
    CHESS = (9 - 1, 8 - 1)
    CORNER = 11
    SQ_MM = 15.00
    VALID_ROI_RATIO = 0
