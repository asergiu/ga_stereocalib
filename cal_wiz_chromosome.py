from random import randrange, randint, random

import cv2
import numpy as np

from cal_wiz_constants import Constants


class Chromosome:
    FLAGS = [cv2.CALIB_FIX_ASPECT_RATIO,
             cv2.CALIB_ZERO_TANGENT_DIST,
             cv2.CALIB_USE_INTRINSIC_GUESS,
             cv2.CALIB_SAME_FOCAL_LENGTH,
             cv2.CALIB_RATIONAL_MODEL,
             cv2.CALIB_FIX_K3,
             cv2.CALIB_FIX_K4,
             cv2.CALIB_FIX_K5]
    FLAGS_STR = ['FIX_ASPECT_RATIO',
                 'ZERO_TANGENT_DIST',
                 'USE_INTRINSIC_GUESS',
                 'SAME_FOCAL_LENGTH',
                 'RATIONAL_MODEL',
                 'FIX_K3',
                 'FIX_K4',
                 'FIX_K5']
    N_FLAGS = len(FLAGS)

    GENES = [g for g in range(Constants.TOTAL)]
    N_GENES = len(GENES)
    MIN_GENES = 15

    def is_valid(self) -> bool:
        result = 0
        for g in self.genes:
            if g:
                result += 1
        return result >= Chromosome.MIN_GENES

    def get_flags(self) -> int:
        result = 0
        for g in range(Chromosome.N_FLAGS):
            if self.flags[g]:
                result |= Chromosome.FLAGS[g]
        return result

    def __init__(self, flags: [bool] = None, genes: [bool] = None):
        if flags is not None and genes is not None:
            assert (len(flags) == Chromosome.N_FLAGS)
            assert (len(genes) == Chromosome.N_GENES)
            self.flags = flags
            self.genes = genes
        else:
            done = False
            while not done:
                self.flags = []
                self.genes = []
                for _ in range(Chromosome.N_FLAGS):
                    if Constants.FLAGS:
                        self.flags.append(True if randint(0, 1) > 0 else False)
                    else:
                        self.flags.append(False)
                for _ in range(Chromosome.N_GENES):
                    self.genes.append(True if randint(0, 1) > 0 else False)
                if self.is_valid():
                    done = True

    def __str__(self) -> str:
        result = ""
        for g in range(Chromosome.N_FLAGS):
            if self.flags[g]:
                result += str(Chromosome.FLAGS_STR[g]) + ' '
        for g in range(Chromosome.N_GENES):
            if self.genes[g]:
                result += str(Chromosome.GENES[g]) + ' '
        return result

    @staticmethod
    def clone(chromosome):
        return Chromosome(flags=np.array(chromosome.flags, dtype=bool).tolist(),
                          genes=np.array(chromosome.genes, dtype=bool).tolist())

    @staticmethod
    def to_string(chromosome):
        result = ''
        for g in range(Chromosome.N_FLAGS):
            result += '1' if chromosome.flags[g] else '0'
        for g in range(Chromosome.N_GENES):
            result += '1' if chromosome.genes[g] else '0'
        return result

    @staticmethod
    def from_string(chromosome):
        flags = []
        genes = []
        for g in range(Chromosome.N_FLAGS):
            flags.append(True if chromosome[g] == '1' else False)
        for g in range(Chromosome.N_FLAGS, Chromosome.N_FLAGS + Chromosome.N_GENES):
            genes.append(True if chromosome[g] == '1' else False)
        return Chromosome(flags=flags, genes=genes)

    @staticmethod
    def operator_mutation(chromosome):
        flags = []
        genes = []
        for g in chromosome.flags:
            flags.append(g)
        for g in chromosome.genes:
            genes.append(g)
        thr = 0.25
        p = random()
        if Constants.FLAGS and p < thr:
            p = randrange(Chromosome.N_FLAGS)
            flags[p] = not flags[p]
        if not Constants.FLAGS or p >= thr:
            p = randrange(Chromosome.N_GENES)
            genes[p] = not genes[p]
        return Chromosome(flags=flags, genes=genes)

    @staticmethod
    def operator_crossover(chromosome_1, chromosome_2):
        flags_1 = chromosome_1.flags
        flags_2 = chromosome_2.flags
        genes_1 = []
        genes_2 = []
        p = randrange(Chromosome.N_GENES)
        for g in range(0, p):
            genes_1.append(chromosome_1.genes[g])
            genes_2.append(chromosome_2.genes[g])
        for g in range(p, Chromosome.N_GENES):
            genes_1.append(chromosome_2.genes[g])
            genes_2.append(chromosome_1.genes[g])
        return Chromosome(flags=flags_1, genes=genes_1), Chromosome(flags=flags_2, genes=genes_2)
