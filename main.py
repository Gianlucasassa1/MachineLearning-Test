import numpy as np
import matplotlib.pyplot as plt

import argparse
from load import *
from Train.train import *
from Preprocess.DatasetPlots import *
from Preprocess.PCA import *
from Preprocess.LDA import *
from Models.Gaussian.utils import *
from Models.LogisticRegression.LogisticRegression import LR_Classifier, QuadraticLR_Classifier
from Models.SupportVector.SVM import SVMClassifier
from Models.SupportVector.SVMKernel import SVMKernelClassifier
from Models.MixtureModels.GMM import GMMClassifier
from Evaluation.evaluation import *
from Calibration.calibration import *
import os
import tqdm
import glob

LOG_FOLDER = './runs'

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

all_files = sorted(glob.glob(os.path.join(LOG_FOLDER, '*')))

if len(all_files) == 0:
    last_file_number = -1
    FILE_PATH = os.path.join(LOG_FOLDER, 'run_0.txt')
else:
    last_file = all_files[-1]
    last_file_number = int(os.path.splitext(os.path.basename(last_file))[0].split('_')[1])
    FILE_PATH = os.path.join(LOG_FOLDER, f'run_{last_file_number + 1}.txt')


def log(string_to_add, print_the_string=False):
    """
    Log into a txt file
    """
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'w') as file:
            file.write(f'###### RUN {last_file_number + 1} ###### \n')
        file.close()

    if string_to_add:
        with open(FILE_PATH, 'a') as file:
            file.write(string_to_add + '\n')
        file.close()

    if print_the_string:
        print(string_to_add)


def main(type, mode):
    with open('output_info.txt', 'a') as file:
        (DTR, LTR), (DTE, LTE) = loadTrainingAndTestData('Data/Train.txt', 'Data/Test.txt')
        DTR = DTR.T
        LTR = LTR.T
        DTE = DTE.T
        LTE = LTE.T

        # Dataset
        # ██████   █████  ████████  █████  ███████ ███████ ████████
        # ██   ██ ██   ██    ██    ██   ██ ██      ██         ██
        # ██   ██ ███████    ██    ███████ ███████ █████      ██
        # ██   ██ ██   ██    ██    ██   ██      ██ ██         ██
        # ██████  ██   ██    ██    ██   ██ ███████ ███████    ██
        #

        if mode == 'dataset' or mode == 'all':
            file.write("Dataset" + '\n')

            m = 12

            #generalPlot(DTR, LTR)
            #featurePlot(DTR, LTR, m, "Base")
            #mixedPlot(DTR, LTR, m, "Base")
            #
            # DC = centerData(DTR)
            # featurePlot(DC, LTR, m, "Base+Center")
            # mixedPlot(DC, LTR, m, "Base+Center")
            #
            #
            # # PCA
            # DP, P = PCA_impl(DTR, m)  # try with 2-dimension subplot
            # DTEP = np.dot(P.T, DTE)
            # mixedPlot(DP, LTR, m, "PCA")  # plotting the data on a 2-D cartesian graph
            # featurePlot(DP, LTR, m, "PCA")
            #
            # LDA
            # maleFemaleFeaturesPlot(DTR, LTR)
            # DW, W = LDA_impl(DTR, LTR, m)
            # mixedPlot(DW, LTR, m, "LDA")
            # featurePlot(DW, LTR, m, "LDA")
            #
            # # LDA and PCA
            # DPW, W = LDA_impl(DP, LTR, m)
            # mixedPlot(DW, LTR, m, "LDA+PCA")
            # featurePlot(DW, LTR, m, "LDA+PCA")
            #
            # # Pearson
            # maleFemaleFeaturesPlot(DTR, LTR)
            # correlationPlotP(DTR, LTR, 0)
            # heatmapPlot(DTR, LTR)

            # PCA and variance plot
            PCA_plot(DTR)

        # Train
        # ████████ ██████   █████  ██ ███    ██
        #    ██    ██   ██ ██   ██ ██ ████   ██
        #    ██    ██████  ███████ ██ ██ ██  ██
        #    ██    ██   ██ ██   ██ ██ ██  ██ ██
        #    ██    ██   ██ ██   ██ ██ ██   ████
        #

        if mode == 'train' or mode == 'all':
            file.write("Train..." + '\n')

            # Gaussian
            #    __  __  __      __   _____
            #  |  \/  | \ \    / /  / ____|
            #  | \  / |  \ \  / /  | |  __
            #  | |\/| |   \ \/ /   | | |_ |
            #  | |  | |    \  /    | |__| |
            #  |_|  |_|     \/      \_____|
            #

            if type == "MVG" or type == 'all':
                trainGaussian(DTR, LTR, file)

            # Logistic Regression
            #   _        _____
            #  | |      |  __ \
            #  | |      | |__) |
            #  | |      |  _  /
            #  | |____  | | \ \
            #  |______| |_|  \_\
            #

            if type == "LR" or type == 'all':
                trainLR(DTR, LTR, file)

            # Logistic Regression - Quadratic
            #   _        _____                  ____                        _                  _     _
            #  | |      |  __ \                / __ \                      | |                | |   (_)
            #  | |      | |__) |    ______    | |  | |  _   _    __ _    __| |  _ __    __ _  | |_   _    ___
            #  | |      |  _  /    |______|   | |  | | | | | |  / _` |  / _` | | '__|  / _` | | __| | |  / __|
            #  | |____  | | \ \               | |__| | | |_| | | (_| | | (_| | | |    | (_| | | |_  | | | (__
            #  |______| |_|  \_\               \___\_\  \__,_|  \__,_|  \__,_| |_|     \__,_|  \__| |_|  \___|
            #

            if type == "QLR" or type == 'all':
                trainQLR(DTR, LTR, file)

            # Support Vector Machine
            #    _____  __      __  __  __
            #   / ____| \ \    / / |  \/  |
            #  | (___    \ \  / /  | \  / |
            #   \___ \    \ \/ /   | |\/| |
            #   ____) |    \  /    | |  | |
            #  |_____/      \/     |_|  |_|
            #

            if type == "SVM" or type == 'all':
                trainSVM(DTR, LTR, file)

            # Support Vector Machine - Kernel
            #    _____  __      __  __  __                _  __                               _
            #   / ____| \ \    / / |  \/  |              | |/ /                              | |
            #  | (___    \ \  / /  | \  / |    ______    | ' /    ___   _ __   _ __     ___  | |
            #   \___ \    \ \/ /   | |\/| |   |______|   |  <    / _ \ | '__| | '_ \   / _ \ | |
            #   ____) |    \  /    | |  | |              | . \  |  __/ | |    | | | | |  __/ | |
            #  |_____/      \/     |_|  |_|              |_|\_\  \___| |_|    |_| |_|  \___| |_|
            #

            if type == "KSVM" or type == 'all':
                trainKSVM(DTR, LTR, file)

            # Gaussian Mixture Models
            #     _____   __  __   __  __
            #   / ____| |  \/  | |  \/  |
            #  | |  __  | \  / | | \  / |
            #  | | |_ | | |\/| | | |\/| |
            #  | |__| | | |  | | | |  | |
            #   \_____| |_|  |_| |_|  |_|
            #

            if type == "GMM" or type == 'all':
                trainGMM(DTR, LTR, file)

        # Calibration
        #  ██████  █████  ██      ██ ██████  ██████   █████  ████████ ██  ██████  ███    ██
        # ██      ██   ██ ██      ██ ██   ██ ██   ██ ██   ██    ██    ██ ██    ██ ████   ██
        # ██      ███████ ██      ██ ██████  ██████  ███████    ██    ██ ██    ██ ██ ██  ██
        # ██      ██   ██ ██      ██ ██   ██ ██   ██ ██   ██    ██    ██ ██    ██ ██  ██ ██
        #  ██████ ██   ██ ███████ ██ ██████  ██   ██ ██   ██    ██    ██  ██████  ██   ████
        #

        if mode == 'calib' or mode == 'all':
            file.write("Calibration" + '\n')

            # Logistic Regression
            #   _        _____
            #  | |      |  __ \
            #  | |      | |__) |
            #  | |      |  _  /
            #  |______| |_|  \_\
            #

            if type == "LR":
                prior, Cfp, Cfn = (0.5, 1, 1)
                lambda_ = 0.01
                #pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
                pi = 0.5
                QuadLogReg = LR_Classifier(lambda_, pi)

                LogObj = QuadraticLR_Calibration(1e-2, 0.5)

                params = {"K": 5, "pi": 0.5, "pca": 10, "costs": (1, 1), "logCalibration": LogObj, "znorm": False}

                DCF_effPrior, DCF_effPrior_min, scores, final_score, labels = CalibrationKFold(DTR,
                                                                                               LTR,
                                                                                               QuadLogReg,
                                                                                               params,
                                                                                               True)

                print(f"DCF_effPrior returned to main: {DCF_effPrior}\n\n")
                print(f"DCF_effPrior_min returned to main: {DCF_effPrior_min}\n\n")
                # print(f"scores returned to main: {scores}\n\n")
                # print(f"final_score returned to main: {final_score}\n\n")
                # print(f"labels returned to main: {labels}\n\n")

                post_prob = binary_posterior_prob(scores, prior, Cfn, Cfp)
                thresholds = np.sort(post_prob)
                lr_FPR, lr_TPR = ROC(thresholds, post_prob, labels)

                print(f"lr_FPR calculated: {lr_FPR}\n\n")
                print(f"lr_TPR calculated: {lr_TPR}\n\n")


            # Support Vector Machine
            #    _____  __      __  __  __
            #   / ____| \ \    / / |  \/  |
            #  | (___    \ \  / /  | \  / |
            #   \___ \    \ \/ /   | |\/| |
            #   ____) |    \  /    | |  | |
            #  |_____/      \/     |_|  |_|
            #

            if type == "SVM":
                prior, Cfp, Cfn = (0.5, 1, 1)
                K_svm = 0
                C = 10
                mode = "rbf"
                gamma = 1e-3
                pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
                SVMObj = SVMKernelClassifier(K_svm, C, pi_tilde, mode, gamma)

                LogObj = QuadraticLR_Calibration(1e-2, 0.5)

                params = {"K": 5,
                          "pi": 0.5,
                          "pca": 6,
                          "costs": (1, 1),
                          "logCalibration": LogObj,
                          "znorm": False
                          }

                DCF_effPrior, DCF_effPrior_min, svm_not_calibr_scores, svm_labels = CalibrationKFold(DTR, LTR, SVMObj,
                                                                                                     params,
                                                                                                     True)
                post_prob = binary_posterior_prob(svm_not_calibr_scores, prior, Cfn, Cfp)
                thresholds = np.sort(post_prob)
                svm_FPR, svm_TPR = ROC(thresholds, post_prob, svm_labels)
                print(f"svm_FPR: {svm_FPR}")
                print(f"svm_TPR: {svm_TPR}")

            # Gaussian Mixture Models
            #     _____   __  __   __  __
            #   / ____| |  \/  | |  \/  |
            #  | |  __  | \  / | | \  / |
            #  | | |_ | | |\/| | | |\/| |
            #  | |__| | | |  | | | |  | |
            #   \_____| |_|  |_| |_|  |_|
            #

            if type == "GMM" or type == "all":
                prior, Cfp, Cfn = (0.5, 1, 1)
                target_max_comp = 2
                not_target_max_comp = 8
                mode_target = "diag"
                mode_not_target = "diag"
                psi = 0.01
                alpha = 0.1
                pca = None
                #pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

                GMMObj = GMMClassifier(target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alpha, prior,
                                       Cfp,
                                       Cfn)

                LogObj = QuadraticLR_Calibration(1e-2, 0.5)

                params = {"K": 5,
                          "pi": 0.5,
                          "pca": pca,
                          "costs": (1, 1),
                          "logCalibration": LogObj,
                          "znorm": False}

                DCF_effPrior, DCF_effPrior_min, gmm_not_calibr_scores, gmm_labels = CalibrationKFold(DTR, LTR, GMMObj,
                                                                                                     params,
                                                                                                     True)
                post_prob = binary_posterior_prob(gmm_not_calibr_scores, prior, Cfn, Cfp)
                thresholds = np.sort(post_prob)
                gmm_FPR, gmm_TPR = ROC(thresholds, post_prob, gmm_labels)
                #print(f"gmm_FPR: {gmm_FPR}")
                #print(f"gmm_TPR: {gmm_TPR}")

        # Evaluation
        # ███████ ██    ██  █████  ██      ██    ██  █████  ████████ ██  ██████  ███    ██
        # ██      ██    ██ ██   ██ ██      ██    ██ ██   ██    ██    ██ ██    ██ ████   ██
        # █████   ██    ██ ███████ ██      ██    ██ ███████    ██    ██ ██    ██ ██ ██  ██
        # ██       ██  ██  ██   ██ ██      ██    ██ ██   ██    ██    ██ ██    ██ ██  ██ ██
        # ███████   ████   ██   ██ ███████  ██████  ██   ██    ██    ██  ██████  ██   ████
        #

        if mode == "eval" or mode == "all":
            file.write("Test" + '\n')

            # Logistic Regression - Quadratic
            #   _        _____
            #  | |      |  __ \
            #  | |      | |__) |
            #  | |      |  _  /
            #  | |____  | | \ \
            #  |______| |_|  \_\
            #

            if type == "QLR" or type == "all":
                prior, Cfp, Cfn = (0.5, 1, 1)
                pca = 11
                znorm = False
                lambda_ = 0.01
                pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
                qlr_classifier = QuadraticLR_Classifier(lambda_, pi_tilde)

                if znorm:
                    DTR, mu, sigma = normalize_zscore(DTR)
                    DTE, _, _ = normalize_zscore(DTE, mu, sigma)

                if pca is not None:  # PCA needed
                    DTR, P = PCA_impl(DTR, pca)
                    DTE = np.dot(P.T, DTE)

                qlr_classifier.train(DTR, LTR)
                lr_scores = qlr_classifier.compute_scores(DTE)
                lr_scores = np.array(lr_scores)
                DCF_min, _, _ = DCF_min_impl(lr_scores, LTE, prior, Cfp, Cfn)


                pred = [1 if x > 0 else 0 for x in lr_scores]
                acc, _ = accuracy(pred, LTE)
                print("QLR:")
                print("Accuracy: " + str(100 - acc))
                print("DCF min: " + str(DCF_min))

            # Support Vector Machine
            #    _____  __      __  __  __
            #   / ____| \ \    / / |  \/  |
            #  | (___    \ \  / /  | \  / |
            #   \___ \    \ \/ /   | |\/| |
            #   ____) |    \  /    | |  | |
            #  |_____/      \/     |_|  |_|
            #

            if type == "SVM" or type == "all":

                prior, Cfp, Cfn = (0.5, 1, 1)
                pca = 11
                znorm = False
                K_svm = 0
                C = 10
                mode = "rbf"
                gamma = 1e-3
                pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
                SVMObj = SVMKernelClassifier(K_svm, C, pi_tilde, mode, gamma)

                if znorm:
                    DTR, mu, sigma = normalize_zscore(DTR)
                    DTE, _, _ = normalize_zscore(DTE, mu, sigma)

                if pca is not None:  # PCA needed
                    DTR, P = PCA_impl(DTR, pca)
                    DTE = np.dot(P.T, DTE)

                SVMObj.train(DTR, LTR);
                svm_scores = SVMObj.compute_scores(DTE)
                svm_scores = np.array(svm_scores)
                DCF_min, _, _ = DCF_min_impl(svm_scores, LTE, prior, Cfp, Cfn)



                pred = [1 if x > 0 else 0 for x in svm_scores]
                acc, _ = accuracy(pred, LTE)
                print("SVM:")
                print("Accuracy: " + str(100 - acc))
                print("DCF min: " + str(DCF_min))

            # Gaussian Mixture Models
            #     _____   __  __   __  __
            #   / ____| |  \/  | |  \/  |
            #  | |  __  | \  / | | \  / |
            #  | | |_ | | |\/| | | |\/| |
            #  | |__| | | |  | | | |  | |
            #   \_____| |_|  |_| |_|  |_|
            #

            if type == "GMM" or type == "all":

                print("in test")

                prior, Cfp, Cfn = (0.5, 1, 1)
                pca = None
                znorm = False
                K_svm = 0
                mode_target = "diag"
                mode_not_target = "diag"
                not_target_max_comp = 8
                target_max_comp = 2
                psi = 0.01
                alfa = 0.1

                pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
                GMMObj = GMMClassifier(target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alfa, prior,
                                       Cfp,
                                       Cfn)

                if znorm:
                    DTR, mu, sigma = normalize_zscore(DTR)
                    DTE, _, _ = normalize_zscore(DTE, mu, sigma)

                if pca is not None:  # PCA needed
                    DTR, P = PCA_impl(DTR, pca)
                    DTE = np.dot(P.T, DTE)

                GMMObj.train(DTR, LTR)
                gmm_scores = GMMObj.compute_scores(DTE)
                gmm_scores = np.array(gmm_scores)
                DCF_min, _, _ = DCF_min_impl(gmm_scores, LTE, prior, Cfp, Cfn)


                pred = [1 if x > 0 else 0 for x in gmm_scores]
                acc, _ = accuracy(pred, LTE)
                print("GMM:")
                print("Accuracy: " + str(100 - acc))
                print("DCF min: " + str(DCF_min))

        # if mode == 'eval2' or mode == 'all':
        #     file.write("Evaluation" + '\n')
        #
        #     # Logistic Regression
        #     #   _        _____
        #     #  | |      |  __ \
        #     #  | |      | |__) |
        #     #  | |      |  _  /
        #     #  | |____  | | \ \
        #     #  |______| |_|  \_\
        #     #
        #
        #     if type == "QLR" or type == "all":
        #         prior, Cfp, Cfn = (0.5, 1, 1)
        #         lambda_ = 0.01
        #         pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        #         QuadLogReg = LR_Classifier(lambda_, pi_tilde)
        #
        #         LogObj = QuadraticLR_Calibration(1e-2, 0.5)
        #
        #         params = {"K": 5, "pi": 0.5, "pca": 11, "costs": (1, 1), "logCalibration": LogObj, "znorm": False}
        #
        #         DCF_effPrior, DCF_effPrior_min, lr_not_calibr_scores, lr_calibr_scores, lr_labels = CalibrationKFold(DTE,
        #                                                                                                              LTE,
        #                                                                                                              QuadLogReg,
        #                                                                                                              params,
        #                                                                                                              True)
        #
        #         post_prob = binary_posterior_prob(lr_not_calibr_scores, prior, Cfn, Cfp)
        #         thresholds = np.sort(post_prob)
        #         nc_lr_FPR, nc_lr_TPR = ROC(thresholds, post_prob, lr_labels)
        #         print(f"nc_lr_FPR: {nc_lr_FPR}")
        #         print(f"nc_lr_TPR: {nc_lr_TPR}")
        #
        #         post_prob = binary_posterior_prob(lr_calibr_scores, prior, Cfn, Cfp)
        #         thresholds = np.sort(post_prob)
        #         c_lr_FPR, c_lr_TPR = ROC(thresholds, post_prob, lr_labels)
        #         print(f"c_lr_FPR: {c_lr_FPR}")
        #         print(f"c_lr_TPR: {c_lr_TPR}")
        #
        #     # Support Vector Machine
        #     #    _____  __      __  __  __
        #     #   / ____| \ \    / / |  \/  |
        #     #  | (___    \ \  / /  | \  / |
        #     #   \___ \    \ \/ /   | |\/| |
        #     #   ____) |    \  /    | |  | |
        #     #  |_____/      \/     |_|  |_|
        #     #
        #
        #     if type == "SVM" or type == "all":
        #         prior, Cfp, Cfn = (0.5, 1, 1)
        #         K = 5
        #         pca = 11
        #         pi = 0.5
        #         znorm = False
        #         K_svm = 0
        #         C = 10
        #         gamma = 1e-3
        #
        #         pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        #         SVMObj = SVMKernelClassifier(K_svm, C, pi_tilde, mode, gamma)
        #
        #         LogObj = QuadraticLR_Calibration(1e-2, 0.5)
        #
        #         params = {"K": K,
        #                   "pi": pi,
        #                   "pca": pca,
        #                   "costs": (1, 1),
        #                   "logCalibration": LogObj,
        #                   "znorm": znorm}
        #
        #         DCF_effPrior, DCF_effPrior_min, svm_not_calibr_scores, svm_calibr_scores, svm_labels = CalibrationKFold(DTE,
        #                                                                                                                 LTE,
        #                                                                                                                 SVMObj,
        #                                                                                                                 params,
        #                                                                                                                 True)
        #
        #         post_prob = binary_posterior_prob(svm_not_calibr_scores, prior, Cfn, Cfp)
        #         thresholds = np.sort(post_prob)
        #         svm_FPR, svm_TPR = ROC(thresholds, post_prob, svm_labels)
        #         print(f"svm_FPR: {svm_FPR}")
        #         print(f"svm_TPR: {svm_TPR}")
        #
        #     # Gaussian Mixture Models
        #     #     _____   __  __   __  __
        #     #   / ____| |  \/  | |  \/  |
        #     #  | |  __  | \  / | | \  / |
        #     #  | | |_ | | |\/| | | |\/| |
        #     #  | |__| | | |  | | | |  | |
        #     #   \_____| |_|  |_| |_|  |_|
        #     #
        #
        #     if type == "GMM" or type == "all":
        #         prior, Cfp, Cfn = (0.5, 1, 1)
        #         pca = None
        #         znorm = False
        #         mode_target = "full"
        #         mode_not_target = "full"
        #         not_target_max_comp = 8
        #         target_max_comp = 2
        #         psi = 0.01
        #         alfa = 0.1
        #
        #         pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        #         GMMObj = GMMClassifier(target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alfa, prior,
        #                                Cfp,
        #                                Cfn)
        #         LogObj = QuadraticLR_Calibration(1e-2, 0.5)
        #
        #         params = {"K": 2,
        #                   "pi": 0.5,
        #                   "pca": pca,
        #                   "costs": (1, 1),
        #                   "logCalibration": LogObj,
        #                   "znorm": znorm}
        #
        #         if znorm:
        #             DTR, mu, sigma = normalize_zscore(DTR)
        #             DTE, _, _ = normalize_zscore(DTE, mu, sigma)
        #
        #         if pca is not None:  # PCA needed
        #             DTR, P = PCA_impl(DTR, pca)
        #             DTE = np.dot(P.T, DTE)
        #
        #         DCF_effPrior, DCF_effPrior_min, gmm_not_calibr_scores, gmm_calibr_scores, gmm_labels = CalibrationKFold(DTE,
        #                                                                                                                 LTE,
        #                                                                                                                 GMMObj,
        #                                                                                                                 params,
        #                                                                                                                 True)
        #
        #         post_prob = binary_posterior_prob(gmm_not_calibr_scores, prior, Cfn, Cfp)
        #         thresholds = np.sort(post_prob)
        #         gmm_FPR, gmm_TPR = ROC(thresholds, post_prob, gmm_labels)
        #         print(f"gmm_FPR: {gmm_FPR}")
        #         print(f"gmm_TPR: {gmm_FPR}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification.")
    parser.add_argument('--type', type=str, default='all', help='Percorso del file di input')
    parser.add_argument('--mode', type=str, default='all', help='Percorso del file di input')

    args = parser.parse_args()

    main(args.type, args.mode)
