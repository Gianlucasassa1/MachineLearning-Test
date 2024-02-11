import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from load import *
from Train.train import *
from Preprocess.feature_analysis import *
from Preprocess.PCA import *
from Preprocess.LDA import *
from Models.Gaussian.new_MVG_model import *
from Models.Gaussian.class_MVG import *
from Models.LogisticRegression.logreg import LR_Classifier, QuadraticLR_Classifier
from Models.SupportVector.svm import SVMClass
from Models.SupportVector.svm_kernel import SVMClass
from Models.MixtureModels.gmm import GMMClass
from Evaluation.evaluation import *
from Calibration.calibration import *


# Non ancora visti
# 
# 
# 
#


def main(type, mode):
    (DTR, LTR), (DTE, LTE) = loadTrainingAndTestData('Data/Train.txt', 'Data/Test.txt')
    DTR = DTR.T
    LTR = LTR.T
    DTE = DTE.T
    LTE = LTE.T

    ############################      DATA ANALISYS         ############################

    # plotSingle(DTR, LTR,10)
    # plotCross(DTR, LTR,10)
    # DC = centerData(DTR)              #delete the mean component from the data

    # plot the features
    # plotSingle(DC, LTR,10)
    # plotCross(DC, LTR,10)

    # mp = 6
    ## PCA implementation
    # DP,P = PCA_impl(DTR, mp)        #try with 2-dimension subplot
    # DTEP = np.dot(P.T,DTE)

    # plotCross(DP,LTR,m)        #plotting the data on a 2-D cartesian graph
    # plotSingle(DP, LTR,m)

    # ml = 1
    ## LDA implementation
    # DW,W = LDA_impl(DTR,LTR,ml)
    # DTEW = np.dot(W.T,DTE)
    # plotCross(DW,LTR,m)
    # plotSingle(DW, LTR, ml)

    ## LDA + PCA implementation
    # DPW,W = LDA_impl(DP,LTR,ml)
    # DTEPW = np.dot(W.T,DTEP)

    ## Pearson correlation

    # plot_features(DTR, LTR)

    ## PCA and variance plot
    # PCA_plot(DTR)

    #
    #
    #
    # TTTTTTTTTTTTTTTTTTTTTTTEEEEEEEEEEEEEEEEEEEEEE   SSSSSSSSSSSSSSS TTTTTTTTTTTTTTTTTTTTTTT
    # T:::::::::::::::::::::TE::::::::::::::::::::E SS:::::::::::::::ST:::::::::::::::::::::T
    # T:::::::::::::::::::::TE::::::::::::::::::::ES:::::SSSSSS::::::ST:::::::::::::::::::::T
    # T:::::TT:::::::TT:::::TEE::::::EEEEEEEEE::::ES:::::S     SSSSSSST:::::TT:::::::TT:::::T
    # TTTTTT  T:::::T  TTTTTT  E:::::E       EEEEEES:::::S            TTTTTT  T:::::T  TTTTTT
    #         T:::::T          E:::::E             S:::::S                    T:::::T
    #         T:::::T          E::::::EEEEEEEEEE    S::::SSSS                 T:::::T
    #         T:::::T          E:::::::::::::::E     SS::::::SSSSS            T:::::T
    #         T:::::T          E:::::::::::::::E       SSS::::::::SS          T:::::T
    #         T:::::T          E::::::EEEEEEEEEE          SSSSSS::::S         T:::::T
    #         T:::::T          E:::::E                         S:::::S        T:::::T
    #         T:::::T          E:::::E       EEEEEE            S:::::S        T:::::T
    #       TT:::::::TT      EE::::::EEEEEEEE:::::ESSSSSSS     S:::::S      TT:::::::TT
    #       T:::::::::T      E::::::::::::::::::::ES::::::SSSSSS:::::S      T:::::::::T
    #       T:::::::::T      E::::::::::::::::::::ES:::::::::::::::SS       T:::::::::T
    #       TTTTTTTTTTT      EEEEEEEEEEEEEEEEEEEEEE SSSSSSSSSSSSSSS         TTTTTTTTTTT
    #
    #

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #   _        _____
    #  | |      |  __ \
    #  | |      | |__) |
    #  | |      |  _  /
    #  | |____  | | \ \
    #  |______| |_|  \_\
    #

    if mode == 'train':
        if type == "MVG":
            prior, Cfp, Cfn = (0.5, 10, 1)
            params = {"K": 5, "pca": 6, "pi": 0.5, "costs": (1, 1)}

            # MVG
            K = 5
            MVG_obj = GaussianClassifier("MVG", prior, Cfp, Cfn)
            NB_obj = GaussianClassifier("NB", prior, Cfp, Cfn)
            TCG_obj = GaussianClassifier("TCG", prior, Cfp, Cfn)
            TCGNB_obj = GaussianClassifier("TCGNB", prior, Cfp, Cfn)

            mvg_pca6 = 0
            mvg_pca7 = 0
            mvg_pca8 = 0
            mvg_pca9 = 0
            mvg_pcaNone = 0

            for model in [MVG_obj, NB_obj, TCG_obj, TCGNB_obj]:
                for pca in [6, 7, 8, 9, None]:
                    params = {"K": 5,
                              "pca": pca,
                              "pi": 0.5,
                              "znorm": False,
                              "costs": (1, 1)}

                    min_DCF, scores, labels = train_KFold(DTR, LTR, model, params)
                    if pca == 6:
                        mvg_pca6 = min_DCF
                    if pca == 7:
                        mvg_pca7 = min_DCF
                    if pca == 8:
                        mvg_pca8 = min_DCF
                    if pca == 9:
                        mvg_pca9 = min_DCF
                    if pca == None:
                        mvg_pcaNone = min_DCF

                    print(f"{model.mode} min_DCF con K = {K} , pca = {pca}: {min_DCF} ")

                mvg_pca = np.array([mvg_pca6,
                                    mvg_pca7,
                                    mvg_pca8,
                                    mvg_pca9,
                                    mvg_pcaNone])

                # Definisci il percorso della nuova cartella
                folder_path = 'Images/Gaussian'

                # Crea la nuova cartella
                os.makedirs(folder_path, exist_ok=True)

                plt.xlabel("PCA dimensions")
                plt.ylabel("DCF_min")
                # plt.legend()
                plt.title(model.mode)
                path = "Images/Gaussian/" + str(model.mode)
                plt.plot(np.linspace(6, 10, 5), mvg_pca)
                plt.savefig(path)
                #  plt.show()

        # Logistic Regression

        #
        #    _        _____
        #  | |      |  __ \
        #  | |      | |__) |
        #  | |      |  _  /
        #  | |____  | | \ \
        #  |______| |_|  \_\
        #
        #

        if type == "LR":
            K = 5
            # piT = 0.1
            lambda_plot = np.logspace(-7, 2, num=9)

            for piT in [0.1, 0.5, 0.9]:
                lr_pca6 = []
                lr_pca7 = []
                lr_pca8 = []
                lr_pca9 = []
                lr_pcaNone = []
                for zscore in [True, False]:
                    for lambda_temp in np.logspace(-6, 2, num=9):
                        # we saw that piT=0.1 is the best value
                        for pca in [6, 7, 8, 9, None]:
                            params = {"K": 5,
                                      "pca": pca,
                                      "pi": 0.5,
                                      "costs": (1, 1),
                                      "znorm": zscore}
                            logObj = LR_Classifier(lambda_temp, piT)
                            min_DCF, scores, labels = train_KFold(DTR, LTR, logObj, params)
                            print(
                                f"Log Reg min_DCF con K = {K} , pca = {pca}, l = {lambda_temp}, piT = {piT}, znorm = {zscore}: {min_DCF} ")

                            if pca == 6:
                                lr_pca6.append(min_DCF)
                            if pca == 7:
                                lr_pca7.append(min_DCF)
                            if pca == 8:
                                lr_pca8.append(min_DCF)
                            if pca == 9:
                                lr_pca9.append(min_DCF)
                            if pca == None:
                                lr_pcaNone.append(min_DCF)

                    plt.semilogx(lambda_plot, lr_pca6, label="PCA 6")
                    plt.semilogx(lambda_plot, lr_pca7, label="PCA 7")
                    plt.semilogx(lambda_plot, lr_pca8, label="PCA 8")
                    plt.semilogx(lambda_plot, lr_pca9, label="PCA 9")
                    plt.semilogx(lambda_plot, lr_pcaNone, label="No PCA")

                    plt.xlabel("Lambda")
                    plt.ylabel("DCF_min")
                    plt.legend()

                    #   TODO: can be optimized
                    # Definisci il percorso della nuova cartella
                    folder_path_1 = 'Images/LogisticRegression/MinimumLambda'
                    folder_path_2 = 'Images/LogisticRegression/MediumLambda'
                    folder_path_3 = 'Images/LogisticRegression/MaximumLambda'

                    # Crea la nuova cartella
                    os.makedirs(folder_path_1, exist_ok=True)
                    os.makedirs(folder_path_2, exist_ok=True)
                    os.makedirs(folder_path_3, exist_ok=True)

                    if piT == 0.1:
                        path = folder_path_1
                    if piT == 0.5:
                        path = folder_path_2
                    if piT == 0.9:
                        path = folder_path_3

                    plt.title(piT)
                    plt.savefig(path)
                    #  plt.show()

        # Logistic Regression - Quadratic
        #
        #   _        _____                  ____                        _                  _     _
        #  | |      |  __ \                / __ \                      | |                | |   (_)
        #  | |      | |__) |    ______    | |  | |  _   _    __ _    __| |  _ __    __ _  | |_   _    ___
        #  | |      |  _  /    |______|   | |  | | | | | |  / _` |  / _` | | '__|  / _` | | __| | |  / __|
        #  | |____  | | \ \               | |__| | | |_| | | (_| | | (_| | | |    | (_| | | |_  | | | (__
        #  |______| |_|  \_\               \___\_\  \__,_|  \__,_|  \__,_| |_|     \__,_|  \__| |_|  \___|
        #
        #

        if type == "QLR":

            K = 5

            # Results to be plotted
            PCA10_plot = []
            PCA11_plot = []
            PCA12_plot = []
            NoPCA_plot = []

            for pi in [0.1, 0.5, 0.9]:
                for piT in [0.1, 0.5, 0.9]:
                    PCA10_qlr = []
                    PCA11_qlr = []
                    PCA12_qlr = []
                    NoPCA_qlr = []
                    for zscore in [False, True]:
                        for lambda_ in np.logspace(-3, 2, num=6):  # TODO: [0.01] quanti ne facciamo?
                            for pca in [10, 11, 12, None]:
                                params = {"K": 5, "pca": pca, "pi": pi, "costs": (1, 1),
                                          "znorm": zscore}  # TODO: da capire pi: pi

                                qlr_classifier = QuadraticLR_Classifier(lambda_, piT)

                                min_DCF, scores, labels = train_KFold(DTR, LTR, qlr_classifier, params)

                                print(
                                    f"Log Reg min_DCF con K = {K} , pca = {pca}, l = {lambda_} , piT = {piT}, pi = {pi} zscore={zscore}: {min_DCF} ")
                                if pca == 10:
                                    PCA10_qlr.append(min_DCF)
                                if pca == 11:
                                    PCA11_qlr.append(min_DCF)
                                if pca == 12:
                                    PCA12_qlr.append(min_DCF)
                                if pca is None:
                                    NoPCA_qlr.append(min_DCF)

                    PCA10_plot.append(
                        (f"Log Reg min_DCF con K = {K} , pca = 6, piT = {piT} pi={pi} zscore={zscore}", PCA10_qlr))
                    PCA11_plot.append((f"Log Reg min_DCF con K = {K} , pca = 7, piT = {piT} pi={pi}", PCA11_qlr))
                    PCA12_plot.append((f"Log Reg min_DCF con K = {K} , pca = 8, piT = {piT} pi={pi}", PCA12_qlr))
                    NoPCA_plot.append((f"Log Reg min_DCF con K = {K} , pca = no, piT = {piT} pi={pi}", NoPCA_qlr))

            lambda_plot = np.logspace(-4, 2, num=7)

            plt.semilogx(lambda_plot, PCA10_qlr, label="PCA 10")
            plt.semilogx(lambda_plot, PCA11_qlr, label="PCA 11")
            plt.semilogx(lambda_plot, PCA12_qlr, label="PCA 12")
            plt.semilogx(lambda_plot, NoPCA_qlr, label="No PCA")

            # TODO: can be optimized
            # Definisci il percorso della nuova cartella
            folder_path_1 = 'Images/QuadraticLogisticRegression/MinimumPit'
            folder_path_2 = 'Images/QuadraticLogisticRegression/OneThirdPit'
            folder_path_3 = 'Images/QuadraticLogisticRegression/MediumPit'
            folder_path_4 = 'Images/QuadraticLogisticRegression/MaximumPit'

            # Crea la nuova cartella
            os.makedirs(folder_path_1, exist_ok=True)
            os.makedirs(folder_path_2, exist_ok=True)
            os.makedirs(folder_path_3, exist_ok=True)
            os.makedirs(folder_path_4, exist_ok=True)
            if piT == 0.1:
                path = folder_path_1
            if piT == 0.33:
                path = folder_path_2
            if piT == 0.5:
                path = folder_path_3
            if piT == 0.9:
                path = folder_path_4

            plt.xlabel("Lambda")
            plt.ylabel("DCF_min")
            plt.legend()
            plt.title(0.33)
            plt.savefig(path)

        #
        #    _____  __      __  __  __
        #   / ____| \ \    / / |  \/  |
        #  | (___    \ \  / /  | \  / |
        #   \___ \    \ \/ /   | |\/| |
        #   ____) |    \  /    | |  | |
        #  |_____/      \/     |_|  |_|
        #
        #
        #
        #

        if type == "SVM":
            K = 5
            K_svm = 1  # else 0
            pi = 0.5
            for piT in [0.1, 0.5, 0.9]:
                svm_pca6_no_zscore = []
                svm_pca6_zscore = []
                svm_pca6 = []
                svm_pca7 = []
                svm_pca8 = []
                svm_pca9 = []
                svm_pcaNone = []
                C_values = np.logspace(-5, 2, num=8)
                for C in np.logspace(-5, 2, num=8):  # TODO da capire quanti gìvgoliamo prenderne
                    for zscore in [True, False]:
                        # we saw that piT=0.1 is the best value
                        for pca in [10, 11, 12, None]:
                            params = {"K": 5, "pca": pca, "pi": pi, "costs": (1, 1), "znorm": zscore}

                            SVMObj = SVMClass(K_svm, C, piT)
                            min_DCF, scores, labels = train_KFold(DTR, LTR, SVMObj, params)

                            if min_DCF > 1:
                                min_DCF = 1
                            print(
                                f"SVM min_DCF con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}, pi={pi}, znorm={zscore}: {min_DCF}")

                            if pca == 6:
                                if zscore == False:
                                    svm_pca6_no_zscore.append(min_DCF)
                                else:
                                    svm_pca6_zscore.append(min_DCF)
                            if pca == 7:
                                svm_pca7.append(min_DCF)
                            if pca == 8:
                                svm_pca8.append(min_DCF)
                            if pca == 9:
                                svm_pca9.append(min_DCF)
                            if pca == None:
                                svm_pcaNone.append(min_DCF)

                plt.semilogx(C_values, svm_pca6_no_zscore, label="PCA 6")
                plt.semilogx(C_values, svm_pca6_zscore, label="PCA 6 - Z_NORM")
                # plt.semilogx(C_values,svm_pca8, label = "PCA 8")
                # plt.semilogx(C_values,svm_pca9, label = "PCA 9")
                # plt.semilogx(C_values,svm_pcaNone, label = "No PCA")

                #  TODO: can be optimized
                # Definisci il percorso della nuova cartella
                folder_path_1 = 'Images/SupportVector/MinimumPit'
                folder_path_2 = 'Images/SupportVector/OneThirdPit'
                folder_path_3 = 'Images/SupportVector/MediumPit'
                folder_path_4 = 'Images/SupportVector/MaximumPit'

                # Crea la nuova cartella
                os.makedirs(folder_path_1, exist_ok=True)
                os.makedirs(folder_path_2, exist_ok=True)
                os.makedirs(folder_path_3, exist_ok=True)
                os.makedirs(folder_path_4, exist_ok=True)

                plt.xlabel("C")
                plt.ylabel("DCF_min")
                plt.legend()
                if piT == 0.1:
                    path = folder_path_1
                if piT == 0.33:
                    path = folder_path_2
                if piT == 0.5:
                    path = folder_path_3
                if piT == 0.9:
                    path = folder_path_4

                path = path + "Znorm"

                plt.title(piT)
                plt.savefig(path)
                #  plt.show()

                ############ KERNEL SVM  ############

        #
        #    _____  __      __  __  __                _  __                               _
        #   / ____| \ \    / / |  \/  |              | |/ /                              | |
        #  | (___    \ \  / /  | \  / |    ______    | ' /    ___   _ __   _ __     ___  | |
        #   \___ \    \ \/ /   | |\/| |   |______|   |  <    / _ \ | '__| | '_ \   / _ \ | |
        #   ____) |    \  /    | |  | |              | . \  |  __/ | |    | | | | |  __/ | |
        #  |_____/      \/     |_|  |_|              |_|\_\  \___| |_|    |_| |_|  \___| |_|
        #
        #
        #

        if type == "KSVM":
            K = 5
            piT = 0.1
            poly_svm_pca6 = {}
            poly_svm_pca8 = {}
            poly_svm_pcaNone = {}
            rbf_svm_pca6 = {}
            rbf_svm_pca8 = {}
            rbf_svm_pcaNone = {}
            for piT in [0.1]:
                for kernel in ["poly"]:
                    if kernel == "poly":
                        ci = [0, 1]
                        string = "d=2 c= "
                    else:
                        ci = [0.01, 0.001, 0.0001]
                        string = "gamma= "
                    for value in ci:
                        svm_pca6 = []
                        svm_pca6_noznorm = []
                        svm_pcaNone = []
                        svm_pcaNone_noznorm = []
                        # svm_pcaNone = []
                        C_values = np.logspace(-3, -1, num=3)
                        for C in np.logspace(-3, -1, num=3):
                            for K_svm in [1]:  # else 0
                                # we saw that piT=0.1 is the best value
                                for pca in [6, None]:
                                    for znorm in [False, True]:
                                        params = {"K": 5,
                                                  "pca": pca,
                                                  "pi": 0.5,
                                                  "costs": (1, 1),
                                                  "znorm": znorm}
                                        SVMObj = SVMClass(K_svm, C, piT, kernel, value)
                                        min_DCF, scores, labels = train_KFold(DTR, LTR, SVMObj, params)
                                        if min_DCF > 1:
                                            min_DCF = 1

                                        print(
                                            f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} znorm: {znorm}")

                                        if pca == 6:
                                            if znorm == True:
                                                svm_pca6.append(min_DCF)
                                                if kernel == "poly":
                                                    poly_svm_pca6.setdefault(
                                                        f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",
                                                        min_DCF)
                                            #     else:
                                            #         rbf_svm_pca6.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",min_DCF)
                                            else:
                                                svm_pca6_noznorm.append(min_DCF)
                                                if kernel == "poly":
                                                    poly_svm_pca6.setdefault(
                                                        f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",
                                                        min_DCF)
                                            # else:
                                            #     rbf_svm_pca6.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",min_DCF)

                                        # if pca == 7:
                                        #     svm_pca7.append(min_DCF)
                                        # if pca == 8:
                                        #     svm_pca8.append(min_DCF)
                                        #     if kernel=="poly" :
                                        #         poly_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                        #     else:
                                        #         rbf_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                        # # if pca == 9:
                                        # #     svm_pca9.append(min_DCF)
                                        if pca == None:
                                            if znorm == True:
                                                svm_pcaNone.append(min_DCF)
                                                if kernel == "poly":
                                                    poly_svm_pcaNone.setdefault(
                                                        f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",
                                                        min_DCF)
                                                # else:
                                                #     rbf_svm_pcaNone.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                            else:
                                                svm_pcaNone_noznorm.append(min_DCF)
                                                if kernel == "poly":
                                                    poly_svm_pcaNone.setdefault(
                                                        f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",
                                                        min_DCF)

                        plt.semilogx(C_values, svm_pca6, label="PCA 6")
                        # plt.semilogx(C_values,svm_pca7, label = "PCA 7")
                        plt.semilogx(C_values, svm_pca6_noznorm, label="PCA 6 No Znorm")
                        # plt.semilogx(C_values,svm_pca9, label = "PCA 9")
                        plt.semilogx(C_values, svm_pcaNone, label="No PCA")
                        plt.semilogx(C_values, svm_pcaNone_noznorm, label="No PCA No Znorm")

                        plt.xlabel("C")
                        plt.ylabel("DCF_min")
                        plt.legend()

                        #  TODO: can be optimized
                        # Definisci il percorso della nuova cartella
                        folder_path_1 = 'Images/SupportVector/MinimumPit'
                        folder_path_2 = 'Images/SupportVector/OneThirdPit'
                        folder_path_3 = 'Images/SupportVector/MediumPit'
                        folder_path_4 = 'Images/SupportVector/MaximumPit'

                        # Crea la nuova cartella
                        os.makedirs(folder_path_1, exist_ok=True)
                        os.makedirs(folder_path_2, exist_ok=True)
                        os.makedirs(folder_path_3, exist_ok=True)
                        os.makedirs(folder_path_4, exist_ok=True)

                        if piT == 0.1:
                            path = folder_path_1
                        if piT == 0.33:
                            path = folder_path_2
                        if piT == 0.5:
                            path = folder_path_3
                        if piT == 0.9:
                            path = folder_path_4

                        if kernel == "rbf":
                            gamma = " gamma : " + str(value)
                        else:
                            gamma = " ci: " + str(value)

                        title = str(piT) + " " + str(kernel) + " " + str(gamma)
                        plt.title(title)
                        plt.savefig(path)
                        #  plt.show()

                        ###############   GMM   ###############

        #
        #     _____   __  __   __  __
        #   / ____| |  \/  | |  \/  |
        #  | |  __  | \  / | | \  / |
        #  | | |_ | | |\/| | | |\/| |
        #  | |__| | | |  | | | |  | |
        #   \_____| |_|  |_| |_|  |_|
        #
        #
        #

        if type == "GMM":
            K = 5
            prior = 0.5
            Cfp = 10
            Cfn = 1
            gmm_pca6_glob = {}
            gmm_pca7_glob = {}
            gmm_pca8_glob = {}
            gmm_pca9_glob = {}
            gmm_pcaNone_glob = {}

            for mode_target in ["diag", "tied"]:
                for mode_not_target in ["full", "diag", "tied"]:
                    gmm_pca6 = []
                    gmm_pca7 = []
                    gmm_pca8 = []
                    gmm_pca9 = []
                    gmm_pcaNone = []
                    for pca in [6, 8, None]:
                        for t_max_n in [1, 2]:
                            gmm_tmp = []
                            for nt_max_n in [2, 4, 8]:
                                for znorm in [False]:
                                    alfa = 0.1
                                    psi = 0.01
                                    params = {"K": 5,
                                              "pca": pca,
                                              "pi": 0.5,
                                              "costs": (1, 1),
                                              "znorm": znorm}

                                    GMMObj = GMMClass(t_max_n, nt_max_n, mode_target, mode_not_target, psi, alfa, prior,
                                                      Cfp, Cfn)
                                    min_DCF, scores, labels = train_KFold(DTR, LTR, GMMObj, params)
                                    # if min_DCF > 1:
                                    #     min_DCF = 1

                                    print(
                                        f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}")

                                    # un vettore che si annulla ad ogni nuovo t_max_n
                                    gmm_tmp.append(min_DCF)
                                    if pca == 6:
                                        # if znorm==True:

                                        gmm_pca6.append(min_DCF)
                                        gmm_pca6_glob.setdefault((
                                            f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                            min_DCF))

                                        # else:
                                        #     gmm_pca6_noznorm.append(min_DCF)

                                    # if pca == 7:
                                    #     gmm_pca7.append(min_DCF)
                                    #     gmm_pca7_glob.setdefault((f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",min_DCF))

                                    if pca == 8:
                                        gmm_pca8.append(min_DCF)
                                        gmm_pca8_glob.setdefault((
                                            f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                            min_DCF))

                                    #     if kernel=="poly" :
                                    #         poly_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                    #     else:
                                    #         rbf_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                    # if pca == 9:
                                    #     gmm_pca9.append(min_DCF)
                                    #     gmm_pca9_glob.setdefault((f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",min_DCF))

                                    if pca == None:
                                        # if znorm==True:
                                        gmm_pcaNone.append(min_DCF)
                                        gmm_pcaNone_glob.setdefault((
                                            f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                            min_DCF))

                                        # else:
                                        #     gmm_pcaNone_noznorm.append(min_DCF)

                            fig = plt.figure()
                            plt.plot([2, 4, 8], gmm_tmp)
                            plt.xlabel("nt_max_n")
                            plt.ylabel("DCF_min")
                            titolo = f"mode_target: {mode_target}, mode_non_target:{mode_not_target} PCA: {pca}, t_max_n: {t_max_n}"
                            plt.title(titolo)
                            # plt.show()

                        # Creazione del grafico
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')

                        # Aggiunta dei dati al grafico
                        ax.scatter(t_max_n, min_DCF, nt_max_n)

                        # Impostazione delle etichette degli assi
                        ax.set_xlabel('t_max_n')
                        ax.set_ylabel('DCF_min')
                        ax.set_zlabel('nt_max_n')
                        name_graph = "GMM " + mode_target + " " + mode_not_target
                        plt.title(name_graph)
                        # plt.show()
                        plt.semilogx(C_values, svm_pca6, label="PCA 6")
                        # plt.semilogx(C_values,svm_pca7, label = "PCA 7")
                        plt.semilogx(C_values, svm_pca6_noznorm, label="PCA 6 No Znorm")
                        # plt.semilogx(C_values,svm_pca9, label = "PCA 9")
                        plt.semilogx(C_values, svm_pcaNone, label="No PCA")
                        plt.semilogx(C_values, svm_pcaNone_noznorm, label="No PCA No Znorm")

                        plt.xlabel("C")
                        plt.ylabel("DCF_min")
                        plt.legend()

                        #  TODO: can be optimized
                        # Definisci il percorso della nuova cartella
                        folder_path_1 = 'Images/MixtureModels/MinimumPit'
                        folder_path_2 = 'Images/MixtureModels/OneThirdPit'
                        folder_path_3 = 'Images/MixtureModels/MediumPit'
                        folder_path_4 = 'Images/MixtureModels/MaximumPit'

                        # Crea la nuova cartella
                        os.makedirs(folder_path_1, exist_ok=True)
                        os.makedirs(folder_path_2, exist_ok=True)
                        os.makedirs(folder_path_3, exist_ok=True)
                        os.makedirs(folder_path_4, exist_ok=True)

                        if piT == 0.1:
                            path = folder_path_1
                        if piT == 0.33:
                            path = folder_path_2
                        if piT == 0.5:
                            path = folder_path_3
                        if piT == 0.9:
                            path = folder_path_4

                        title = str(piT) + " " + str(kernel) + " " + str(gamma)

                        plt.savefig(path)
                        #  plt.show()

    #
    #
    #
    #
    #
    #         CCCCCCCCCCCCC               AAA               LLLLLLLLLLL             IIIIIIIIIIBBBBBBBBBBBBBBBBB   RRRRRRRRRRRRRRRRR                  AAA               TTTTTTTTTTTTTTTTTTTTTTTIIIIIIIIII     OOOOOOOOO     NNNNNNNN        NNNNNNNN
    #      CCC::::::::::::C              A:::A              L:::::::::L             I::::::::IB::::::::::::::::B  R::::::::::::::::R                A:::A              T:::::::::::::::::::::TI::::::::I   OO:::::::::OO   N:::::::N       N::::::N
    #    CC:::::::::::::::C             A:::::A             L:::::::::L             I::::::::IB::::::BBBBBB:::::B R::::::RRRRRR:::::R              A:::::A             T:::::::::::::::::::::TI::::::::I OO:::::::::::::OO N::::::::N      N::::::N
    #   C:::::CCCCCCCC::::C            A:::::::A            LL:::::::LL             II::::::IIBB:::::B     B:::::BRR:::::R     R:::::R            A:::::::A            T:::::TT:::::::TT:::::TII::::::IIO:::::::OOO:::::::ON:::::::::N     N::::::N
    #  C:::::C       CCCCCC           A:::::::::A             L:::::L                 I::::I    B::::B     B:::::B  R::::R     R:::::R           A:::::::::A           TTTTTT  T:::::T  TTTTTT  I::::I  O::::::O   O::::::ON::::::::::N    N::::::N
    # C:::::C                        A:::::A:::::A            L:::::L                 I::::I    B::::B     B:::::B  R::::R     R:::::R          A:::::A:::::A                  T:::::T          I::::I  O:::::O     O:::::ON:::::::::::N   N::::::N
    # C:::::C                       A:::::A A:::::A           L:::::L                 I::::I    B::::BBBBBB:::::B   R::::RRRRRR:::::R          A:::::A A:::::A                 T:::::T          I::::I  O:::::O     O:::::ON:::::::N::::N  N::::::N
    # C:::::C                      A:::::A   A:::::A          L:::::L                 I::::I    B:::::::::::::BB    R:::::::::::::RR          A:::::A   A:::::A                T:::::T          I::::I  O:::::O     O:::::ON::::::N N::::N N::::::N
    # C:::::C                     A:::::A     A:::::A         L:::::L                 I::::I    B::::BBBBBB:::::B   R::::RRRRRR:::::R        A:::::A     A:::::A               T:::::T          I::::I  O:::::O     O:::::ON::::::N  N::::N:::::::N
    # C:::::C                    A:::::AAAAAAAAA:::::A        L:::::L                 I::::I    B::::B     B:::::B  R::::R     R:::::R      A:::::AAAAAAAAA:::::A              T:::::T          I::::I  O:::::O     O:::::ON::::::N   N:::::::::::N
    # C:::::C                   A:::::::::::::::::::::A       L:::::L                 I::::I    B::::B     B:::::B  R::::R     R:::::R     A:::::::::::::::::::::A             T:::::T          I::::I  O:::::O     O:::::ON::::::N    N::::::::::N
    #  C:::::C       CCCCCC    A:::::AAAAAAAAAAAAA:::::A      L:::::L         LLLLLL  I::::I    B::::B     B:::::B  R::::R     R:::::R    A:::::AAAAAAAAAAAAA:::::A            T:::::T          I::::I  O::::::O   O::::::ON::::::N     N:::::::::N
    #   C:::::CCCCCCCC::::C   A:::::A             A:::::A   LL:::::::LLLLLLLLL:::::LII::::::IIBB:::::BBBBBB::::::BRR:::::R     R:::::R   A:::::A             A:::::A         TT:::::::TT      II::::::IIO:::::::OOO:::::::ON::::::N      N::::::::N
    #    CC:::::::::::::::C  A:::::A               A:::::A  L::::::::::::::::::::::LI::::::::IB:::::::::::::::::B R::::::R     R:::::R  A:::::A               A:::::A        T:::::::::T      I::::::::I OO:::::::::::::OO N::::::N       N:::::::N
    #      CCC::::::::::::C A:::::A                 A:::::A L::::::::::::::::::::::LI::::::::IB::::::::::::::::B  R::::::R     R:::::R A:::::A                 A:::::A       T:::::::::T      I::::::::I   OO:::::::::OO   N::::::N        N::::::N
    #         CCCCCCCCCCCCCAAAAAAA                   AAAAAAALLLLLLLLLLLLLLLLLLLLLLLLIIIIIIIIIIBBBBBBBBBBBBBBBBB   RRRRRRRR     RRRRRRRAAAAAAA                   AAAAAAA      TTTTTTTTTTT      IIIIIIIIII     OOOOOOOOO     NNNNNNNN         NNNNNNN
    #
    #
    #
    #
    #
    #
    #
    #
    #

    #####################                       COST AND CALIBRATION         #######################

    ## ROC and Bayes error ###

    if mode == 'eval':

        # post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
        # thresholds = np.sort(post_prob)
        # ROC_plot(thresholds,post_prob,label)

        ##### CALIBRATE USING A LR #####

        ### BEST QUAD LOG REG ###
        prior, Cfp, Cfn = (0.5, 10, 1)

        lambda_ = 0.01
        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        QuadLogReg = QuadraticLR_Classifier(lambda_, pi_tilde)

        LogObj = LRCalibrClass(1e-2, 0.5)

        params = {"K": 5,
                  "pi": 0.5,
                  "pca": 6,
                  "costs": (1, 1),
                  "logCalibration": LogObj,
                  "znorm": False}

        DCF_effPrior, DCF_effPrior_min, lr_not_calibr_scores, lr_labels = kfold_calib(DTR, LTR, QuadLogReg, params,
                                                                                      True)

        post_prob = binary_posterior_prob(scores, pi, cfn, cfp)
        thresholds = np.sort(post_prob)
        lr_FPR, lr_TPR = ROC_plot(thresholds, post_prob, labels)

        # DCF_effPrior = {-3.0: 0.4199709704927474, -2.7: 0.34309361470383404, -2.4: 0.2727180609497317, -2.1: 0.2753070413616317, -1.8: 0.24085851248468823, -1.5: 0.20404200016447085, -1.2000000000000002: 0.17805365648606708, -0.8999999999999999: 0.14919130695948832, -0.6000000000000001: 0.12082609601357178, -0.30000000000000027: 0.1060755788345864, 0.0: 0.08936475409836064, 0.2999999999999998: 0.1030707357997773, 0.5999999999999996: 0.12293001919009058, 0.8999999999999999: 0.1534975411030923, 1.2000000000000002: 0.16892171827903718, 1.5: 0.18818060176632873, 1.7999999999999998: 0.21035361594185878, 2.0999999999999996: 0.23866710148906534, 2.3999999999999995: 0.2505341966413051, 2.7: 0.3002114380345536, 3.0: 0.30417322247834744}
        # DCF_effPrior_min = {-3.0: 0.34747097049274733, -2.7: 0.303093614703834, -2.4: 0.27021806094973166, -2.1: 0.24496811723012926, -1.8: 0.22553774721428277, -1.5: 0.20066437515419147, -1.2000000000000002: 0.17103341839971403, -0.8999999999999999: 0.14173128845332347, -0.6000000000000001: 0.11957609601357178, -0.30000000000000027: 0.10196073215102967, 0.0: 0.08590163934426229, 0.2999999999999998: 0.09951215766990303, 0.5999999999999996: 0.11336952206860416, 0.8999999999999999: 0.13169719600313934, 1.2000000000000002: 0.15578980831431538, 1.5: 0.18091080531999174, 1.7999999999999998: 0.1985503372533342, 2.0999999999999996: 0.22236121479507462, 2.3999999999999995: 0.2439768195921248, 2.7: 0.26808029049356996, 3.0: 0.29827158313408514}

        #                                             ### BEST SVM  ###

        prior, Cfp, Cfn = (0.5, 10, 1)
        K_svm = 0
        C = 10
        mode = "rbf"
        gamma = 1e-3
        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        SVMObj = SVMClass(K_svm, C, pi_tilde, mode, gamma)

        LogObj = LRCalibrClass(1e-2, 0.5)

        params = {"K": 5,
                  "pi": 0.5,
                  "pca": 6,
                  "costs": (1, 1),
                  "logCalibration": LogObj,
                  "znorm": False}

        DCF_effPrior, DCF_effPrior_min, svm_not_calibr_scores, svm_labels = kfold_calib(DTR, LTR, SVMObj, params, True)
        # DCF_effPrior = {-3.0: 0.4461292827246323, -2.7: 0.36958641357276983, -2.4: 0.3281529980919774, -2.1: 0.28424244481231825, -1.8: 0.23749058470917883, -1.5: 0.21529200016447086, -1.2000000000000002: 0.17587653063509226, -0.8999999999999999: 0.148828452460369, -0.6000000000000001: 0.12810542401459604, -0.30000000000000027: 0.11138649873324608, 0.0: 0.09407786885245903, 0.2999999999999998: 0.10766089973420354, 0.5999999999999996: 0.12362062382845856, 0.8999999999999999: 0.1455423210481037, 1.2000000000000002: 0.13747204224251325, 1.5: 0.1578466222633834, 1.7999999999999998: 0.17744027635220427, 2.0999999999999996: 0.1905355812431686, 2.3999999999999995: 0.21705765900039217, 2.7: 0.25663245154133696, 3.0: 0.2721311475409836}
        # DCF_effPrior_min = {-3.0: 0.4048792827246324, -2.7: 0.347644022621283, -2.4: 0.2970662476149718, -2.1: 0.25959731032875605, -1.8: 0.23183963887754683, -1.5: 0.20005250020558862, -1.2000000000000002: 0.16857491499581492, -0.8999999999999999: 0.1420541614586076, -0.6000000000000001: 0.1222967680125475, -0.30000000000000027: 0.10539981225237001, 0.0: 0.0901844262295082, 0.2999999999999998: 0.10391040461562957, 0.5999999999999996: 0.11571690443599221, 0.8999999999999999: 0.12947088010046864, 1.2000000000000002: 0.13484909142284116, 1.5: 0.1421089173453506, 1.7999999999999998: 0.15190865730831868, 2.0999999999999996: 0.16513692260928553, 2.3999999999999995: 0.1816478229348184, 2.7: 0.18646851711510745, 3.0: 0.19297577361300108}
        post_prob = binary_posterior_prob(svm_not_calibr_scores, prior, Cfn, Cfp)
        thresholds = np.sort(post_prob)
        svm_FPR, svm_TPR = ROC_plot(thresholds, post_prob, svm_labels)

        #                                             ### BEST GMM  ###
        prior, Cfp, Cfn = (0.5, 10, 1)
        target_max_comp = 2
        not_target_max_comp = 8
        mode_target = "diag"
        mode_not_target = "diag"
        psi = 0.01
        alpha = 0.1
        pca = None
        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

        GMMObj = GMMClass(target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alpha, prior, Cfp,
                          Cfn)

        LogObj = LRCalibrClass(1e-2, 0.5)

        params = {"K": 5,
                  "pi": 0.5,
                  "pca": pca,
                  "costs": (1, 1),
                  "logCalibration": LogObj,
                  "znorm": False}

        DCF_effPrior, DCF_effPrior_min, gmm_not_calibr_scores, gmm_labels = kfold_calib(DTR, LTR, GMMObj, params, True)
        # DCF_effPrior={-3.0: 0.35362928272463234, -2.7: 0.3008364135727698, -2.4: 0.2709246857112289, -2.1: 0.22478271377944267, -1.8: 0.19977360276530154, -1.5: 0.18941437515419146, -1.2000000000000002: 0.1633350340389914, -0.8999999999999999: 0.14366701595772693, -0.6000000000000001: 0.11327092801382786, -0.30000000000000027: 0.10071073215102967, 0.0: 0.08311475409836065, 0.2999999999999998: 0.09576166255132909, 0.5999999999999996: 0.1056401373433876, 0.8999999999999999: 0.11709823130299822, 1.2000000000000002: 0.12653164324963864, 1.5: 0.1441328121523433, 1.7999999999999998: 0.1456580733876385, 2.0999999999999996: 0.14566132642076338, 2.3999999999999995: 0.167230072099145, 2.7: 0.18801900144332961, 3.0: 0.19297577361300108}
        # DCF_effPrior_min = {-3.0: 0.3293001266086899, -2.7: 0.28833641357276985, -2.4: 0.24663131047272605, -2.1: 0.21478271377944266, -1.8: 0.19118869304591482, -1.5: 0.17370981258737517, -1.2000000000000002: 0.15821964287047052, -0.8999999999999999: 0.13386843395420417, -0.6000000000000001: 0.11207609601357178, -0.30000000000000027: 0.09435919898659688, 0.0: 0.08002049180327869, 0.2999999999999998: 0.08920428550214876, 0.5999999999999996: 0.10143014266159533, 0.8999999999999999: 0.11099240732309193, 1.2000000000000002: 0.12085430176246939, 1.5: 0.13101805805398264, 1.7999999999999998: 0.13809601405712232, 2.0999999999999996: 0.14434985101092732, 2.3999999999999995: 0.1514923671811122, 2.7: 0.16113375554169027, 3.0: 0.16983606557377048}
        post_prob = binary_posterior_prob(gmm_not_calibr_scores, prior, Cfn, Cfp)
        thresholds = np.sort(post_prob)
        gmm_FPR, gmm_TPR = ROC_plot(thresholds, post_prob, gmm_labels)

        ################################################################# EVALUATION TEST ###########################################################

        ### QUAD LOG REG BEST CONFIG  ###
        prior, Cfp, Cfn = (0.5, 10, 1)
        pca = 6
        znorm = False
        lambda_ = 0.01
        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        qlr_classifier = QuadraticLR_Classifier(lambda_, pi_tilde)

        if znorm == True:
            DTR, mu, sigma = normalize_zscore(DTR)
            DTE, _, _ = normalize_zscore(DTE, mu, sigma)

        if pca is not None:  # PCA needed
            DTR, P = PCA_impl(DTR, pca)
            DTE = np.dot(P.T, DTE)

        qlr_classifier.train(DTR, LTR);
        lr_scores = qlr_classifier.compute_scores(DTE)
        lr_scores = np.array(lr_scores)
        DCF_min, _, _ = DCF_min_impl(lr_scores, LTE, prior, Cfp, Cfn)
        ##DCF_min_lr = 0.2508###

        pred = [1 if x > 0 else 0 for x in lr_scores]
        acc, _ = accuracy(pred, LTE)
        print("Accuracy: " + str(100 - acc))
        print("DCF min: " + str(DCF_min))

        ####### SVM REG BEST CONFIG  ########
        prior, Cfp, Cfn = (0.5, 10, 1)
        pca = 6
        znorm = False
        K_svm = 0
        C = 10
        mode = "rbf"
        gamma = 1e-3
        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        SVMObj = SVMClass(K_svm, C, pi_tilde, mode, gamma)

        if znorm == True:
            DTR, mu, sigma = normalize_zscore(DTR)
            DTE, _, _ = normalize_zscore(DTE, mu, sigma)

        if pca is not None:  # PCA needed
            DTR, P = PCA_impl(DTR, pca)
            DTE = np.dot(P.T, DTE)

        SVMObj.train(DTR, LTR);
        svm_scores = SVMObj.compute_scores(DTE)
        svm_scores = np.array(svm_scores)
        DCF_min, _, _ = DCF_min_impl(svm_scores, LTE, prior, Cfp, Cfn)

        ##DCF_min_svm = 0.2411###

        pred = [1 if x > 0 else 0 for x in svm_scores]
        # acc,_ = accuracy(pred,LTE)
        # print("Accuracy: "+str(100-acc))
        print("DCF min: " + str(DCF_min))

        ######## GMM BEST CONFIG  #################
        prior, Cfp, Cfn = (0.5, 10, 1)
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
        GMMObj = GMMClass(target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alfa, prior, Cfp,
                          Cfn)

        if znorm == True:
            DTR, mu, sigma = normalize_zscore(DTR)
            DTE, _, _ = normalize_zscore(DTE, mu, sigma)

        if pca is not None:  # PCA needed
            DTR, P = PCA_impl(DTR, pca)
            DTE = np.dot(P.T, DTE)

        GMMObj.train(DTR, LTR);
        gmm_scores = GMMObj.compute_scores(DTE)
        gmm_scores = np.array(gmm_scores)
        DCF_min, _, _ = DCF_min_impl(gmm_scores, LTE, prior, Cfp, Cfn)
        ###DCF_min_svm = 0.2182####

        pred = [1 if x > 0 else 0 for x in gmm_scores]
        # acc,_ = accuracy(pred,LTE)
        # print("Accuracy: "+str(100-acc))
        print("DCF min: " + str(DCF_min))

        #
        #
        #
        #
        #
        #

        #
        # EEEEEEEEEEEEEEEEEEEEEEVVVVVVVV           VVVVVVVV               AAA               LLLLLLLLLLL             UUUUUUUU     UUUUUUUU               AAA               TTTTTTTTTTTTTTTTTTTTTTTIIIIIIIIII     OOOOOOOOO     NNNNNNNN        NNNNNNNN
        # E::::::::::::::::::::EV::::::V           V::::::V              A:::A              L:::::::::L             U::::::U     U::::::U              A:::A              T:::::::::::::::::::::TI::::::::I   OO:::::::::OO   N:::::::N       N::::::N
        # E::::::::::::::::::::EV::::::V           V::::::V             A:::::A             L:::::::::L             U::::::U     U::::::U             A:::::A             T:::::::::::::::::::::TI::::::::I OO:::::::::::::OO N::::::::N      N::::::N
        # EE::::::EEEEEEEEE::::EV::::::V           V::::::V            A:::::::A            LL:::::::LL             UU:::::U     U:::::UU            A:::::::A            T:::::TT:::::::TT:::::TII::::::IIO:::::::OOO:::::::ON:::::::::N     N::::::N
        #   E:::::E       EEEEEE V:::::V           V:::::V            A:::::::::A             L:::::L                U:::::U     U:::::U            A:::::::::A           TTTTTT  T:::::T  TTTTTT  I::::I  O::::::O   O::::::ON::::::::::N    N::::::N
        #   E:::::E               V:::::V         V:::::V            A:::::A:::::A            L:::::L                U:::::D     D:::::U           A:::::A:::::A                  T:::::T          I::::I  O:::::O     O:::::ON:::::::::::N   N::::::N
        #   E::::::EEEEEEEEEE      V:::::V       V:::::V            A:::::A A:::::A           L:::::L                U:::::D     D:::::U          A:::::A A:::::A                 T:::::T          I::::I  O:::::O     O:::::ON:::::::N::::N  N::::::N
        #   E:::::::::::::::E       V:::::V     V:::::V            A:::::A   A:::::A          L:::::L                U:::::D     D:::::U         A:::::A   A:::::A                T:::::T          I::::I  O:::::O     O:::::ON::::::N N::::N N::::::N
        #   E:::::::::::::::E        V:::::V   V:::::V            A:::::A     A:::::A         L:::::L                U:::::D     D:::::U        A:::::A     A:::::A               T:::::T          I::::I  O:::::O     O:::::ON::::::N  N::::N:::::::N
        #   E::::::EEEEEEEEEE         V:::::V V:::::V            A:::::AAAAAAAAA:::::A        L:::::L                U:::::D     D:::::U       A:::::AAAAAAAAA:::::A              T:::::T          I::::I  O:::::O     O:::::ON::::::N   N:::::::::::N
        #   E:::::E                    V:::::V:::::V            A:::::::::::::::::::::A       L:::::L                U:::::D     D:::::U      A:::::::::::::::::::::A             T:::::T          I::::I  O:::::O     O:::::ON::::::N    N::::::::::N
        #   E:::::E       EEEEEE        V:::::::::V            A:::::AAAAAAAAAAAAA:::::A      L:::::L         LLLLLL U::::::U   U::::::U     A:::::AAAAAAAAAAAAA:::::A            T:::::T          I::::I  O::::::O   O::::::ON::::::N     N:::::::::N
        # EE::::::EEEEEEEE:::::E         V:::::::V            A:::::A             A:::::A   LL:::::::LLLLLLLLL:::::L U:::::::UUU:::::::U    A:::::A             A:::::A         TT:::::::TT      II::::::IIO:::::::OOO:::::::ON::::::N      N::::::::N
        # E::::::::::::::::::::E          V:::::V            A:::::A               A:::::A  L::::::::::::::::::::::L  UU:::::::::::::UU    A:::::A               A:::::A        T:::::::::T      I::::::::I OO:::::::::::::OO N::::::N       N:::::::N
        # E::::::::::::::::::::E           V:::V            A:::::A                 A:::::A L::::::::::::::::::::::L    UU:::::::::UU     A:::::A                 A:::::A       T:::::::::T      I::::::::I   OO:::::::::OO   N::::::N        N::::::N
        # EEEEEEEEEEEEEEEEEEEEEE            VVV            AAAAAAA                   AAAAAAALLLLLLLLLLLLLLLLLLLLLLLL      UUUUUUUUU      AAAAAAA                   AAAAAAA      TTTTTTTTTTT      IIIIIIIIII     OOOOOOOOO     NNNNNNNN         NNNNNNN
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #

        ####################################################    EVALUATION - CALIBRATION    #########################################################
        ##QUAD LOG REG###
        prior, Cfp, Cfn = (0.5, 10, 1)
        lambda_ = 0.01
        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        QuadLogReg = QuadraticLR_Classifier(lambda_, pi_tilde)

        LogObj = LRCalibrClass(1e-2, 0.5)

        params = {"K": 5,
                  "pi": 0.5,
                  "pca": 6,
                  "costs": (1, 1),
                  "logCalibration": LogObj,
                  "znorm": False}

        DCF_effPrior, DCF_effPrior_min, lr_not_calibr_scores, lr_calibr_scores, lr_labels = kfold_calib(DTE, LTE,
                                                                                                        QuadLogReg,
                                                                                                        params, True)

        post_prob = binary_posterior_prob(lr_not_calibr_scores, prior, Cfn, Cfp)
        thresholds = np.sort(post_prob)
        nc_lr_FPR, nc_lr_TPR = ROC_plot(thresholds, post_prob, lr_labels)

        post_prob = binary_posterior_prob(lr_calibr_scores, prior, Cfn, Cfp)
        thresholds = np.sort(post_prob)
        c_lr_FPR, c_lr_TPR = ROC_plot(thresholds, post_prob, lr_labels)

        ##SVM###
        prior, Cfp, Cfn = (0.5, 10, 1)
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
        SVMObj = SVMClass(K_svm, C, pi_tilde, mode, gamma)

        LogObj = LRCalibrClass(1e-2, 0.5)

        params = {"K": 2,
                  "pi": 0.5,
                  "pca": 6,
                  "costs": (1, 1),
                  "logCalibration": LogObj,
                  "znorm": False}

        DCF_effPrior, DCF_effPrior_min, svm_not_calibr_scores, svm_calibr_scores, svm_labels = kfold_calib(DTE, LTE,
                                                                                                           SVMObj,
                                                                                                           params, True)

        post_prob = binary_posterior_prob(svm_not_calibr_scores, prior, Cfn, Cfp)
        thresholds = np.sort(post_prob)
        svm_FPR, svm_TPR = ROC_plot(thresholds, post_prob, svm_labels)

        ##GMM###
        prior, Cfp, Cfn = (0.5, 10, 1)
        pca = None
        znorm = False
        mode_target = "diag"
        mode_not_target = "diag"
        not_target_max_comp = 8
        target_max_comp = 2
        psi = 0.01
        alfa = 0.1

        pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        GMMObj = GMMClass(target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alfa, prior, Cfp,
                          Cfn)
        LogObj = LRCalibrClass(1e-2, 0.5)

        params = {"K": 2,
                  "pi": 0.5,
                  "pca": pca,
                  "costs": (1, 1),
                  "logCalibration": LogObj,
                  "znorm": znorm}

        if znorm == True:
            DTR, mu, sigma = normalize_zscore(DTR)
            DTE, _, _ = normalize_zscore(DTE, mu, sigma)

        if pca is not None:  # PCA needed
            DTR, P = PCA_impl(DTR, pca)
            DTE = np.dot(P.T, DTE)

        DCF_effPrior, DCF_effPrior_min, gmm_not_calibr_scores, gmm_calibr_scores, gmm_labels = kfold_calib(DTE, LTE,
                                                                                                           GMMObj,
                                                                                                           params, True)

        post_prob = binary_posterior_prob(gmm_not_calibr_scores, prior, Cfn, Cfp)
        thresholds = np.sort(post_prob)
        gmm_FPR, gmm_TPR = ROC_plot(thresholds, post_prob, gmm_labels)

        #############################################    EVALUATION - COMPARATION AMONG SEVERAL MODELS    ########################################

        ##LOG REG - EVAL AND VAL
        prior, Cfp, Cfn = (0.5, 10, 1)
        K = 5
        lambda_plot = np.logspace(-4, 2, num=7)
        lr_pca6_glob = []
        lr_pca7_glob = []
        lr_pca8_glob = []
        lr_pca9_glob = []
        lr_pcaNone_glob = []
        lr_pca6_glob_zscore = []

        pi = 0.5
        piT = 0.1
        lr_pca6 = []
        lr_pca7 = []
        lr_pca8 = []
        lr_pca9 = []
        lr_pcaNone = []
        lr_pca6_zscore = []
        lr_pcaNone_zscore = []

        # prior,Cfp,Cfn = (0.5,10,1)
        # pca=6
        # znorm = False
        # l=0.01
        pi_tilde = (pi * Cfn) / (pi * Cfn + (1 - pi) * Cfp)
        # quadLogObj=quadLogRegClass(l, pi_tilde)

        for zscore in [True]:
            for pca in [6, None]:

                DTRe = DTR
                DTEe = DTE

                if zscore == True:
                    DTRe, mu, sigma = normalize_zscore(DTRe)
                    DTEe, _, _ = normalize_zscore(DTEe, mu, sigma)

                if pca is not None:  # PCA needed
                    DTRe, P = PCA_impl(DTRe, pca)
                    DTEe = np.dot(P.T, DTEe)

                for lambda_ in lambda_plot:
                    # we saw that piT=0.1 is the best value

                    # pi = 0.5
                    params = {"K": 5,
                              "pca": pca,
                              "pi": pi,
                              "costs": (1, 1),
                              "znorm": zscore}

                    qlr_classifier = QuadraticLR_Classifier(lambda_, pi_tilde)
                    qlr_classifier.train(DTRe, LTR);
                    lr_scores = qlr_classifier.compute_scores(DTEe)
                    lr_scores = np.array(lr_scores)
                    min_DCF, _, _ = DCF_min_impl(lr_scores, LTE, pi, Cfp, Cfn)
                    print(
                        f"EVAL Log Reg min_DCF con K = {K} , pca = {pca}, l = {lambda_} , piT = {piT}, pi = {pi} zscore={zscore}: {min_DCF} ")

                    if pca == 6:
                        if zscore == True:
                            lr_pca6_zscore.append(min_DCF)
                        else:
                            lr_pca6.append(min_DCF)

                    # if pca == 7:
                    #     lr_pca7.append(min_DCF)
                    # if pca == 8:
                    #     lr_pca8.append(min_DCF)
                    # if pca == 9:
                    #     lr_pca9.append(min_DCF)
                    if pca == None:
                        if zscore == True:
                            lr_pcaNone_zscore.append(min_DCF)
                        else:
                            lr_pcaNone.append(min_DCF)

        lr_pca6_glob.append(
            (f"EVAL Log Reg min_DCF con K = {K} , pca = 6, piT = {pi_tilde} pi={pi} zscore={zscore}", lr_pca6))
        lr_pca6_glob_zscore.append(
            (f"EVAL Log Reg min_DCF con K = {K} , pca = 6, piT = {pi_tilde} pi={pi} zscore={zscore}", lr_pca6_zscore))
        # lr_pca7_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 7, piT = {piT} pi={pi}",lr_pca7))
        # lr_pca8_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 8, piT = {piT} pi={pi}",lr_pca8))
        # lr_pca9_glob.append((f"EVAL Log Reg min_DCF con K = {K} , pca = 9, piT = {piT} pi={pi}",lr_pca9))
        lr_pcaNone_glob.append((
            f"EVAL Log Reg min_DCF con K = {K} , pca = No, piT = {pi_tilde} pi={pi} zscore={zscore}"",lr_pcaNone",
            lr_pcaNone))

        lr_pcaNone = [0.2909445701357466, 0.2853092006033182, 0.2764027149321267, 0.2864762443438914,
                      0.28230957767722475, 0.2687160633484163, 0.29662141779788836]
        lr_pca6 = [0.25190422322775263, 0.2505184766214178, 0.25084087481146305, 0.2540082956259427,
                   0.25702865761689286, 0.26502828054298644, 0.2934445701357466]
        lr_pca6_zscore = [0.27620663650075417, 0.27039404223227753, 0.269643665158371, 0.28359162895927603,
                          0.3204769984917044, 0.3496436651583711, 0.35650829562594266]
        lr_pcaNone_zscore = [0.2836632730015083, 0.26483031674208146, 0.24457013574660635, 0.25703808446455506,
                             0.28994532428355957, 0.3271021870286576, 0.3367269984917044]

        val_lr_pca6 = [0.2718032786885246, 0.2718032786885246, 0.26305327868852457, 0.28616803278688524,
                       0.3027049180327869, 0.3280327868852459, 0.3560860655737705]
        val_lr_pca6_zscore = [0.26524590163934425, 0.26430327868852455, 0.2814344262295082, 0.32237704918032783,
                              0.34616803278688524, 0.3458606557377049, 0.3458606557377049]
        val_lr_pcaNone = [0.3230327868852459, 0.3114549180327869, 0.2889549180327869, 0.3046106557377049,
                          0.3105327868852459, 0.3176844262295082, 0.34987704918032786]

        plt.semilogx(lambda_plot, val_lr_pca6, label="PCA 6 (VAL)", linestyle='--', color="blue")
        plt.semilogx(lambda_plot, val_lr_pca6_zscore, label="PCA 6 - ZSCORE (VAL)", linestyle='--', color="orange")
        plt.semilogx(lambda_plot, val_lr_pcaNone, label="PCA None (VAL)", linestyle='--', color="green")
        plt.semilogx(lambda_plot, lr_pca6, label="PCA 6 (EVAL)", color="blue")
        plt.semilogx(lambda_plot, lr_pca6_zscore, label="PCA 6 - ZSCORE (EVAL)")
        plt.semilogx(lambda_plot, lr_pcaNone, label="PCA None (EVAL)")
        plt.semilogx(lambda_plot, lr_pcaNone_zscore, label="PCA None - ZSCORE (EVAL)")
        # plt.semilogx(lamb,lr_pca7, label = "PCA 7")
        # plt.semilogx(lamb,lr_pca8, label = "PCA 8")
        # plt.semilogx(lamb,lr_pca9, label = "PCA 9")
        # plt.semilogx(lamb,lr_pcaNone, label = "No PCA")

        plt.xlabel("Lambda")
        plt.ylabel("DCF_min")
        plt.legend()
        # if piT == 0.1:
        #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_min"
        # if piT == 0.33:
        #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_033"
        # if piT == 0.5:
        #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_medium"
        # if piT == 0.9:
        #     path = "plots/quadLogReg/copy/DCF_su_lambda_piT_max"
        titolo = "Quad Log Reg - EVAL and VAL confr"
        plt.title(titolo)
        # plt.savefig(path)
        plt.show()

        ##SVM - SEVERAL CONFIG (POLY)####

        K = 5
        piT = 0.1
        prior, Cfp, Cfn = (0.5, 10, 1)
        poly_svm_pca6 = {}
        poly_svm_pca8 = {}
        poly_svm_pcaNone = {}
        rbf_svm_pca6 = {}
        rbf_svm_pca8 = {}
        rbf_svm_pcaNone = {}
        for piT in [0.1]:
            for kernel in ["poly"]:
                if kernel == "poly":
                    ci = [0, 1]
                    string = "d=2 c= "
                else:
                    ci = [0.01, 0.001, 0.0001]
                    string = "gamma= "
                svm_pca6_y1 = []
                svm_pca6_y2 = []

                svm_pca6_y1_noz = []
                svm_pca6_y2_noz = []

                svm_pcaNone_y1 = []
                svm_pcaNone_y2 = []

                svm_pcaNone_y1_noz = []
                svm_pcaNone_y2_noz = []

                for value in ci:

                    # svm_pcaNone = []
                    C_values = np.logspace(-3, -1, num=3)
                    for pca in [6, None]:
                        DTRe = DTR
                        DTEe = DTE
                        for K_svm in [0]:
                            # we saw that piT=0.1 is the best value
                            for C in np.logspace(-3, -1, num=3):
                                for znorm in [False, True]:
                                    params = {"K": 5,
                                              "pca": pca,
                                              "pi": 0.5,
                                              "costs": (1, 1),
                                              "znorm": znorm
                                              }
                                    SVMObj = SVMClass(K_svm, C, piT, kernel, value)

                                    if znorm == True:
                                        DTRe, mu, sigma = normalize_zscore(DTRe)
                                        DTEe, _, _ = normalize_zscore(DTEe, mu, sigma)

                                    if pca is not None:  # PCA needed
                                        DTRe, P = PCA_impl(DTRe, pca)
                                        DTEe = np.dot(P.T, DTEe)

                                    SVMObj.train(DTRe, LTR);
                                    svm_scores = SVMObj.compute_scores(DTEe)
                                    svm_scores = np.array(svm_scores)
                                    min_DCF, _, _ = DCF_min_impl(svm_scores, LTE, prior, Cfp, Cfn)

                                    print(
                                        f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} znorm: {znorm}")

                                    if pca == 6:
                                        if znorm == True:
                                            if value == 0:
                                                svm_pca6_y1.append(min_DCF)
                                            if value == 1:
                                                svm_pca6_y2.append(min_DCF)

                                            if kernel == "poly":
                                                poly_svm_pca6.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",
                                                    min_DCF)
                                            else:
                                                rbf_svm_pca6.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",
                                                    min_DCF)
                                        else:
                                            if value == 0:
                                                svm_pca6_y1_noz.append(min_DCF)
                                            if value == 1:
                                                svm_pca6_y2_noz.append(min_DCF)

                                            if kernel == "poly":
                                                poly_svm_pca6.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",
                                                    min_DCF)
                                            else:

                                                rbf_svm_pca6.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",
                                                    min_DCF)

                                    # if pca == 7:
                                    #     svm_pca7.append(min_DCF)
                                    # if pca == 8:
                                    #     svm_pca8.append(min_DCF)
                                    #     if kernel=="poly" :
                                    #         poly_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                    #     else:

                                    #         rbf_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                                    # # if pca == 9:
                                    # #     svm_pca9.append(min_DCF)
                                    if pca == None:
                                        if znorm == True:
                                            if value == 0:
                                                svm_pcaNone_y1.append(min_DCF)
                                            if value == 1:
                                                svm_pcaNone_y2.append(min_DCF)

                                            if kernel == "poly":
                                                poly_svm_pcaNone.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",
                                                    min_DCF)
                                            else:
                                                rbf_svm_pcaNone.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",
                                                    min_DCF)
                                        else:
                                            if value == 0:
                                                svm_pcaNone_y1_noz.append(min_DCF)
                                            if value == 1:
                                                svm_pcaNone_y2_noz.append(min_DCF)

                                            if kernel == "poly":
                                                poly_svm_pcaNone.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",
                                                    min_DCF)
                                            else:
                                                rbf_svm_pcaNone.setdefault(
                                                    f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",
                                                    min_DCF)

        plt.semilogx(C_values, svm_pca6_y1, label="PCA 6 c1=0")
        plt.semilogx(C_values, svm_pca6_y2, label="PCA 6 ci=1")

        plt.semilogx(C_values, svm_pcaNone_y1, label="No PCA ci=0")
        plt.semilogx(C_values, svm_pcaNone_y2, label="No PCA ci=1")

        # plt.semilogx(C_values,svm_pca7, label = "PCA 7")
        plt.xlabel("C")
        plt.ylabel("DCF_min")
        plt.legend()
        plt.title("Poly with Znorm")
        # plt.savefig(path)
        plt.show()

        plt.semilogx(C_values, svm_pca6_y1_noZ, label="PCA 6 ci=0 noZnorm")
        plt.semilogx(C_values, svm_pca6_y2_noZ, label="PCA 6 ci=1 noZnorm")

        plt.semilogx(C_values, svm_pcaNone_y1_noz, label="No PCA ci=0 noZnorm")
        plt.semilogx(C_values, svm_pcaNone_y2_noz, label="No PCA ci=1 noZnorm")

        # plt.semilogx(C_values,svm_pca9, label = "PCA 9")

        plt.xlabel("C")
        plt.ylabel("DCF_min")
        plt.legend()
        plt.title("Poly no Znorm")
        # plt.savefig(path)
        plt.show()

        ##############   GMM - several config  ###############
        K = 5
        prior = 0.5
        Cfp = 10
        Cfn = 1
        gmm_pca6_glob = {}
        gmm_pcaNone_glob = {}

        for mode_target in ["full", "diag", "tied"]:
            for mode_not_target in ["full", "diag", "tied"]:
                gmm_pca6 = []
                gmm_pcaNone = []
                for pca in [6, None]:

                    DTRe = DTR
                    DTEe = DTE

                    if pca is not None:  # PCA needed
                        DTRe, P = PCA_impl(DTRe, pca)
                        DTEe = np.dot(P.T, DTEe)

                    for t_max_n in [1, 2]:
                        gmm_tmp = []
                        for nt_max_n in [2, 4, 8]:
                            for znorm in [False]:
                                alfa = 0.1
                                psi = 0.01
                                params = {"K": 5,
                                          "pca": pca,
                                          "pi": 0.5,
                                          "costs": (1, 1),
                                          "znorm": znorm}

                                GMMObj = GMMClass(t_max_n, nt_max_n, mode_target, mode_not_target, psi, alfa, prior,
                                                  Cfp, Cfn)
                                GMMObj.train(DTRe, LTR);
                                gmm_scores = GMMObj.compute_scores(DTEe)
                                gmm_scores = np.array(gmm_scores)
                                min_DCF, _, _ = DCF_min_impl(gmm_scores, LTE, prior, Cfp, Cfn)

                                print(
                                    f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}")

                                # un vettore che si annulla ad ogni nuovo t_max_n
                                gmm_tmp.append(min_DCF)
                                if pca == 6:
                                    gmm_pca6.append(min_DCF)
                                    gmm_pca6_glob.setdefault((
                                        f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                        min_DCF))

                                if pca == None:
                                    gmm_pcaNone.append(min_DCF)
                                    gmm_pcaNone_glob.setdefault((
                                        f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                        min_DCF))

                        fig = plt.figure()
                        plt.plot([2, 4, 8], gmm_tmp)
                        plt.xlabel("nt_max_n")
                        plt.ylabel("DCF_min")
                        titolo = f"mode_target: {mode_target}, mode_non_target:{mode_not_target} PCA: {pca}, t_max_n: {t_max_n}"
                        plt.title(titolo)
                        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classificazione.")
    parser.add_argument('--type', type=str, default='MVG', help='Percorso del file di input')
    parser.add_argument('--mode', type=str, default='train', help='Percorso del file di input')

    args = parser.parse_args()

    main(args.type, args.mode)
