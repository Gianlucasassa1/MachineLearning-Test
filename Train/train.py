import numpy
from Preprocess.PCA import *
from Preprocess.Znorm import znorm_impl, normalize_zscore
from Models.Gaussian.utils import *
from Models.Gaussian.Gaussian import *
from Evaluation.evaluation import *
from Evaluation.evaluation import *
from Models.Gaussian.utils import *
from Models.LogisticRegression.logreg import LR_Classifier, QuadraticLR_Classifier
from Models.SupportVector.svm import SVMClassifier
from Models.SupportVector.svm_kernel import SVMKernelClassifier
from Models.MixtureModels.gmm import GMMClassifier
from Models.SupportVector.svm import SVMClassifier
from Models.SupportVector.svm_kernel import SVMKernelClassifier
from Models.MixtureModels.gmm import GMMClassifier


def train_KFold(D, L, classifier, options):
    # options
    K = options["K"]
    pca = options["pca"]
    pi = options["pi"]
    (cfn, cfp) = options["costs"]
    znorm = options["znorm"]

    samplesNumber = D.shape[1]
    N = int(samplesNumber / K)

    numpy.random.seed(seed=0)
    indexes = numpy.random.permutation(D.shape[1])

    scores = numpy.array([])
    labels = numpy.array([])

    for i in range(K):
        idxTest = indexes[i * N:(i + 1) * N]

        idxTrainLeft = indexes[0:i * N]
        idxTrainRight = indexes[(i + 1) * N:]
        idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        # zed-normalizes the data with the mu and sigma computed with DTR
        # DTR, mu, sigma = normalize_zscore(DTR)
        # DTE, mu, sigma = normalize_zscore(DTE, mu, sigma)

        if znorm == True:
            DTR, mu, sigma = normalize_zscore(DTR)
            DTE, _, _ = normalize_zscore(DTE, mu, sigma)

        if pca is not None:  # PCA needed
            DTR, P = PCA_impl(DTR, pca)
            DTE = numpy.dot(P.T, DTE)

        classifier.train(DTR, LTR)

        scores_i = classifier.compute_scores(DTE)
        # print("SMV score: ")
        # print(scores_i)
        scores = numpy.append(scores, scores_i)
        labels = numpy.append(labels, LTE)

    labels = np.array(labels, dtype=int)
    min_DCF, _, _ = DCF_min_impl(scores, labels, pi, cfp, cfn)

    return min_DCF, scores, labels


def trainGaussian(DTR, LTR, file):
    print("... Gaussian")
    prior, Cfp, Cfn = (0.5, 1, 1)
    params = {"K": 5, "pca": 6, "pi": 0.5, "costs": (1, 1)}
    K = 5
    MVG_obj = GaussianClassifier("MVG", prior, Cfp, Cfn)
    NB_obj = GaussianClassifier("NB", prior, Cfp, Cfn)
    TCG_obj = GaussianClassifier("TCG", prior, Cfp, Cfn)
    TCGNB_obj = GaussianClassifier("TCGNB", prior, Cfp, Cfn)
    PCA10_plot = 0
    PCA11_plot = 0
    PCA12_plot = 0
    NoPCA_plot = 0
    for model in [MVG_obj, NB_obj, TCG_obj, TCGNB_obj]:
        for pca in [10, 11, 12, None]:
            params = {"K": 5, "pca": pca, "pi": 0.5, "znorm": False, "costs": (1, 1)}
            min_DCF, scores, labels = train_KFold(DTR, LTR, model, params)

            if pca == 10:
                PCA10_plot = min_DCF
            if pca == 11:
                PCA11_plot = min_DCF
            if pca == 12:
                PCA12_plot = min_DCF
            if pca is None:
                NoPCA_plot = min_DCF

            info = f"{model.mode} | PCA ({pca}) ||| min_DCF = \033[1m {min_DCF} \033[0m"
            print(info)
            file.write(info + '\n')

        mvg_pca = np.array([PCA10_plot, PCA11_plot, PCA12_plot, NoPCA_plot])

        # Definisci il percorso della nuova cartella
        folder_path = 'Images/Gaussian/'

        # Crea la nuova cartella
        os.makedirs(folder_path, exist_ok=True)

        plt.xlabel("PCA dimensions")
        plt.ylabel("DCF_min")
        plt.title(model.mode)
        path = "Images/Gaussian/" + str(model.mode)
        plt.plot(np.linspace(10, 13, 4), mvg_pca)
        plt.savefig(path)
        plt.close()


def trainLR(DTR, LTR, file):
    base_folder_path = 'Images/LogisticRegression/'
    piT_values = [0.1, 0.5, 0.9]
    lambda_values = np.logspace(-5, 5, num=2)
    pca_values = [11, None]
    zscore_values = [True, False]
    K = 5
    # piT = 0.1
    # lambda_plot = np.logspace(-1, 1, num=3)
    for piT in piT_values:
        plt.figure()
        for zscore in zscore_values:
            PCA10_LR = []
            PCA11_LR = []
            PCA12_LR = []
            NoPCA_LR = []
            for lambda_temp in lambda_values:
                for pca in pca_values:
                    params = {"K": 5, "pca": pca, "pi": 0.5, "costs": (1, 1), "znorm": zscore}
                    logObj = LR_Classifier(lambda_temp, piT)
                    min_DCF, scores, labels = train_KFold(DTR, LTR, logObj, params)

                    info = f"Logistic Regression | piT{piT} | Znorm: {zscore}| lambda {lambda_temp} | PCA ({pca}) ||| min_DCF = \033[1m {min_DCF} \033[0m"
                    print(info)
                    file.write(info + '\n')

                    # if zscore:
                    if pca == 10:
                        PCA10_LR.append(min_DCF)
                    if pca == 11:
                        PCA11_LR.append(min_DCF)
                    if pca == 12:
                        PCA12_LR.append(min_DCF)
                    if pca is None:
                        NoPCA_LR.append(min_DCF)
                    # else:
                    #     if pca == 10:
                    #         PCA10_LR_NoZscore.append(min_DCF)
                    #     if pca == 11:
                    #         PCA11_LR_NoZscore.append(min_DCF)
                    #     if pca == 12:
                    #         PCA12_LR_NoZscore.append(min_DCF)
                    #     if pca is None:
                    #         NoPCA_LR_NoZscore.append(min_DCF)

            label_suffix = "Znorm" if zscore else "no Znorm"
            # plt.semilogx(lambda_values, PCA10_LR, label=f"PCA 10 {label_suffix}")
            plt.semilogx(lambda_values, PCA11_LR, label=f"PCA 11 {label_suffix}")
            # plt.semilogx(lambda_values, PCA12_LR, label=f"PCA 12 {label_suffix}")
            plt.semilogx(lambda_values, NoPCA_LR, label=f"No PCA {label_suffix}")

        plt.xlabel("Lambda")
        plt.ylabel("DCF_min")
        plt.legend()

        #   TODO: can be optimized
        # Definisci il percorso della nuova cartella
        folder_path_1 = 'Images/LogisticRegression/MinimumPIT/'
        folder_path_2 = 'Images/LogisticRegression/MediumPIT/'
        folder_path_3 = 'Images/LogisticRegression/MaximumPIT/'

        # path = "Images/Gaussian/" + str(model.mode)

        # # Crea la nuova cartella
        # os.makedirs(folder_path_1, exist_ok=True)
        # os.makedirs(folder_path_2, exist_ok=True)
        # os.makedirs(folder_path_3, exist_ok=True)

        # if piT == 0.1:
        #     path = folder_path_1 + "result.png"
        # elif piT == 0.5:
        #     path = folder_path_2 + "result.png"
        # elif piT == 0.9:
        #     path = folder_path_3 + "result.png"

        plt.title(f"Logistic Regression with πT = {piT}")
        folder_path = base_folder_path + f'piT{int(piT * 10)}/'
        os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't exist
        path = folder_path + "result.png"

        plt.savefig(path)
        plt.close()


def trainQLR(DTR, LTR, file):
    base_folder_path = 'Images/QuadraticLogisticRegression/'
    pi_values = [0.1, 0.5, 0.9]
    piT_values = [0.1, 0.5, 0.9]
    lambda_values = np.logspace(-1, 1, num=3)
    pca_values = [10, None]
    zscore_values = [True, False]
    K = 5
    # Results to be plotted
    # PCA10_plot = []
    # PCA11_plot = []
    # PCA12_plot = []
    # NoPCA_plot = []
    for pi in pi_values:
        piT = 1 - pi
        plt.figure()
        for zscore in zscore_values:
            PCA10_qlr = []
            PCA11_qlr = []
            PCA12_qlr = []
            NoPCA_qlr = []
            for lambda_ in lambda_values:
                for pca in pca_values:
                    params = {"K": 5, "pca": pca, "pi": pi, "costs": (1, 1), "znorm": zscore}

                    qlr_classifier = QuadraticLR_Classifier(lambda_, piT)

                    min_DCF, scores, labels = train_KFold(DTR, LTR, qlr_classifier, params)

                    info = f"Logistic Regression - Quadratic | pi {pi} | piT {piT} | Znorm {zscore}| Lambda {lambda_} | PCA: ({pca}) ||| min_DCF = \033[1m {min_DCF} \033[0m"
                    print(info)
                    file.write(info + '\n')

                    if pca == 10:
                        PCA10_qlr.append(min_DCF)
                    if pca == 11:
                        PCA11_qlr.append(min_DCF)
                    if pca == 12:
                        PCA12_qlr.append(min_DCF)
                    if pca is None:
                        NoPCA_qlr.append(min_DCF)

            label_suffix = f"Znorm={zscore}"
            if PCA10_qlr:
                plt.semilogx(lambda_values, PCA10_qlr, label=f"PCA 10, {label_suffix}")
            # if PCA11_qlr:
            #     plt.semilogx(lambda_values, PCA11_qlr, label=f"PCA 11, {label_suffix}")
            # if PCA12_qlr:
            #     plt.semilogx(lambda_values, PCA12_qlr, label=f"PCA 12, {label_suffix}")
            if NoPCA_qlr:
                plt.semilogx(lambda_values, NoPCA_qlr, label=f"No PCA, {label_suffix}")

        plt.xlabel("Lambda")
        plt.ylabel("DCF_min")
        plt.legend()
        plt.title(f"Quadratic Logistic Regression with piT = {piT}, pi = {pi}")

        # Create the directory for the current pi and piT
        folder_path = os.path.join(base_folder_path, f'piT_{piT}_pi_{pi}')
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f"Plot_piT_{piT}_pi_{pi}.png"))
        plt.close()

        # PCA10_plot.append(
        #     (f"Log Reg min_DCF con K = {K} , pca = 6, piT = {piT}, pi={pi} zscore={zscore}", PCA10_qlr))
        # PCA11_plot.append((f"Log Reg min_DCF con K = {K} , pca = 7, piT = {piT}, pi={pi}", PCA11_qlr))
        # PCA12_plot.append((f"Log Reg min_DCF con K = {K} , pca = 8, piT = {piT} pi={pi}", PCA12_qlr))
        # NoPCA_plot.append((f"Log Reg min_DCF con K = {K} , pca = no, piT = {piT} pi={pi}", NoPCA_qlr))

        # lambda_plot = np.logspace(-4, 2, num=7)
        #
        # plt.semilogx(lambda_plot, PCA10_qlr, label="PCA 10")
        # plt.semilogx(lambda_plot, PCA11_qlr, label="PCA 11")
        # plt.semilogx(lambda_plot, PCA12_qlr, label="PCA 12")
        # plt.semilogx(lambda_plot, NoPCA_qlr, label="No PCA")
        #
        # # Definisci il percorso della nuova cartella
        # folder_path = f'Images/QuadraticLogisticRegression/PiT_{piT}_pi_{pi}'
        #
        # # Crea la nuova cartella
        # os.makedirs(folder_path, exist_ok=True)
        #
        # plt.xlabel("Lambda")
        # plt.ylabel("DCF_min")
        # plt.legend()
        # plt.title(str(piT))
        # plt.savefig(os.path.join(folder_path, f"Plot_piT_{piT}.png"))
        # plt.close()


def trainSVM(DTR, LTR, file):
    K = 5
    K_svm = 1  # else 0
    pi = 0.5
    base_folder_path = 'Images/SupportVectorMachine/'
    for piT in [0.1]:
        svm_pca10_no_zscore = []
        svm_pca10_zscore = []
        svm_pca11_no_zscore = []
        svm_pca11_zscore = []
        svm_pcaNone_no_zscore = []
        svm_pcaNone_zscore = []
        svm_pca6 = []
        svm_pca7 = []
        svm_pca8 = []
        svm_pca9 = []
        svm_pcaNone = []
        C_values = np.logspace(-1, 0, num=2)
        for C in C_values:
            for zscore in [True, False]:
                # we saw that piT=0.1 is the best value
                for pca in [10, None]:
                    params = {"K": 5, "pca": pca, "pi": pi, "costs": (1, 1), "znorm": zscore}

                    SVMObj = SVMClassifier(K_svm, C, piT)
                    min_DCF, scores, labels = train_KFold(DTR, LTR, SVMObj, params)

                    if min_DCF > 1:
                        min_DCF = 1
                    info = f"Support Vector Machine | piT {piT} | C {C} | Znorm {zscore}| PCA: ({pca}) ||| min_DCF = \033[1m {min_DCF} \033[0m"
                    print(info)
                    file.write(info + '\n')

                    if pca == 10:
                        if not zscore:
                            svm_pca10_no_zscore.append(min_DCF)
                        else:
                            svm_pca10_zscore.append(min_DCF)
                    elif pca == 11:
                        if not zscore:
                            svm_pca11_no_zscore.append(min_DCF)
                        else:
                            svm_pca11_zscore.append(min_DCF)
                    elif pca is None:
                        if not zscore:
                            svm_pcaNone_no_zscore.append(min_DCF)
                        else:
                            svm_pcaNone_zscore.append(min_DCF)

        # SVM PCA 10
        plt.semilogx(C_values, svm_pca10_no_zscore, label="PCA 10 without Z-Norm")
        plt.semilogx(C_values, svm_pca10_zscore, label="PCA 10 with Z-Norm")
        # SVM PCA 11
        # plt.semilogx(C_values, svm_pca11_no_zscore, label="PCA 11 without Z-Norm")
        # plt.semilogx(C_values, svm_pca11_zscore, label="PCA 11 with Z-Norm")
        # SVM None PCA
        plt.semilogx(C_values, svm_pcaNone_no_zscore, label="No PCA without Z-Norm")
        plt.semilogx(C_values, svm_pcaNone_zscore, label="No PCA with Z-Norm")

        #  TODO: can be optimized
        # Definisci il percorso della nuova cartella
        # folder_path = f'Images/SupportVector/C={C}/'

        # Crea la nuova cartella
        # os.makedirs(folder_path, exist_ok=True)

        plt.xlabel("C")
        plt.ylabel("DCF_min")
        plt.legend()
        plt.title(f"Support Vector with piT = {piT}")

        # Create the directory for the current pi and piT
        folder_path = os.path.join(base_folder_path, f'piT_{piT}')
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f"Plot_piT_{piT}.png"))
        plt.close()

        # plt.title(piT)
        #
        # plt.savefig(os.path.join(folder_path, f"Plot_{piT}.png"))
        # plt.close()


def trainKSVM(DTR, LTR, file):
    K = 5
    piT_values = [0.1, 0.5, 0.9]
    C_values = np.logspace(-3, -1, num=3)
    kernels = [("poly", C_values, "d=2 c= "), ("rbf", [0.1, 0.01, 0.001], "gamma= ")]
    pca_values = [10, 11, None]
    for piT in piT_values:
        results = {}

        for kernel, ci_values, kernel_string in kernels:
            for ci in ci_values:
                for pca in pca_values:
                    for znorm in [False, True]:
                        # Initialize the results list for each configuration
                        results_key = f"PCA {pca if pca is not None else 'None'} {kernel} {kernel_string}{ci} Znorm {znorm}"
                        results[results_key] = []

                        for C in C_values:
                            params = {"K": K, "pca": pca, "pi": 0.5, "costs": (1, 1), "znorm": znorm}
                            SVMObj = SVMKernelClassifier(K, C, piT, kernel, ci)
                            min_DCF, scores, labels = train_KFold(DTR, LTR, SVMObj, params)
                            min_DCF = min(min_DCF, 1)  # Ensure min_DCF doesn't exceed 1
                            results[results_key].append(min_DCF)
                            info = f"SVM - Kernel | piT {piT} | Kernel {kernel} | {kernel_string}{ci} | C {C} | PCA: {pca if pca is not None else 'None'} | Znorm {znorm} ||| min_DCF = {min_DCF}"
                            print(info)
                            file.write(info + '\n')

                # Plotting for each kernel and ci value
                plt.figure()
                for znorm in [False, True]:
                    for pca in pca_values:
                        label = f"PCA {pca if pca is not None else 'None'} {'Znorm' if znorm else 'No Znorm'}"
                        results_key = f"PCA {pca if pca is not None else 'None'} {kernel} {kernel_string}{ci} Znorm {znorm}"
                        plt.semilogx(C_values, results[results_key], label=label)

                plt.xlabel("C Value")
                plt.ylabel("Min DCF")
                plt.legend()
                plt.title(f"{kernel.capitalize()} Kernel with {kernel_string}{ci} (piT={piT})")

                # Create the folder if not exists
                folder_path = f'Images/SupportVectorMachines/{kernel}/piT_{piT}'
                os.makedirs(folder_path, exist_ok=True)

                # Saving the plot
                plt.savefig(os.path.join(folder_path, f"Plot_{piT}_{kernel}_{ci}.png"))
                plt.close()


def trainGMM(DTR, LTR, file):
    K = 5
    prior = 0.5
    Cfp = 10
    Cfn = 1
    # Define the configurations
    # modes_old = ["diag", "tied", "full"]

    modes = [["full", "full"],
             ["diag", "diag"],
             ["tied", "tied"],
             ["diag", "tied"], ]
    pca_configs = [None]
    t_max_n_values = [1, 2, 4, 8, 16, 32, 64]
    nt_max_n_values = [64]  # [1, 2, 4, 8, 16, 32, 64]
    znormS = [True, False]

    results = {}
    for znorm in znormS:
        for mode_target, mode_not_target in modes:
            for pca in pca_configs:
                # Initialize lists to hold min_DCF for each configuration
                results_key = f"PCA {pca if pca is not None else 'None'} Target {mode_target} Non-Target {mode_not_target}"
                results[results_key] = []

                for t_max_n in t_max_n_values:
                    for nt_max_n in nt_max_n_values:
                        alfa = 0.1
                        psi = 0.01
                        params = {"K": K, "pca": pca, "pi": prior, "costs": (Cfn, Cfp), "znorm": znorm}

                        GMMObj = GMMClassifier(t_max_n, nt_max_n, mode_target, mode_not_target, psi, alfa, prior,
                                               Cfp, Cfn)
                        min_DCF, scores, labels = train_KFold(DTR, LTR, GMMObj, params)

                        info = f"Gaussian Mixture Models | Target {mode_target} | Non-Target {mode_not_target} | PCA: ({pca}) | t_max_n={t_max_n} | nt_max_n={nt_max_n} | Znorm {znorm} ||| min_DCF = {min_DCF}"
                        print(info)
                        file.write(info + '\n')

                        results[results_key].append(min_DCF)

            # Plotting for each PCA configuration
            plt.figure()
            x_values = [1, 2, 4, 8, 16, 32, 64]
            for key, min_DCF_values in results.items():
                if "znorm" in key:
                    label = key
                    plt.plot(x_values, min_DCF_values, label=label)
                    plt.xticks(x_values, x_values)

            plt.xlabel("Components")
            plt.ylabel("Min DCF")
            plt.legend()
            string_title = "GMM"
            # plt.title(f"GMM Performance ) #not used
            # Title
            if mode_target == "full":
                string_title = string_title + ""
            elif mode_target == "diag":
                string_title = string_title + "Diag"
            elif mode_target == "tied":
                string_title = string_title + "Tied"
                if mode_not_target == "Diag":
                    string_title = string_title + "Diag"

            plt.title(string_title)
            # Create the folder if not exists
            folder_path = f'Images/GMM/{mode_target}_{mode_not_target}'
            os.makedirs(folder_path, exist_ok=True)

            # Saving the plot
            plt.savefig(os.path.join(folder_path, f"GMM_{mode_target}_{mode_not_target}.png"))
            plt.close()
