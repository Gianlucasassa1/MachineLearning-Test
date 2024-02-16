import numpy as np
import scipy
import math
from Models.Gaussian.utils import *


def vcol(vector):
    return vector.reshape((vector.shape[0], 1))


def vrow(vector):
    return vector.reshape((1, vector.shape[0]))


def dataset_mean(dataset):
    return dataset.mean(1).reshape(
        (dataset.shape[0], 1))  # dataset.shape[0] è numero features per rendere codice generico


def num_samples(dataset):
    return dataset.shape[1];


def ML_estimate(X):
    mu_ml = dataset_mean(dataset=X)
    covariance_matrix_ml = (np.dot(((X - mu_ml)), (X - mu_ml).T)) / num_samples(X)

    return mu_ml, covariance_matrix_ml


def loglikelihood(X, m_ML, C_ML):
    return np.sum(logpdf_GAU_ND(X, m_ML, C_ML))


def logpdf_GMM(X, gmm):
    num_components = len(gmm)
    logS = np.empty((num_components, X.shape[1]))

    for component_index in range(num_components):
        logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
        logS[component_index, :] += np.log(gmm[component_index][0])  # computing the joint loglikelihood densities
    logSMarginal = scipy.special.logsumexp(logS, axis=0)

    return logSMarginal


class GMMClassifier:

    def __init__(self, target_max_comp, not_target_max_comp, mode_target, mode_not_target, psi, alpha, prior, Cfp, Cfn):
        self.not_target_max_comp = not_target_max_comp  # number max components in LBG (after all the splits), the number of components will be n
        self.target_max_comp = target_max_comp  # number max components in LBG (after all the splits), the number of components will be n
        self.gmm0 = None
        self.gmm1 = None
        self.model = None
        self.mode_target = mode_target
        self.mode_not_target = mode_not_target
        self.psi = psi  # psi (lower bound for eigenvalues of the cov matrices in LBG)
        self.alpha = alpha  # alfa LBG factor
        # self.tiedness = tiedness        #boolean

        self.eff_prior = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        # print("eff prior: "+str(self.eff_prior))

    # we train two different GMM (1 for the target class and one for the non target class)
    def train(self, DTR, LTR):
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        model = []

        if self.mode_not_target == "full":
            self.gmm0 = self.LBG(D0, self.psi, self.alpha, self.not_target_max_comp)
        elif self.mode_not_target == "diag":
            self.gmm0 = self.Diag_LBG(D0, self.psi, self.alpha, self.not_target_max_comp)
        elif self.mode_not_target == "tied":
            self.gmm0 = self.TiedCov_LBG(D0, self.psi, self.alpha, self.not_target_max_comp)
        else:
            print("Error for non target mode")

        model.append(self.gmm0)

        if self.mode_target == "full":
            self.gmm1 = self.LBG(D1, self.psi, self.alpha, self.target_max_comp)
        elif self.mode_target == "diag":
            self.gmm1 = self.Diag_LBG(D1, self.psi, self.alpha, self.target_max_comp)
        elif self.mode_target == "tied":
            self.gmm1 = self.TiedCov_LBG(D1, self.psi, self.alpha, self.target_max_comp)
        else:
            print("Error for target mode")

        model.append(self.gmm1)

        self.model = model

        return model

    def compute_scores(self, DTE):
        llr = self.predict_log(self.model, DTE, [1 - self.eff_prior, self.eff_prior])
        return llr

    def GMMExpectationMaximization(self, X, gmm, psi):
        while True:
            num_components = len(gmm)
            logS = np.empty((num_components, X.shape[1]))
            # ---START of E step---
            for component_index in range(num_components):
                logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
                logS[component_index, :] += np.log(
                    gmm[component_index][0])  # computing the joint loglikelihood densities
            logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
            logSPost = logS - logSMarginal
            SPost = np.exp(logSPost)

            old_loss = logSMarginal
            # ---END of E step---
            # ---START of M step---
            Z = np.sum(SPost, axis=1)

            F = np.zeros((X.shape[0], num_components))
            for component_index in range(num_components):
                F[:, component_index] = np.sum(SPost[component_index, :] * X, axis=1)

            S = np.zeros((X.shape[0], X.shape[0], num_components))
            for component_index in range(num_components):
                S[:, :, component_index] = np.dot(SPost[component_index, :] * X, X.T)

            mu = F / Z
            cov_mat = S / Z

            for component_index in range(num_components):
                cov_mat[:, :, component_index] -= np.dot(vcol(mu[:, component_index]), vrow(mu[:, component_index]))
                U, s, _ = np.linalg.svd(cov_mat[:, :, component_index])  # avoiding degenerate solutions
                s[s < psi] = psi
                cov_mat[:, :, component_index] = np.dot(U, vcol(s) * U.T)
            w = Z / np.sum(Z)

            gmm = [(w[index_component], vcol(mu[:, index_component]), cov_mat[:, :, index_component]) for
                   index_component in range(num_components)]
            new_loss = logpdf_GMM(X, gmm)
            # ---END of M step---
            # if np.mean(new_loss)-np.mean(old_loss)>0:
            #     print("Bene")
            # if np.mean(new_loss)-np.mean(old_loss)<0:
            #     print("Male")

            if np.mean(new_loss) - np.mean(old_loss) < 1e-6:
                # print("EM terminato")
                break
        # print(np.mean(new_loss))
        return gmm

    def GMMDiagExpectationMaximization(self, X, gmm, psi):  # gmm list of tuple
        while True:
            num_components = len(gmm)
            logS = np.empty((num_components, X.shape[1]))
            # ---START of E step---
            for component_index in range(num_components):
                logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
                logS[component_index, :] += np.log(
                    gmm[component_index][0])  # computing the joint loglikelihood densities
            logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
            logSPost = logS - logSMarginal
            SPost = np.exp(logSPost)

            old_loss = logSMarginal
            # ---END of E step---
            # ---START of M step---
            Z = np.sum(SPost, axis=1)

            F = np.zeros((X.shape[0], num_components))
            for component_index in range(num_components):
                F[:, component_index] = np.sum(SPost[component_index, :] * X, axis=1)

            S = np.zeros((X.shape[0], X.shape[0], num_components))
            for component_index in range(num_components):
                S[:, :, component_index] = np.dot(SPost[component_index, :] * X, X.T)

            mu = F / Z
            cov_mat = S / Z

            for component_index in range(num_components):
                cov_mat[:, :, component_index] -= np.dot(vcol(mu[:, component_index]), vrow(mu[:, component_index]))

            w = Z / np.sum(Z)

            for index_component in range(num_components):
                cov_mat[:, :, index_component] = np.diag(np.diag(cov_mat[:, :, index_component]))  # diag cov mat
                U, s, _ = np.linalg.svd(cov_mat[:, :, index_component])  # avoiding degenerate solutions
                s[s < psi] = psi
                cov_mat[:, :, index_component] = np.dot(U, vcol(s) * U.T)

            gmm = [(w[index_component], vcol(mu[:, index_component]), cov_mat[:, :, index_component]) for
                   index_component in range(num_components)]
            new_loss = logpdf_GMM(X, gmm)
            # ---END of M step---
            # if np.mean(new_loss)-np.mean(old_loss)>0:
            #     print("BENE")
            # if np.mean(new_loss)-np.mean(old_loss)<0:
            #     print("MALE")    
            if np.mean(new_loss) - np.mean(old_loss) < 1e-6:
                # print("EM terminato")
                break
        # print(np.mean(new_loss))
        return gmm

    def GMMTiedExpectationMaximization(self, X, gmm, psi):
        while True:
            num_components = len(gmm)
            logS = np.empty((num_components, X.shape[1]))
            # ---START of E step---
            for component_index in range(num_components):
                logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
                logS[component_index, :] += np.log(
                    gmm[component_index][0])  # computing the joint loglikelihood densities
            logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
            logSPost = logS - logSMarginal
            SPost = np.exp(logSPost)

            old_loss = logSMarginal
            # ---END of E step---
            # ---START of M step---
            Z = np.sum(SPost, axis=1)

            F = np.zeros((X.shape[0], num_components))
            for component_index in range(num_components):
                F[:, component_index] = np.sum(SPost[component_index, :] * X, axis=1)

            S = np.zeros((X.shape[0], X.shape[0], num_components))
            for component_index in range(num_components):
                S[:, :, component_index] = np.dot(SPost[component_index, :] * X, X.T)

            mu = F / Z
            cov_mat = S / Z

            for component_index in range(num_components):
                cov_mat[:, :, component_index] -= np.dot(vcol(mu[:, component_index]), vrow(mu[:, component_index]))
            w = Z / np.sum(Z)

            cov_updated = np.zeros(cov_mat[:, :, 0].shape)
            for component_index in range(num_components):
                cov_updated += (1 / num_samples(X)) * (Z[component_index] * cov_mat[:, :, component_index])

            U, s, _ = np.linalg.svd(cov_updated)  # avoiding degenerate solutions
            s[s < psi] = psi
            cov_updated = np.dot(U, vcol(s) * U.T)
            gmm = [(w[index_component], vcol(mu[:, index_component]), cov_updated) for index_component in
                   range(num_components)]

            new_loss = logpdf_GMM(X, gmm)
            # ---END of M step---
            # if np.mean(new_loss)-np.mean(old_loss)>0:
            #     # print("BENE")
            # if np.mean(new_loss)-np.mean(old_loss)<0:
            #     # print("MALE")      
            if np.mean(new_loss) - np.mean(old_loss) < 1e-6:
                # print("EM terminato")
                break
        # print(np.mean(new_loss))
        return gmm

    def LBG(self, X, psi, alpha=0.1, num_components_max=4):
        w_start = 1
        mu_start, C_start = ML_estimate(X)
        U, s, _ = np.linalg.svd(C_start)  # avoiding degenerate solutions
        s[s < psi] = psi
        C_start = np.dot(U, vcol(s) * U.T)

        gmm_start = []
        if num_components_max == 1:
            gmm_start.append((w_start, vcol(mu_start), C_start))
            return gmm_start
        new_w = w_start / 2
        U, s, Vh = np.linalg.svd(C_start)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        mu_new_1 = mu_start + d
        mu_new_2 = mu_start - d
        gmm_start.append((new_w, vcol(mu_new_1), C_start))
        gmm_start.append((new_w, vcol(mu_new_2), C_start))

        while True:
            gmm_start = self.GMMExpectationMaximization(X, gmm_start, psi)
            num_components = len(gmm_start)

            if num_components == num_components_max:
                break
            new_gmm = []
            for index_component in range(num_components):
                new_w = gmm_start[index_component][0] / 2
                U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_new_1 = gmm_start[index_component][1] + d
                mu_new_2 = gmm_start[index_component][1] - d
                new_gmm.append((new_w, vcol(mu_new_1), gmm_start[index_component][2]))
                new_gmm.append((new_w, vcol(mu_new_2), gmm_start[index_component][2]))
            gmm_start = new_gmm
        return gmm_start

    def Diag_LBG(self, X, psi, alpha=0.1, num_components_max=4):
        w_start = 1
        mu_start, C_start = ML_estimate(X)

        C_start = np.diag(np.diag(C_start))

        U, s, _ = np.linalg.svd(C_start)  # avoiding degenerate solutions
        s[s < psi] = psi
        C_start = np.dot(U, vcol(s) * U.T)

        gmm_start = []

        if num_components_max == 1:
            gmm_start.append((w_start, vcol(mu_start), C_start))
            return gmm_start

        new_w = w_start / 2
        U, s, Vh = np.linalg.svd(C_start)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        mu_new_1 = mu_start + d
        mu_new_2 = mu_start - d
        gmm_start.append((new_w, vcol(mu_new_1), C_start))
        gmm_start.append((new_w, vcol(mu_new_2), C_start))

        while True:
            gmm_start = self.GMMDiagExpectationMaximization(X, gmm_start, psi)
            num_components = len(gmm_start)
            if num_components == num_components_max:
                break
            new_gmm = []
            for index_component in range(num_components):
                new_w = gmm_start[index_component][0] / 2
                U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_new_1 = gmm_start[index_component][1] + d
                mu_new_2 = gmm_start[index_component][1] - d
                new_gmm.append((new_w, vcol(mu_new_1), gmm_start[index_component][2]))
                new_gmm.append((new_w, vcol(mu_new_2), gmm_start[index_component][2]))
            gmm_start = new_gmm
            # print("GMMMM START....")
            # print(len(gmm_start))
        return gmm_start

    def TiedCov_LBG(self, X, psi, alpha=0.1, num_components_max=4):
        w_start = 1
        mu_start, C_start = ML_estimate(X)

        U, s, _ = np.linalg.svd(C_start)  # avoiding degenerate solutions
        s[s < psi] = psi
        C_start = np.dot(U, vcol(s) * U.T)

        gmm_start = []

        if num_components_max == 1:
            gmm_start.append((w_start, vcol(mu_start), C_start))
            return gmm_start

        new_w = w_start / 2
        U, s, Vh = np.linalg.svd(C_start)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        mu_new_1 = mu_start + d
        mu_new_2 = mu_start - d
        gmm_start.append((new_w, vcol(mu_new_1), C_start))
        gmm_start.append((new_w, vcol(mu_new_2), C_start))
        while True:
            gmm_start = self.GMMTiedExpectationMaximization(X, gmm_start, psi)
            num_components = len(gmm_start)

            if num_components == num_components_max:
                break
            new_gmm = []
            for index_component in range(num_components):
                new_w = gmm_start[index_component][0] / 2
                U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_new_1 = gmm_start[index_component][1] + d
                mu_new_2 = gmm_start[index_component][1] - d
                new_gmm.append((new_w, vcol(mu_new_1), gmm_start[index_component][2]))
                new_gmm.append((new_w, vcol(mu_new_2), gmm_start[index_component][2]))
            gmm_start = new_gmm
        return gmm_start

    # def GMM_classifier(train_set, labels, num_classes,psi=0.01,alpha=0.1,num_components_max=[4,4]):
    #     model = []
    #     for index_class in range(num_classes):
    #         class_set = train_set[:,labels==index_class]
    #         gmm_c = LBG(class_set,psi,alpha,num_components_max[index_class])
    #         model.append(gmm_c)

    #     return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (gmm1)...(gmm-n)

    # def GMM_Diag_classifier(train_set, labels, num_classes,psi=0.01,alpha=0.1,num_components_max=[4,4]):
    #     model = []
    #     for index_class in range(num_classes):
    #         class_set = train_set[:,labels==index_class]
    #         gmm_c = Diag_LBG(class_set,psi,alpha,num_components_max[index_class])
    #         model.append(gmm_c)

    #     return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (gmm1)...(gmm-n)

    # def GMM_TiedCov_classifier(train_set, labels, num_classes,psi=0.01,alpha=0.1,num_components_max=[4,4]):
    #     model = []
    #     for index_class in range(num_classes):
    #         class_set = train_set[:,labels==index_class]
    #         gmm_c = TiedCov_LBG(class_set,psi,alpha,num_components_max[index_class])
    #         model.append(gmm_c)

    #     return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (gmm1)...(gmm-n)

    def predict_log(self, model, test_samples, prior):  # use this function to avoid numerical problems
        num_classes = len(model)
        # print("Len model: "+str(len(model)))
        likelihoods = []
        logSJoint = []
        # class_posterior_probs = []
        for index_class in range(num_classes):
            gmm_c = model[index_class]
            likelihood_c = logpdf_GMM(test_samples, gmm_c)
            likelihoods.append(likelihood_c)
            logSJoint.append(likelihood_c + np.log(prior[index_class]))

        logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        llr = logSPost[1, :] - logSPost[0, :] - np.log(prior[1] / prior[0])
        # SPost = np.exp(logSPost)    

        # predicted_label = np.argmax(np.array(SPost),axis=0)

        return llr

def accuracy(predicted_labels,original_labels):
    if len(predicted_labels)!=len(original_labels):
        return
    total_samples = len(predicted_labels)


    correct = (predicted_labels==original_labels).sum()
    return (correct/total_samples)*100


def error_rate(predicted_labels,original_labels):
    return 100 - accuracy(predicted_labels,original_labels)



