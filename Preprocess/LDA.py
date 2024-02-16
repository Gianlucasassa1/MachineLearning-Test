import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy as sp

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA_plot_skilearn(D, L):
    # Creare un'istanza della classe LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()

    # Adattare il modello ai dati
    lda.fit(D, L)

    # Proiettare i dati sulla direzione LDA
    X_lda = lda.transform(D)
    plt.hist(X_lda, bins=10)
    plt.title('LDA Dataset Characteristics')
    # plt.show()
    path = f'Images/Dataset/LDA/LDA'
    os.makedirs(path, exist_ok=True)
    plt.title(f'LDA')
    plt.savefig(path)


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


def createCenteredSWc(DC):  # for already centered data
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:, i:i + 1], (DC[:, i:i + 1]).T)
    C = C / float(DC.shape[1])
    return C


def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)  # broadcasting applied
    return DC


def createSBSW(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    DC0 = centerData(D0)
    DC1 = centerData(D1)

    SW0 = createCenteredSWc(DC0)
    SW1 = createCenteredSWc(DC1)

    centeredSamples = [DC0, DC1]
    allSWc = [SW0, SW1]

    samples = [D0, D1]
    mu = vcol(D.mean(1))

    SB = 0
    SW = 0

    for x in range(2):
        m = vcol(samples[x].mean(1))
        SW = SW + (allSWc[x] * centeredSamples[x].shape[1])
        SB = SB + samples[x].shape[1] * np.dot((m - mu), (
                m - mu).T)  # here we don't use centered samples because we apply a covariance between classed
        # and we take the mean off in the formula yet

    SB = SB / float(D.shape[1])
    SW = SW / float(D.shape[1])

    return SB, SW


def LDA1(D, L, m):
    SB, SW = createSBSW(D, L)
    s, U = sp.linalg.eigh(SB, SW)  # heigbert generalization eigenvectors scomposition
    W = U[:, ::-1][:, 0:m]  # we must take the first m columns of U matrix
    return W


def LDA(D, L, m):
    # compute matrices S_w , S_b with the following function
    S_w, S_b = compute_Sw_Sb(D, L)
    # compute eigenvectors of S_b ^( -1) * S_w
    _, U = scipy.linalg.eigh(S_b, S_w)
    # take eigenvecs associated with the m biggest eigenvals
    W = U[:, ::-1][:, :m]
    # return the data projected on the m - dimensional subspace
    return numpy.dot(W.T, D)


def compute_Sw_Sb(D, L):
    # retrieve the number of classes | (0, 1 , 2) -> 3
    num_classes = L.max() + 1
    # separate the data into classes
    D_c = [D[:, L == i] for i in range(num_classes)]
    # get the number of elements for each class
    n_c = [D_c[i].shape[1] for i in range(num_classes)]
    # compute the dataset mean
    mu = vcol(D.mean(1))
    # compute the mean for each class
    mu_c = [vcol(D_c[i].mean(1)) for i in range(len(D_c))]
    # compute S_w and S_b as previously explained
    S_w, S_b = 0, 0
    for i in range(num_classes):
        DC = D_c[i] - mu_c[i]
        C_i = numpy.dot(DC, DC.T) / DC.shape[1]
        S_w += n_c[i] * C_i
        diff = mu_c[i] - mu
        S_b += n_c[i] * numpy.dot(diff, diff.T)
    S_w /= D.shape[1]
    S_b /= D.shape[1]
    # return S_w and S_b
    return S_w, S_b


def LDA2(D, L, m):
    SB, SW = createSBSW(D, L)
    U, s, _ = np.linalg.svd(SW)
    P1 = np.dot(U, vcol(
        1.0 / (s ** 0.5)) * U.T)  # first transformation (whitening transformation) to apply a samples "CENTRIFICATION"
    SBtilde = np.dot(P1, np.dot(SB, P1.T))
    U, _, _ = np.linalg.svd(SBtilde)
    P2 = U[:, 0:m]  # second tranformation (samples rotation) to obtain SB diagonalization

    return np.dot(P1.T, P2)


def LDA_impl(D, L, m):
    W1 = LDA(D, L, m) # We tried different versions of implementations of LDA
    DW = np.dot(W1.T, D)  # D projection on sub-space W1
    return DW, W1
