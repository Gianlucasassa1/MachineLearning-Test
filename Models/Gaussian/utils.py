import numpy as np
import scipy as sc


def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


def createCov(D):
    mu = 0
    C = 0
    mu = D.mean(1)
    for i in range(D.shape[1]):
        C = C + np.dot(D[:, i:i + 1] - mu, (D[:, i:i + 1] - mu).T)  # scalar product using numpy
        # with this formule we have just centered the data (PCA on NON CENTERED DATA is quite an unsafe operation)

    C = C / float(D.shape[1])  # where the divider is the dimension N of our data
    return C


def createCenteredCov(DC):  # for centered data yet
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:, i:i + 1], (DC[:, i:i + 1]).T)
    C = C / float(DC.shape[1])
    return C


def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)  # broadcasting applied
    return DC


# logpdf_GAU_ND algorithm for an array(just one sample at time)
def logpdf_GAU_ND_1Sample(x, mu, C):
    # it seems that also for just one row at time we should use mu and C of the whole matrix
    xc = x - mu
    M = x.shape[0]
    logN = 0
    const = - 0.5 * M * np.log(2 * np.pi)
    log_determ = np.linalg.slogdet(C)[1]
    lamb = np.linalg.inv(C)
    third_elem = np.dot(xc.T, np.dot(lamb, xc)).ravel()
    logN = const - 0.5 * log_determ - 0.5 * third_elem

    return logN


def logpdf_GAU_ND(X, mu, C):  # logpdf_GAU_ND algorithm for a 2-D matrix
    logN = []
    # print("Dim di X in logpdf_GAU_ND: "+str(X.shape))
    for i in range(X.shape[1]):
        # [:,i:i+1] notation allows us to take just the i-th column at time
        # remember that with this notation we mean [i,i+1) (left-extreme not included)
        logN.append(logpdf_GAU_ND_1Sample(X[:, i:i + 1], mu, C))
    return np.array(logN).ravel()


def loglikelihood(X, m, C):
    logN = 0
    logi = logpdf_GAU_ND(X, m, C)
    for i in range(logi.shape[0]):
        # [:,i:i+1] notation allows us to take just the i-th column at time
        # remember that with this notation we mean [i,i+1) (left-extreme not included)

        logN += logi[i]

    return logN


def conf_plot(X, m, C):
    plt.figure()
    plt.hist(X.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))


# Prova funzioni :

#    X1D = np.load("./X1D.npy") 
#    XND = np.load("./XND.npy")

#    X1D_cent=centerData(X1D)
#    XND_cent = centerData(XND)
#    m_ML = XND.mean(1)
#    m_ML = vcol(m_ML)
#    m_ML2 = vcol(X1D.mean(1))
#    C_ML = createCenteredCov(XND_cent)
#    C_ML2= createCenteredCov(X1D_cent)

#    p = logpdf_GAU_ND(XND_cent, m_ML, C_ML)

#    ll = loglikelihood(XND,m_ML,C_ML)
#    ll2 = loglikelihood(X1D,m_ML2,C_ML2)

#    conf_plot(X1D,m_ML2,C_ML2)


def MVG_model(D, L):
    c0 = []
    c1 = []
    means = []
    S_matrices = []

    for i in range(D.shape[1]):
        if L[i] == 0:
            c0.append(D[:, i])
        elif L[i] == 1:
            c1.append(D[:, i])

    c0 = (np.array(c0)).T
    c1 = (np.array(c1)).T

    c0_cent = centerData(c0)
    c1_cent = centerData(c1)

    # you can find optimizations for this part in Lab03

    S_matrices.append(createCenteredCov(c0_cent))
    S_matrices.append(createCenteredCov(c1_cent))

    means.append(vcol(c0.mean(1)))
    means.append(vcol(c1.mean(1)))

    return means, S_matrices, (c0.shape[1], c1.shape[1])


def TCG_model(D, L):
    S_matrix = 0
    means, S_matrices, cN = MVG_model(D, L)

    cN = np.array(cN)

    S_matrices = np.array(S_matrices)

    D_cent = centerData(D)

    for i in range(cN.shape[0]):
        S_matrix += cN[i] * S_matrices[i]

    S_matrix /= D.shape[1]

    return means, S_matrix


def loglikelihoods(DTE, means, S_matrices, prior):
    likelihoods = []
    logSJoint = []
    logSMarginal = 0
    for i in range(2):
        mu = means[i]
        c = S_matrices[i]
        ll = logpdf_GAU_ND(DTE, mu, c)
        likelihoods.append(ll)
        logSJoint.append(ll + np.log(prior[i]))

    logSMarginal = vrow(sc.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1, :] - logSPost[0, :] - np.log(prior[1] / prior[0])

    # ll0 = []
    # ll1 = []
    # for i in range(DTE.shape[1]):
    #         ll0.append(loglikelihood(DTE[:,i:i+1] , means[0], S_matrices[0]))

    #         ll1.append(loglikelihood(DTE[:,i:i+1], means[1], S_matrices[1]))

    return llr


def posterior_prob(SJoint):
    # Calcola le densità marginali sommando le probabilità congiunte su tutte le classi
    SMarginal = vrow(SJoint.sum(axis=0))

    # Calcola le probabilità posteriori di classe dividendo le probabilità congiunte per le densità marginali
    SPost = SJoint / SMarginal

    # Calcola l'array delle etichette previste utilizzando il metodo argmax con la parola chiave axis
    pred = np.argmax(SPost, axis=0)

    return pred


def log_post_prob(log_SJoint):
    # log_SMarginal_sol = np.load('logMarginal_MVG.npy')
    log_SMarginal = vrow(sc.special.logsumexp(log_SJoint, axis=0))
    # print(np.abs(log_SMarginal - log_SMarginal_sol).max())

    # print(log_SMarginal.shape)
    # print(log_SJoint.shape)
    log_SPost = log_SJoint - log_SMarginal
    # log_SPost_sol = np.load('logPosterior_MVG.npy')

    log_pred = np.argmax(log_SPost, axis=0)

    return log_pred


def accuracy(pred, LTE):
    mask = (pred == LTE)

    mask = np.array(mask, dtype=bool)

    corr = np.count_nonzero(mask)
    tot = LTE.shape[0]

    acc = float(corr) / tot

    return acc, tot - corr


def MVG_approach(D, L, Pc, DTE, LTE):
    # remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes

    # using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means, S_matrices, _ = MVG_model(D, L)  # 3 means and 3 S_matrices -> 1 for each class (3 classes)

    # we create a NxNc matrix with the log-likelihoods elements
    # each row represents a class and each column represents a sample
    # so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    log_score_matrix = loglikelihoods(DTE, means, S_matrices)

    # adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    # Pc is passed as argument
    # for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint
    sm_joint = np.exp(log_score_matrix) * Pc
    # SJoint_sol = np.load('SJoint_MVG.npy')
    log_sm_joint = log_score_matrix + np.log(Pc)
    # log_sm_joint_sol = np.load('logSJoint_MVG.npy')

    # let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    # be careful! These functions below return prediction labels yet! The Posterio probability
    # computation is made inside the functions!
    pred = posterior_prob(sm_joint)
    log_pred = log_post_prob(log_sm_joint)

    # simple function to evaluate the accuracy of our model
    # acc,_ = accuracy(pred,LTE)  
    # acc_2,_=accuracy(log_pred,LTE)
    # inacc = 1-acc

    return log_pred


def MVG_llr(D, L, DTE):
    # remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes

    # using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means, S_matrices, _ = MVG_model(D, L)  # 3 means and 3 S_matrices -> 1 for each class (3 classes)

    # we create a NxNc matrix with the log-likelihoods elements
    # each row represents a class and each column represents a sample
    # so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    ll0, ll1 = loglikelihoods(DTE, means, S_matrices)
    return ll1 - ll0


def NB_approach(D, L, Pc, DTE, LTE):
    # remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes

    # using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means, S_matrices, _ = MVG_model(D, L)  # 3 means and 3 S_matrices -> 1 for each class (3 classes)

    for i in range(np.array(S_matrices).shape[0]):
        S_matrices[i] = S_matrices[i] * np.eye(S_matrices[i].shape[0], S_matrices[i].shape[1])

    # we create a NxNc matrix with the log-likelihoods elements
    # each row represents a class and each column represents a sample
    # so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    log_score_matrix = loglikelihoods(DTE, means, S_matrices)

    # adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    # Pc = passed as argument
    # for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint
    sm_joint = np.exp(log_score_matrix) * Pc
    # SJoint_sol = np.load('SJoint_MVG.npy')
    log_sm_joint = log_score_matrix + np.log(Pc)
    # log_sm_joint_sol = np.load('logSJoint_MVG.npy')

    # let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    # be careful! These functions below return prediction labels yet! The Posterior probability
    # computation is made inside the functions!
    pred = posterior_prob(sm_joint)
    log_pred = log_post_prob(log_sm_joint)

    # simple function to evaluate the accuracy of our model
    acc, _ = accuracy(pred, LTE)
    acc_2, _ = accuracy(log_pred, LTE)
    inacc = 1 - acc

    return log_pred


def NB_llr(D, L, DTE):
    # remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes

    # using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means, S_matrices, _ = MVG_model(D, L)  # 3 means and 3 S_matrices -> 1 for each class (3 classes)

    for i in range(np.array(S_matrices).shape[0]):
        S_matrices[i] = S_matrices[i] * np.eye(S_matrices[i].shape[0], S_matrices[i].shape[1])

    # we create a NxNc matrix with the log-likelihoods elements
    # each row represents a class and each column represents a sample
    # so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    ll0, ll1 = loglikelihoods(DTE, means, S_matrices)
    return ll1 - ll0


def TCG_approach(D, L, Pc, DTE, LTE):
    means, S_matrix = TCG_model(D,
                                L)  # 3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes

    # to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix
    S_matrices = [S_matrix, S_matrix, S_matrix]

    log_score_matrix = loglikelihoods(DTE, means, S_matrices)

    # adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    # Pc passed as argument
    # for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint
    sm_joint = np.exp(log_score_matrix) * Pc
    # SJoint_sol = np.load('SJoint_TiedMVG.npy')

    log_sm_joint = log_score_matrix + np.log(Pc)
    # log_sm_joint_sol = np.load('logSJoint_TiedMVG.npy')

    # log_marginal_sol = np.load('logMarginal_TiedMVG.npy')

    # let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    # be careful! These functions below return prediction labels yet! The Posterio probability
    # computation is made inside the functions!
    pred = posterior_prob(sm_joint)
    log_pred = log_post_prob(log_sm_joint)

    # log_SPost_sol=np.load('logPosterior_TiedMVG.npy')
    # SPost_sol=np.load('Posterior_TiedMVG.npy')

    # simple function to evaluate the accuracy of our model
    acc, _ = accuracy(pred, LTE)
    acc_2, _ = accuracy(log_pred, LTE)
    inacc = 1 - acc

    return log_pred


def TCG_llr(D, L, DTE):
    means, S_matrix = TCG_model(D,
                                L)  # 3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes

    # to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix
    S_matrices = [S_matrix, S_matrix, S_matrix]

    ll0, ll1 = loglikelihoods(DTE, means, S_matrices)
    return ll1 - ll0


def TCNBG_approach(D, L, Pc, DTE, LTE):
    means, S_matrix = TCG_model(D,
                                L)  # 3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes

    S_matrix = S_matrix * np.eye(S_matrix.shape[0], S_matrix.shape[1])
    # to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix
    S_matrices = [S_matrix, S_matrix, S_matrix]

    log_score_matrix = loglikelihoods(DTE, means, S_matrices)

    # adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    # Pc passed as argument
    # for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint
    sm_joint = np.exp(log_score_matrix) * Pc
    # SJoint_sol = np.load('SJoint_TiedMVG.npy')

    log_sm_joint = log_score_matrix + np.log(Pc)
    # log_sm_joint_sol = np.load('logSJoint_TiedMVG.npy')

    # log_marginal_sol = np.load('logMarginal_TiedMVG.npy')

    # let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    # be careful! These functions below return prediction labels yet! The Posterio probability
    # computation is made inside the functions!
    pred = posterior_prob(sm_joint)
    log_pred = log_post_prob(log_sm_joint)

    # log_SPost_sol=np.load('logPosterior_TiedMVG.npy')
    # SPost_sol=np.load('Posterior_TiedMVG.npy')

    # simple function to evaluate the accuracy of our model
    acc, _ = accuracy(pred, LTE)
    acc_2, _ = accuracy(log_pred, LTE)
    inacc = 1 - acc

    return log_pred


def LOO(D, L):
    MVG_pred = []
    NB_pred = np.zeros(D.shape[1])
    TCG_pred = np.zeros(D.shape[1])
    TCNBG_pred = np.zeros(D.shape[1])

    for i in range(D.shape[1]):
        # K-fold Leave One Out method adopted to slit up training set into training set and validation
        DTE = D[:, i:i + 1]  # 1 sample of the dataset used for testing and the other for testing

        # it deletes a single sample at time
        DTR = np.delete(D, i, axis=1)
        LTR = np.delete(L, i)
        LTE = L[i:i + 1]

        pred_LOO_MVG = MVG_approach(DTR, LTR, DTE)[0]
        MVG_pred.append(pred_LOO_MVG)

        pred_LOO_NB = NB_approach(DTR, LTR, DTE)
        NB_pred[i] = pred_LOO_NB

        pred_LOO_TCG = TCG_approach(DTR, LTR, DTE)
        TCG_pred[i] = pred_LOO_TCG

        pred_LOO_TCNBG = TCNBG_approach(DTR, LTR, DTE)
        TCNBG_pred[i] = pred_LOO_TCNBG

    return np.array(MVG_pred), np.array(NB_pred), np.array(TCG_pred), np.array(TCNBG_pred)

# Prova funzioni:


#     pred_MVG = MVG_approach(DTR,LTR,DTE)

#     #for so few data we can apply the same code as used for MVG approach and make the S_matrices diagonal using a np.eye()
#     pred_Naive_Bayes = NB_approach(DTR,LTR,DTE)

#     pred_Tied_cov_Gauss = TCG_approach(DTR,LTR,DTE)

#     #accuracy evaluation system
#     print(accuracy(pred_MVG,LTE)[0]*100)
#     print(accuracy(pred_Naive_Bayes,LTE)[0]*100)
#     print(accuracy(pred_Tied_cov_Gauss,LTE)[0]*100)


#     #let's try with Leave One Out EVALUATION system (Leave One Out variant)

#     #NOT SURE THIS IS THE RIGHT SOLUTION EXPECIALLY ABOUT WHAT KIND OF DATASET WE NEED TO USE!!!!!
#     MVG_pred,NB_pred,TCG_pred,TCNBG_pred = LOO(DTR,LTR)

#     MVG_acc,MVG_err = accuracy(MVG_pred, L)

#     NB_acc,NB_err = accuracy(NB_pred,L)

#     TCG_acc,TCG_err = accuracy(TCG_pred, L)

#     TCNBG_acc,TCNBG_err = accuracy(TCNBG_pred, L)

#     print("LLO_MVG_err_ratio: ",(1-MVG_acc)*100)
#     print("LLO_NB_err_ratio: ",(1-NB_acc)*100)
#     print("LLO_TCG_err_ratio: ",(1-TCG_acc)*100)
#     print("LLO_TCNBG_err_ratio: ",(1-TCNBG_acc)*100)
