import numpy as np
import scipy as sc


# Definizione di funzioni di utilità per convertire array in vettori riga e colonna
def vrow(col):
    return col.reshape((1, col.size))


def vcol(row):
    return row.reshape((row.size, 1))


class QuadraticLR_Classifier:

    def __init__(self, lambda_, piT):
        # Inizializzazione degli iperparametri del modello
        self.lambda_ = lambda_
        self.piT = piT  # Peso per la classe positiva

    # Funzione per calcolare il gradiente della funzione obiettivo della Quadratic Logistic Regression
    def gradient_test(self, DTR, LTR, l, pt, nt, nf):
        z = np.empty((LTR.shape[0]))
        z = 2 * LTR - 1  # Trasformazione delle etichette in valori z = {-1, 1}

        def gradient(v):
            w, b = v[0:-1], v[-1]
            first_term = l * w  # Termine di regolarizzazione L2
            second_term = 0  # Termine per i campioni della classe positiva
            third_term = 0  # Termine per i campioni della classe negativa

            for i in range(DTR.shape[1]):
                S = np.dot(w.T, DTR[:, i]) + b
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:
                    internal_term = np.dot(np.exp(-ziSi), (np.dot(-z[i], DTR[:, i]))) / (1 + np.exp(-ziSi))
                    second_term += internal_term
                else:
                    internal_term_2 = np.dot(np.exp(-ziSi), (np.dot(-z[i], DTR[:, i]))) / (1 + np.exp(-ziSi))
                    third_term += internal_term_2

            derivative_w = first_term + (pt / nt) * second_term + (1 - pt) / nf * third_term
            derivative_b = (pt / nt) * first_term + (1 - pt) / nf * second_term
            grad = np.hstack((derivative_w, derivative_b))
            return grad

        return gradient

    # Funzione per calcolare la funzione obiettivo della Quadratic Logistic Regression (da minimizzare)
    def quad_logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        w = vcol(w)
        n = self.fi_x.shape[1]
        regularization = (self.lambda_ / 2) * np.sum(w ** 2)
        loss_c0 = 0
        loss_c1 = 0

        # Calcolo della loss per ciascuna classe
        for i in range(n):
            if self.LTR[i] == 1:
                zi = 1
                loss_c1 += np.logaddexp(0, -zi * (np.dot(w.T, self.fi_x[:, i:i + 1]) + b))
            else:
                zi = -1
                loss_c0 += np.logaddexp(0, -zi * (np.dot(w.T, self.fi_x[:, i:i + 1]) + b))

        # Calcolo della funzione obiettivo
        J = regularization + (self.piT / self.nT) * loss_c1 + (1 - self.piT) / self.nF * loss_c0
        grad = self.grad_funct(v)
        return J, grad

    # Funzione per addestrare il modello Quadratic Logistic Regression
    def train(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        self.nt = DTR[:, LTR == 1].shape[1]
        self.nf = DTR.shape[1] - self.nt

        # Calcolo delle feature espansione
        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        expanded_DTR = np.apply_along_axis(vecxxT, 0, DTR)
        self.fi_x = np.vstack([expanded_DTR, DTR])

        # Inizializzazione dei parametri e del gradiente
        x0 = np.zeros(self.fi_x.shape[0] + 1)
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        self.grad_funct = self.gradient_test(self.fi_x, self.LTR, self.lambda_, self.piT, self.nt, self.nf)

        # Ottimizzazione dei parametri utilizzando L-BFGS
        params, _, _ = sc.optimize.fmin_l_bfgs_b(self.quad_logreg_obj, x0)
        self.b = params[-1]
        self.w = np.array(params[0:-1])
        self.S = []

        return self.b, self.w

    # Funzione per calcolare i punteggi predetti dal modello
    def compute_scores(self, DTE):
        S = self.S
        for i in range(DTE.shape[1]):
            x = vcol(DTE[:, i:i + 1])
            mat_x = np.dot(x, x.T)
            vec_x = vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x, x))
            self.S.append(np.dot(self.w.T, fi_x) + self.b)

        pred = [1 if x > 0 else 0 for x in S]  # Classificazione in base ai punteggi

        return S


class LR_Classifier:

    def __init__(self, lambda_, piT):
        self.lambda_ = lambda_
        self.piT = piT

    # Funzione obiettivo per la regressione logistica
    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        w = vcol(w)
        regularization = (self.lambda_ / 2) * np.sum(w ** 2)
        loss_c0 = 0
        loss_c1 = 0

        for i in range(self.DTR.shape[1]):
            if self.LTR[i] == 1:
                zi = 1
                loss_c1 += np.logaddexp(0, -zi * (np.dot(w.T, self.DTR[:, i:i + 1]) + b))
            else:
                zi = -1
                loss_c0 += np.logaddexp(0, -zi * (np.dot(w.T, self.DTR[:, i:i + 1]) + b))

                J = regularization + (self.piT / self.nT) * loss_c1 + (1 - self.piT) / self.nF * loss_c0
        return J

    # Funzione per addestrare il modello di regressione logistica
    def train(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)

        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])

        # Ottimizzazione dei parametri utilizzando L-BFGS
        params, _, _ = sc.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad=True)
        self.b = params[-1]
        self.w = np.array(params[0:-1])
        self.S = []

        return self.b, self.w

    # Funzione per calcolare i punteggi predetti dal modello
    def compute_scores(self, DTE):
        S = []
        for i in range(DTE.shape[1]):
            x = DTE[:, i:i + 1]
            x = np.array(x)
            x = x.reshape((x.shape[0], 1))
            S.append(np.dot(self.w.T, x) + self.b)

        S = [1 if x > 0 else 0 for x in S]  # Classificazione in base ai punteggi
        llr = np.dot(self.w.T, DTE) + self.b

        return llr

# if __name__ == "__main__":
#     #Numerical optimizations tries
#     x,f_min,d = sc.optimize.fmin_l_bfgs_b(f, (0,0),approx_grad=True);
    
#     x2,f_min2,d2 = sc.optimize.fmin_l_bfgs_b(f_2, (0,0))
    
#     #binary classificator with Logistic Regression applied to the Iris dataset
#     l = 0.000001
#     D, L = load_iris_binary()
#     (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

# def logreg_impl(DTR,LTR,DTE,l):
#         x0 = np.zeros(DTR.shape[0] + 1)
        
        
#         logRegObj = logRegClass(DTR, LTR, l) #I created an object logReg with logreg_obj inside
        
#         #optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
#         #I set approx_grad=True so the function will generate an approximated gradient for each iteration
#         params,f_min,_ = sc.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0,approx_grad=True)
        
#         #the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
        
#         b = params[-1]
        
#         w = np.array(params[0:-1])
        
#         S = []
        
#         #I apply the model just trained to classify the test set samples
#         for i in range(DTE.shape[1]):
#             x = DTE[:,i:i+1]
#             x = np.array(x)
#             x = x.reshape((x.shape[0],1))
#             S.append(np.dot(w.T,x) + b)
        
#         S = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
#         # acc = accuracy(S,LTE)
        
#         # print(100-acc)
        
#         return S