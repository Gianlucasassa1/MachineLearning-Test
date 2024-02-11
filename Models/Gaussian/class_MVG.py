from . import new_MVG_model as model_utils
import numpy as np

class GaussianClassifier:
    """
    Classe per la classificazione basata su modelli Gaussiani.
    Supporta diverse varianti del modello Gaussiano.
    """
    def __init__(self, mode, prior, cost_fp, cost_fn):
        """
        Inizializza il classificatore.

        Args:
        mode: Stringa che indica la modalità del modello ("MVG", "NB", "TCG", "TCGNB").
        prior: Probabilità a priori della classe positiva.
        cost_fp: Costo di un falso positivo.
        cost_fn: Costo di un falso negativo.
        """
        self.mode = mode
        self.prior_probability = prior
        # Calcola la priorità effettiva considerando i costi di errore
        self.effective_prior = (prior * cost_fn) / (prior * cost_fn + (1 - prior) * cost_fp)
        self.means = None  # Vettore delle medie per ciascuna classe
        self.covariance_matrices = None  # Lista delle matrici di covarianza per ciascuna classe

    def train(self, DTR, LTR):
        """
        Allena il modello sui dati di training forniti.

        Args:
        DTR: Dati di training (features).
        LTR: Etichette dei dati di training.
        """

        if self.mode == "MVG":
            self.means, self.covariance_matrices, _ = model_utils.MVG_model(DTR, LTR)

        elif self.mode == "NB":
            self.means, covariance_matrices_temp, _ = model_utils.MVG_model(DTR, LTR)
            for i in range(np.array(covariance_matrices_temp).shape[0]):
                covariance_matrices_temp[i] = covariance_matrices_temp[i]*np.eye(covariance_matrices_temp[i].shape[0],covariance_matrices_temp[i].shape[1])
            self.covariance_matrices = covariance_matrices_temp

        elif self.mode == "TCG":
            # Per TCG, utilizza una matrice di covarianza condivisa tra le classi
            self.means, covariance_matrices_temp = model_utils.TCG_model(DTR, LTR)
            self.covariance_matrices = [covariance_matrices_temp] * 3
        
        elif self.mode == "TCGNB":
            self.means, covariance_matrices_temp = model_utils.TCG_model(DTR, LTR)
            covariance_matrices_temp = covariance_matrices_temp * np.eye(covariance_matrices_temp.shape[0],covariance_matrices_temp.shape[1])
            self.covariance_matrices = [covariance_matrices_temp] * 3

    def compute_scores(self, DTE):
        """
        Calcola i log-likelihood ratios per i dati di test.

        Args:
        DTE: Dati di test (features).

        Returns:
        Log-likelihood ratios per i dati di test.
        """
        llr = model_utils.loglikelihoods(DTE, self.means, self.covariance_matrices, [1 - self.effective_prior, self.effective_prior])
        return llr

    def predict(self, class_probabilities):
        """
        Effettua predizioni basate sui log-likelihood ratios e sulle probabilità a priori delle classi.

        Args:
        class_probabilities: Probabilità a priori delle classi.

        Returns:
        Etichette predette per i dati di test.
        """
        log_sum_joint = np.array((self.ll0, self.ll1)) + np.log(class_probabilities)
        predictions = model_utils.log_post_prob(log_sum_joint)
        return predictions
