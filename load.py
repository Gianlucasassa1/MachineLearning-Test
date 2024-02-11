import sys
import numpy as np
import string
import scipy.special
import itertools

def loadTrainingAndTestData(train_path, test_path):
    """
    Carica i dati di training e test da file.

    Args:
    train_path: Percorso al file dei dati di training.
    test_path: Percorso al file dei dati di test.

    Returns:
    Una tupla di tuple: ((Dati di training, Etichette di training), (Dati di test, Etichette di test))
    """
    # Apertura dei file
    train_file = open(train_path, 'r')
    test_file = open(test_path, 'r')

    # Inizializzazione delle liste per i dati e le etichette
    
    DTR = []
    DTE = []
    LTE = []
    LTR = []

    # Lettura dei dati di training
    for line in train_file:
        splitted = line.split(',')
        DTR.append([float(i) for i in splitted[:-1]])
        LTR.append(int(splitted[-1]))
    DTR = np.array(DTR)
    LTR = np.array(LTR)
    
    # Lettura dei dati di test
    for line in test_file:
        splitted = line.split(',')
        DTE.append([float(i) for i in splitted[:-1]])
        LTE.append(int(splitted[-1]))
    DTE = np.array(DTE)
    LTE = np.array(LTE)
    
    # Chiusura dei file
    train_file.close()
    test_file.close()
    
    return (DTR, LTR), (DTE, LTE)

def mcol(v):
    """
    Trasforma un array 1D in una colonna (2D).

    Args:
    v: Array 1D.

    Returns:
    Array 2D con una sola colonna.
    """
    return v.reshape((v.size, 1))

def loadFile(filename):
    """
    Carica i dati da un singolo file, trasformandoli in una matrice di attributi e un vettore di etichette.

    Args:
    filename: Percorso al file da caricare.

    Returns:
    Una tupla (Dati, Etichette), dove i dati sono in formato matrice e le etichette sono un array.
    """
    data_list = []
    labels_list = []
    # Mappatura delle etichette testuali a valori numerici
    labels_mapping = {
        'Male': 0,
        'Female': 1,
    }

    with open(filename) as file:
        for line in file:
            try:
                # Estrazione degli attributi e conversione in array numpy
                attributes = line.split(',')[0:4]
                attributes = mcol(np.array([float(i) for i in attributes]))
                # Estrazione dell'etichetta testuale e conversione in valore numerico
                label_name = line.split(',')[-1].strip()
                label = labels_mapping[label_name]
                data_list.append(attributes)
                labels_list.append(label)
            except ValueError:
                # Gestisce linee malformate ignorandole
                pass

    # Concatenazione degli array di attributi e conversione delle etichette in array numpy
    return np.hstack(data_list), np.array(labels_list, dtype=np.int32)
