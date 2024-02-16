import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def mcol(v):
    return v.reshape((v.size, 1))


def featurePlot(D, L, m, type):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    print(m)
    for i in range(m):
        plt.figure()
        # plt.xlabel("Feature " + str(i+1))
        plt.ylabel("Number of elements")
        plt.hist(D0[i, :], bins=60, density=True, alpha=0.7, label="Male")
        plt.hist(D1[i, :], bins=60, density=True, alpha=0.7, label="Female")
        plt.legend()
        # plt.show()
        path = f'Images/Dataset/Features_{type}/'
        os.makedirs(path, exist_ok=True)
        if type == "PCA":
            plt.title(f' PCA ')
        elif type == "LDA":
            plt.title(f' LDA ')
        else:
            plt.title(f'Feature {i + 1}')

        plt.savefig(os.path.join(path, f'Feature {i + 1}.png'))
        plt.close()


def generalPlot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.figure()
    plt.xlabel("Feature")
    plt.ylabel("Number of elements")
    plt.hist(D0[:, :], density=True, alpha=0.7, label="Male")
    plt.hist(D1[:, :], density=True, alpha=0.7, label="Female")
    plt.legend()
    path = f'Images/Dataset/General/'
    os.makedirs(path, exist_ok=True)
    plt.title(f'Dataset Features')
    plt.savefig(os.path.join(path, f'General.png'))
    plt.close()


def mixedPlot(D, L, m, type):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            plt.figure()
            plt.xlabel("Feature " + str(i + 1))
            plt.ylabel("Feature " + str(j + 1))
            plt.scatter(D0[i, :], D0[j, :], label="Male")
            plt.scatter(D1[i, :], D1[j, :], label="Female")
            plt.legend()
            # plt.show()
            path = f'Images/Dataset/Cross_{type}/Feature{i}/'
            os.makedirs(path, exist_ok=True)
            plt.title(f'Feature {i + 1}, {j + 1}')
            plt.savefig(os.path.join(path, f'Feature {i + 1}, {j + 1}.png'))
            plt.close()


# P=Pearson
def computeCorrelationP(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    diff_x = x - mean_x
    diff_y = y - mean_y
    diff_prod = diff_x * diff_y
    sum_diff_squares = np.sqrt(np.sum(diff_x ** 2) * np.sum(diff_y ** 2))
    correlation = np.sum(diff_prod) / sum_diff_squares
    return correlation


def correlationPlotP(data, labels, target_class):
    target_data = data[:, labels == target_class]
    non_target_data = data[:, labels != target_class]
    num_features = data.shape[0]
    correlations_target = np.zeros((num_features, num_features))
    correlations_non_target = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            correlations_target[i, j] = computeCorrelationP(target_data[:, i], target_data[:, j])
            correlations_non_target[i, j] = computeCorrelationP(non_target_data[:, i], non_target_data[:, j])

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    im1 = axs[0].matshow(correlations_target, cmap='coolwarm', vmin=-1, vmax=1)
    axs[0].set_title('Target Class')
    im2 = axs[1].matshow(correlations_non_target, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].set_title('Non-Target Class')
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])
    fig.suptitle('Pearson Correlation Coefficient')
    # plt.show()
    path = f'Images/Dataset/Pearson/'
    os.makedirs(path, exist_ok=True)
    plt.title(f'Pearson Correlation')
    plt.savefig(os.path.join(path, 'PearsonCorrelation.png'))
    plt.close()


def maleFemaleFeaturesPlot(DTR, LTR, m=2, appendToTitle=''):
    correlationPlot(DTR, "Dataset" + appendToTitle, cmap="Greys")
    correlationPlot(DTR[:, LTR == 0], "Male" + appendToTitle, cmap="Blues")
    correlationPlot(DTR[:, LTR == 1], "Female" + appendToTitle, cmap="Reds")


def heatmapPlot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    hFea = {
        0: 'Feature_0',
        1: 'Feature_1',
        2: 'Feature_2',
        3: 'Feature_3',
        4: 'Feature_4',
        5: 'Feature_5',
        6: 'Feature_6',
        7: 'Feature_7',
        8: 'Feature_8',
        9: 'Feature_9',
        10: 'Feature_10',
        11: 'Feature_11',
        12: 'Feature_12',
    }

    corr_matrix = np.corrcoef(D1)

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(corr_matrix, cmap='seismic')
    plt.colorbar()

    ax.set_xticks(np.arange(len(corr_matrix)))
    ax.set_yticks(np.arange(len(corr_matrix)))
    ax.set_xticklabels(np.arange(len(corr_matrix)))
    ax.set_yticklabels(np.arange(len(corr_matrix)))

    # plt.title('Pearson Correlation Heatmap - \'Females\' training set')
    # plt.savefig('heatmap_training_set_authentic.png', dpi=300)
    # plt.show()

    plt.legend()
    path = f'Images/Dataset/Pearson/'
    os.makedirs(path, exist_ok=True)
    plt.title(f'Pearson Heatmap V2')
    plt.savefig(os.path.join(path, f'Pearson Heatmap V2.png'), dpi=300)
    plt.close()


def computeCorrelation(X, Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)

    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)

    sum_cross_prod = np.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr


def correlationPlot(DTR, title, cmap):
    corr = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = computeCorrelation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Features")


    path = f'Images/Dataset/Pearson/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'Pearson_Heatmap_{title}.png'), dpi=300)
    plt.savefig("./Images/Dataset/" + title + ".svg")
