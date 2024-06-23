import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import AffinityPropagation, DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import multivariate_normal
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture


def plot_clusters(data, labels, title):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.tight_layout()

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, :-1].values  # A última coluna foi removida pois era a classe
    normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalização dos dados
    labels =  df.iloc[:, -1].values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels) #transforma os labels em labels numéricos (0,1 para binários ou 0,1,3... para mais clusters)
    return normalized_data, labels  # Retorna os dados e as classes reais

def input_int(message): # Garante que o K informado pelo usuário sempre será um número inteiro
    while True:
        try:
            value = int(input(message))
            return value
        except ValueError:
            print("Por favor, insira um número inteiro válido.")

def input_dataset():  # Retorna o nome do dataset escolhido
    datasets = {
        1: "DATA/iris.data",
        2: "DATA/Maternal Health Risk Data Set.csv",
        3: "DATA/Breast Cancer Wisconsin.data",
        4: "DATA/darwin.csv"
    }
    name = {
        1: "Iris",
        2: "Maternal_Health_Risk",
        3: "Breast_Cancer_Wisconsin",
        4: "Darwin"
    }
    
    while True:
        print("Escolha um dataset:")
        print("1 - Iris")
        print("2 - Maternal Health Risk")
        print("3 - Breast Cancer Wisconsin")
        print("4 - Darwin")
        
        try:
            choice = int(input("Digite o número correspondente ao dataset desejado (1-4): "))
            if choice in datasets:
                return datasets[choice], name[choice]
            else:
                print("Por favor, selecione uma opção válida.")
        except ValueError:
            print("Por favor, selecione uma opção correspondente a um dataset.")


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point2 - point1) ** 2))

def calculate_distance_matrix(data):
    n = len(data)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distances[i, j] = euclidean_distance(data[i], data[j])
            distances[j, i] = distances[i, j]
    
    return distances

def single_link(data, n_clusters):
    n = len(data)
    clusters = [[i] for i in range(n)]
    distances = calculate_distance_matrix(data)
    cluster_map = {i: i for i in range(n)}
    cluster_label = n
    
    while len(clusters) > n_clusters:
        min_distance = np.inf
        merge_indices = None
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        if distances[idx1, idx2] < min_distance:
                            min_distance = distances[idx1, idx2]
                            merge_indices = (i, j)
        
        cluster1, cluster2 = clusters[merge_indices[0]], clusters[merge_indices[1]]
        new_cluster = cluster1 + cluster2
        clusters.append(new_cluster)

        cluster1_id = cluster_map[cluster1[0]]
        cluster2_id = cluster_map[cluster2[0]]
        
        for idx in new_cluster:
            cluster_map[idx] = cluster_label
        cluster_label += 1
        
        del clusters[max(merge_indices)]
        del clusters[min(merge_indices)]
    
    final_labels = np.zeros(n, dtype=int)
    for cluster_index, cluster in enumerate(clusters):
        for idx in cluster:
            final_labels[idx] = cluster_index
    
    return final_labels


def calculate_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    silhouette_vals = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == label] for label in np.unique(labels) if label != labels[i]]

        if len(same_cluster) > 1:
            a_i = np.mean([euclidean_distance(X[i], point) for point in same_cluster if not np.array_equal(X[i], point)])
        else:
            a_i = 0

        if other_clusters:
            b_i = np.min([np.mean([euclidean_distance(X[i], point) for point in cluster]) for cluster in other_clusters if len(cluster) > 0])
        else:
            b_i = 0

        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
    
    return np.mean(silhouette_vals)


def calculate_adjusted_rand_index(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)



def gmm(data, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels

# # Função para inicializar os parâmetros do GMM
# def initialize_parameters(data, n_components):
#     n_samples, n_features = data.shape

#     # Inicialização dos pesos de forma uniforme

#     weights = np.full(shape=n_components, fill_value=1/n_components)
    
#     # Seleção aleatória de amostras como médias
#     means = data[np.random.choice(n_samples, n_components, replace=False)]
    
#     # Inicialização das covariâncias com a covariância do conjunto de dados
#     covariances = np.array([np.cov(data.T) for _ in range(n_components)])

#     return weights, means, covariances

# # Função para calcular a densidade de probabilidade da distribuição gaussiana multivariada


# def multivariate_gaussian(data, mean, cov):
#     return multivariate_normal.pdf(data, mean=mean, cov=cov)



# # Expectation Step (E-Step): Calcula as responsabilidades
# def e_step(data, weights, means, covariances):
#     n_samples = data.shape[0]
#     n_components = weights.shape[0]

#     responsibilities = np.zeros((n_samples, n_components))

#     for k in range(n_components):
#         responsibilities[:, k] = weights[k] * multivariate_gaussian(data, means[k], covariances[k])

#     responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
#     responsibilities_sum[responsibilities_sum == 0] = 1e-10  # Evita divisão por zero
#     responsibilities /= responsibilities_sum
#     return responsibilities


# # Maximization Step (M-Step): Atualiza os parâmetros
# def m_step(data, responsibilities, epsilon=1e-6):
#     n_samples, n_features = data.shape
#     n_components = responsibilities.shape[1]

#     weights = responsibilities.sum(axis=0) / n_samples
#     means = np.dot(responsibilities.T, data) / responsibilities.sum(axis=0)[:, np.newaxis]
#     covariances = np.zeros((n_components, n_features, n_features))

#     for k in range(n_components):
#         diff = data - means[k]
#         covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()
#         covariances[k] += np.eye(n_features) * epsilon  # Adiciona epsilon à diagonal

#     return weights, means, covariances


# # Função para calcular a log-verossimilhança

# def log_likelihood(data, weights, means, covariances):
#     n_samples = data.shape[0]
#     n_components = weights.shape[0]

#     log_likelihood = 0.0
#     for i in range(n_samples):
#         tmp = 0.0
#         for k in range(n_components):
#             tmp += weights[k] * multivariate_gaussian(data[i:i+1], means[k], covariances[k])
#         log_likelihood += np.log(max(tmp, 1e-10))  # Evita log de zero ou valor negativo
#     return log_likelihood


# # Algoritmo principal do GMM
# def gmm(data, n_components, n_iter=100, tol=1e-4):
#     weights, means, covariances = initialize_parameters(data, n_components)
#     log_likelihood_old = None

#     # Iterações E-Step e M-Step até a convergência
#     for i in range(n_iter):
#         responsibilities = e_step(data, weights, means, covariances)
#         weights, means, covariances = m_step(data, responsibilities)
#         log_likelihood_new = log_likelihood(data, weights, means, covariances)

#         # Verifica a convergência com base na mudança na log-verossimilhança
#         if log_likelihood_old is not None and abs(log_likelihood_new - log_likelihood_old) < tol:
#             break

#         log_likelihood_old = log_likelihood_new

#     # Obtém os rótulos atribuídos a cada ponto de dados
#     labels = responsibilities.argmax(axis=1)
#     return labels, weights, means, covariances

def initialize_centroids(data, K):
    centroids = data[np.random.choice(data.shape[0], K, replace=False)] #inicializando o centroide de forma aleatória
    return centroids

class KMeans:
    def __init__(self, n_clusters, max_iter=100): #O laço de repetição executado até a convergência do algoritmo terá limite máximo de 100 iterações
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X): #Execução do algoritmo
        self.centroids = initialize_centroids(X, self.n_clusters)
        
        
        for i in range(self.max_iter):
            #Verificar distância e colocar no centróide mais próximo
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2)) #i.	Medida de proximidade: distância Euclidiana
            labels = np.argmin(distances, axis=0) #Casos de empate na associação de um elemento ao centróide: escolher o primeiro.
            
            #Atualizar os Centroides
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            #Verificar se alcançou a convergência
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
                
        self.labels_ = labels
        return self.labels_

def fuzzy_c_means(data, n_clusters, m=2, max_iter=150, error=1e-5):
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    # Inicializa as pertinências aleatoriamente
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    
    # Inicializa os centróides
    centroids = np.zeros((n_clusters, n_features))
    
    for iteration in range(max_iter):
        U_previous = U.copy()
        
        # Atualiza os centróides
        for j in range(n_clusters):
            num = np.sum((U[:, j] ** m).reshape(-1, 1) * data, axis=0)
            den = np.sum(U[:, j] ** m)
            if den == 0:  # Evita divisão por zero
                centroids[j] = np.zeros(n_features)
            else:
                centroids[j] = num / den

        # Atualiza as pertinências
        for i in range(n_samples):
            for j in range(n_clusters):
                num = np.linalg.norm(data[i] - centroids[j])
                if num == 0:  # Evita divisão por zero
                    U[i, j] = 1
                else:
                    U[i, j] = 1 / (num ** (2 / (m - 1)))
            
            U[i] = U[i] / np.sum(U[i])  # Normaliza as pertinências

        # Verifica a convergência
        if np.linalg.norm(U - U_previous) < error:
            break

    # Atribui os clusters com base nas pertinências máximas
    labels = np.argmax(U, axis=1)
    
    return labels,U


def main():
    np.random.seed(42) #Seed para que seja possível replicar os resultados (42 a resposta para qualquer pergunta no universo rs.)

    file_path, dataset = input_dataset()
    print(f"Você escolheu o dataset '{dataset}'.")
    data, labels = preprocess_data(file_path)

    #parametros para execução
    k = input_int("Digite a quantidade de clusters para o K-means, Fuzzy C-Means e Gaussian Mixture Model: ")

    # K-means
    kmeans = KMeans(n_clusters=k)
    kmeans_labels = kmeans.fit(data)
    kmeans_silhouette = calculate_silhouette(data, kmeans_labels)

    # Gaussian Mixture Model
    gmm_labels = gmm(data, k)    
    gmm_silhouette = calculate_silhouette(data, gmm_labels)

    # Fuzzy C-Means
    fcm_labels, fcm_U = fuzzy_c_means(data, k)
    fcm_silhouette = calculate_silhouette(data, fcm_labels)

    # Resultados de Silhouette
    print(f'K-means Silhouette Score: {kmeans_silhouette}')
    print(f'Gaussian Mixture Model Silhouette Score: {gmm_silhouette}')
    print(f'Fuzzy C-Means Silhouette Score: {fcm_silhouette}')

    if kmeans_silhouette > gmm_silhouette and kmeans_silhouette > fcm_silhouette:
        best_algorithm = "K-means"
    elif gmm_silhouette > kmeans_silhouette and gmm_silhouette > fcm_silhouette:
        best_algorithm = "Gaussian Mixture Model"
    else:
        best_algorithm = "Fuzzy C-Means"

    print(f"Melhor algoritmo: {best_algorithm}")

    # Ajustado Rand Index
    kmeans_ari = calculate_adjusted_rand_index(kmeans_labels, labels)
    gmm_ari = calculate_adjusted_rand_index(gmm_labels, labels)
    fcm_ari = calculate_adjusted_rand_index(fcm_labels, labels)

    print(f'K-means Adjusted Rand Index: {kmeans_ari}')
    print(f'Gaussian Mixture Model Adjusted Rand Index: {gmm_ari}')
    print(f'Fuzzy C-Means Adjusted Rand Index: {fcm_ari}')

    max_adjusted_rand_index = max(
        kmeans_ari,
        gmm_ari,
        fcm_ari
    )
    if max_adjusted_rand_index == kmeans_ari:
        best_ari = "K-means"
    elif max_adjusted_rand_index == gmm_ari:
        best_ari = "GMM"
    else:
        best_ari = "Fuzzy C-Means"

    print(f"Melhor algoritmo de acordo com o Adjusted Rand Index: {best_ari}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = f"{dataset}_resultado_{current_time}.txt"
    with open(result_file, 'w') as f:
        f.write(f"K-means Silhouette Score: {kmeans_silhouette}\n")
        f.write(f"Gaussian Mixture Model Silhouette Score: {gmm_silhouette}\n")
        f.write(f"Fuzzy C-Means Silhouette Score: {fcm_silhouette}\n")
        f.write(f"Melhor algoritmo: {best_algorithm}\n\n")

        f.write(f'K-means Adjusted Rand Index: {kmeans_ari}\n')
        f.write(f'Gaussian Mixture Model Adjusted Rand Index: {gmm_ari}\n')
        f.write(f'Fuzzy C-Means Adjusted Rand Index: {fcm_ari}\n')
        f.write(f"Melhor algoritmo de acordo com o Adjusted Rand Index: {best_ari}\n\n")

        f.write("K-means Clusters:\n")
        kmeans_clusters = {i: np.where(kmeans_labels == i)[0].tolist() for i in np.unique(kmeans_labels)}
        for cluster in kmeans_clusters.values():
            f.write(f"{set(cluster)}\n")

        f.write("\nGaussian Mixture Model Clusters:\n")
        gmm_clusters = {i: np.where(gmm_labels == i)[0].tolist() for i in np.unique(gmm_labels)}
        for cluster in gmm_clusters.values():
            f.write(f"{set(cluster)}\n")

        f.write("\nFuzzy C-Means Clusters:\n")
        fcm_clusters = {i: np.where(fcm_labels == i)[0].tolist() for i in np.unique(fcm_labels)}           
        for cluster in fcm_clusters.values():
            f.write(f"{set(cluster)}\n")          

        f.write("\nFuzzy C-Means  Membership Matrix:\n")
   
        for i, row in enumerate(fcm_U, start=1):
            f.write(f"[{i} - {row}]\n")

    # Salva os Agrupamentos em um PDF    
    result_file_pdf = f"{dataset}_resultado_clusters_{current_time}.pdf"
    with PdfPages(result_file_pdf) as pdf:
        plot_clusters(data, labels, 'Original Labels')
        pdf.savefig()
        plt.close()

        plot_clusters(data, kmeans_labels, 'K-means Clustering')
        pdf.savefig()
        plt.close()

        plot_clusters(data, gmm_labels, 'Gaussian Mixture Model Clustering')
        pdf.savefig()
        plt.close()

        plot_clusters(data, fcm_labels, 'Fuzzy C-Means Clustering')
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    main()
