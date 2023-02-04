import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi
from datetime import date
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler


def elbow_method(standardised_dataframe, cluster_k_to_try, todays_date):
    """
    The elbow method is a graphical representation of finding the optimal
    'K' in a K-means cluster.
    This function computes the within cluster sum of squares (WCSS) to choose
    an appropriate number of clusters.
    """
    print('Running the elbow method function')

    inertias = []  # WCSS

    # Try 10 clusters
    for k in range(1, cluster_k_to_try):
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        model.fit(standardised_dataframe)
        inertias.append(model.inertia_)

    # "Elbow Criterion":
    plt.plot(range(1, cluster_k_to_try), inertias, marker='o', linestyle='--')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Within Cluster Sum of Squares')
    plt.title('Optimal Number of Clusters')
    plt.savefig(f'./plots/{todays_date}/elbow_method_{cluster_k_to_try}_clusters.png')
    plt.close()


def silhouette_coefficient(dataframe, cluster_k_to_try, todays_date):
    """
    The silhouette coefficient is a metric used to calculate the goodness of
    a clustering technique. Ranges in value between -1 and 1
    A score of 1 means the clsuters are well apart from each other and clearly distinguished
    A score of 0 means clusters are indifferent
    A score of -1 means the clusters are assigned in the wrong way.
    Ref. Silhouette Coefficient, Validating clustering techniques. Medium article.
    """
    print('Running the silhouette coefficient function')

    silhouette_coefficients = []

    for k in range(2, cluster_k_to_try):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(dataframe)
        # silhouette_score() needs a minimum of 2 clusters
        score = silhouette_score(dataframe, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, cluster_k_to_try), silhouette_coefficients)
    plt.xticks(range(2, cluster_k_to_try))
    plt.xlabel("No. Clusters")
    plt.ylabel("Silhouette Coeff.")
    plt.savefig(f'./plots/{todays_date}/silhouette_coefficients_{cluster_k_to_try}_clusters.png')
    plt.close()


def clustering(k, dataframe, verbose, init_method='k-means++'):
    """
    This function clusters the data into k clusters. It adds the labels to the dataframe
    which is then returned.
    """
    print(f'Clustering the data into {k} clusters using a initialisation method of {init_method}')
    kmeans_pca = KMeans(n_clusters=k, init=init_method, random_state=42)
    kmeans_pca.fit(dataframe)
    labels = kmeans_pca.predict(dataframe)
    dataframe['Cluster'] = kmeans_pca.labels_

    if verbose:
        print(dataframe.head())

    # maybe this can be a groupby?
    c0 = dataframe.loc[dataframe.Cluster == 0, 'Cluster'].count()
    c1 = dataframe.loc[dataframe.Cluster == 1, 'Cluster'].count()
    c2 = dataframe.loc[dataframe.Cluster == 2, 'Cluster'].count()
    c3 = dataframe.loc[dataframe.Cluster == 3, 'Cluster'].count()

    if verbose:
        print(c0, c1, c2, c3)

        print("The % of participants in cluster 0: " + "{:.2f}".format(c0 / 1094 * 100) + "%")
        print("The % of participants in cluster 1: " + "{:.2f}".format(c1 / 1094 * 100) + "%")
        print("The % of participants in cluster 2: " + "{:.2f}".format(c2 / 1094 * 100) + "%")
        print("The % of participants in cluster 3: " + "{:.2f}".format(c3 / 1094 * 100) + "%")

    return dataframe


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct


def component_scores(dataframe, N, todays_date):
    """
    This function looks at each cluster individually. It then looks at the different reading components
    and looks at how 'good' and 'bad' each cluster is at each component.
    NOTE: This is a very crude measure of 'good' and 'bad'. We are simply using a threshold of 40. This may indicate
    what components the clusters are stronger at and weaker but is not a true representation of the cluster.
    :param dataframe: Dataframe containing the clusters reading scores
    :param N: the cluter of interest (integer)
    :param todays_date: date (used for saving the plots)
    :return: None
    """

    print(f'Running the component scores function for cluster {N}')

    Cluster_N = dataframe[dataframe.Cluster == N]
    poorWMI = 0
    goodWMI = 0
    poorWM = 0
    goodWM = 0
    poorPSI = 0
    goodPSI = 0
    poorPatt = 0
    goodPatt = 0
    poorAtt = 0
    goodAtt = 0
    poorEl = 0
    goodEl = 0
    poorBw = 0
    goodBw = 0
    poorVCI = 0
    goodVCI = 0

    for n in Cluster_N['WISC - WMI']:
        if n < 40:
            poorWMI = poorWMI + 1
        elif n >= 40:
            goodWMI = goodWMI + 1

    for n in Cluster_N['NIH - WM']:
        if n < 40:
            poorWM = poorWM + 1
        elif n >= 40:
            goodWM = goodWM + 1

    for n in Cluster_N['WISC - PSI']:
        if n < 40:
            poorPSI = poorPSI + 1
        elif n >= 40:
            goodPSI = goodPSI + 1

    for n in Cluster_N['Pattern Comp.']:
        if n < 40:
            poorPatt = poorPatt + 1
        elif n >= 40:
            goodPatt = goodPatt + 1

    for n in Cluster_N['Visuo-Spat Attention']:
        if n < 40:
            poorAtt = poorAtt + 1
        elif n >= 40:
            goodAtt = goodAtt + 1

    for n in Cluster_N['CTOPP - Elision']:
        if n < 40:
            poorEl = poorEl + 1
        elif n >= 40:
            goodEl = goodEl + 1

    for n in Cluster_N['CTOPP - Blended W']:
        if n < 40:
            poorBw = poorBw + 1
        elif n >= 40:
            goodBw = goodBw + 1

    for n in Cluster_N['WISC - VCI']:
        if n < 40:
            poorVCI = poorVCI + 1
        elif n >= 40:
            goodVCI = goodVCI + 1

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    WMIlabels = 'Poor WMI', 'Good WMI'
    WMlables = 'Poor WM', 'Good WM'
    PSIlabels = 'Poor PSI', 'Good PSI'
    Patternlabels = 'Poor Pattern', 'Good Pattern'
    Attlabels = 'Poor Attent.', 'Good Attent.'
    Ellables = 'Poor Elision', 'Good Elision'
    Bwlabels = 'Poor Blending W.', 'Good Blending W.'
    VCIlabels = 'Poor VCI', 'Good VCI'

    WMIsizes = [poorWMI, goodWMI]
    WMsizes = [poorWM, goodWM]
    PSIsize = [poorPSI, goodPSI]
    Pattsize = [poorPatt, goodPatt]
    Attensize = [poorAtt, goodAtt]
    Elsize = [poorEl, goodEl]
    Bwsize = [poorBw, goodBw]
    VCIsizes = [poorVCI, goodVCI]

    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.suptitle("Cluster " + str(N), fontsize=30)
    axs[0, 0].pie(WMIsizes, labels=WMIlabels, autopct=make_autopct(WMIsizes))
    axs[0, 0].set_title('WISC-WMI')
    axs[0, 1].pie(WMsizes, labels=WMlables, autopct=make_autopct(WMsizes))
    axs[0, 1].set_title('NIH-WM')
    axs[0, 2].pie(Attensize, labels=Attlabels, autopct=make_autopct(Attensize))
    axs[0, 2].set_title('Visuo-Spat Attention')
    axs[1, 0].pie(PSIsize, labels=PSIlabels, autopct=make_autopct(PSIsize))
    axs[1, 0].set_title('WISC-PSI')
    axs[1, 1].pie(Pattsize, labels=Patternlabels, autopct=make_autopct(Pattsize))
    axs[1, 1].set_title('NIH-Patterns')
    axs[1, 2].pie(Elsize, labels=Ellables, autopct=make_autopct(Elsize))
    axs[1, 2].set_title('Elision')
    axs[2, 0].pie(Bwsize, labels=Bwlabels, autopct=make_autopct(Bwsize))
    axs[2, 0].set_title('Blending Words')
    axs[2, 1].pie(VCIsizes, labels=VCIlabels, autopct=make_autopct(VCIsizes))
    axs[2, 1].set_title('WISC-VCI')
    axs[-1, -1].axis('off')
    axs[-1, -2].axis('off')

    plt.savefig(f'./plots/{todays_date}/component_scores_{N}.png')
    plt.close()

    return None


def cluster_by_PCA(dataframe, k, todays_date):
    """
    This function computes 2 Principle Components in order to visualise the clusters. The labels have already
    been assigned and PCs are only used to plot the scatter plot
    :param todays_date:
    :param dataframe:
    :param k:
    :return:
    """
    print('Running the clustering by PCA function')
    full_palette = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    k_palette = full_palette[:k]
    pca = PCA(n_components=2)
    dataframe.columns = dataframe.columns.astype(str)
    PCA_components = pca.fit_transform(dataframe)

    df_2_components = pd.concat([dataframe.reset_index(drop=True), pd.DataFrame(PCA_components)], axis=1)
    df_2_components.columns.values[-2:] = ['Comp. 0', 'Comp. 1']

    sns.scatterplot(df_2_components, x = 'Comp. 0', y= 'Comp. 1', hue=dataframe['Cluster'], palette=k_palette)
    plt.title('Clusters by PCA Components')

    plt.savefig(f'./plots/{todays_date}/clusters_PCA.png')
    plt.close()

    return None


def radar_profiles(dataframe, N, todays_date):
    """
    This function creates radar profiles for each of the clusters so that we can get a full picture of each of the
    clusters ability across the different components.
    :param dataframe: dataframe containing the clusters reading scores
    :param n: cluster number (int)
    :param todays_date: date (used for saving the plots)
    :return: None
    """
    print(f'Running the radar profiles function for cluster {N}')

    cluster_df = dataframe[dataframe.Cluster == N]

    avg_WMI = cluster_df[['WISC - WMI']].mean()
    avg_WM = cluster_df[['NIH - WM']].mean()
    avg_PSI = cluster_df[['WISC - PSI']].mean()
    avg_Pat = cluster_df[['Pattern Comp.']].mean()
    avg_Att = cluster_df[['Visuo-Spat Attention']].mean()
    avg_El = cluster_df[['CTOPP - Elision']].mean()
    avg_BW = cluster_df[['CTOPP - Blended W']].mean()
    avg_VCI = cluster_df[['WISC - VCI']].mean()

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(polar="Ture")

    categories = ['WISC - WMI', 'NIH - WM', 'WISC - PSI', 'Pattern Comp.', 'Visuo-Spat Attention',
                  'CTOPP - Elision', 'CTOPP - Blended W', 'WISC - VCI']

    cat_len = len(categories)

    values = [avg_WMI, avg_WM, avg_PSI, avg_Pat, avg_Att, avg_El, avg_BW, avg_VCI]
    values += values[:1]

    angles = [N / float(cat_len) * 2 * pi for N in range(cat_len)]
    angles += angles[:1]

    plt.polar(angles, values, marker='.')
    plt.fill(angles, values, alpha=0.3)
    plt.xticks(angles[:-1], categories)

    plt.yticks([10, 20, 30, 40, 50, 60, 70], color="grey", size=10)
    plt.title("Cluster: " + str(N))
    plt.savefig(f'./plots/{todays_date}/radar_profile_cluster_{N}.png')
    plt.close()

    return None


def four_reading_components(DF, N, todays_date):
    """
    This function looks at each cluster individually and the 4 reading components by combining several of the
    reading scores together. It then looks at how many 'goo', 'bad', and 'average' people there are in each cluster.
    NOTE: This is a very crude measure of 'good' and 'bad'. We are simply using a threshold of 40. This may indicate
    what components the clusters are stronger at and weaker but is not a true representation of the cluster.
    :param DF:
    :param N:
    :param todays_date:
    :return:
    """

    print(f'Running the four reading components function for cluster {N}')

    cluster_k = DF[DF.Cluster == N]
    cluster_k = cluster_k.reset_index()
    length = cluster_k.shape[0]

    GoodWM = 0
    AvgWM = 0
    PoorWM = 0

    GoodVP = 0
    AvgVP = 0
    PoorVP = 0

    poorAtt = 0
    goodAtt = 0

    GoodPP = 0
    AvgPP = 0
    PoorPP = 0

    poorVCI = 0
    goodVCI = 0

    for i in range(0, length - 1):

        if cluster_k['WISC - WMI'][i] <= 40 and cluster_k['NIH - WM'][i] <= 40:
            PoorWM += 1
        elif cluster_k['WISC - WMI'][i] > 40 and cluster_k['NIH - WM'][i] > 40:
            GoodWM += 1
        else:
            AvgWM += 1

        if cluster_k['WISC - PSI'][i] <= 40 and cluster_k['Pattern Comp.'][i] <= 40:
            PoorVP += 1
        elif cluster_k['WISC - WMI'][i] > 40 and cluster_k['Pattern Comp.'][i] > 40:
            GoodVP += 1
        else:
            AvgVP += 1

        if cluster_k['CTOPP - Elision'][i] <= 40 and cluster_k['CTOPP - Blended W'][i] <= 40:
            PoorPP += 1
        elif cluster_k['CTOPP - Elision'][i] > 40 and cluster_k['CTOPP - Blended W'][i] > 40:
            GoodPP += 1
        else:
            AvgPP += 1

    for n in cluster_k['Visuo-Spat Attention']:
        if n < 40:
            poorAtt = poorAtt + 1
        elif n >= 40:
            goodAtt = goodAtt + 1

    for n in cluster_k['WISC - VCI']:
        if n < 40:
            poorVCI = poorVCI + 1
        elif n >= 40:
            goodVCI = goodVCI + 1

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    WMlabels = 'Poor WM', 'Good WM', 'Avg WM'
    VPlabels = 'Poor VP', 'Good VP', 'Avg VP'
    Attlabels = 'Poor Attent.', 'Good Attent.'
    PPlabels = 'Poor PP', 'Good PP', 'Avg PP'
    VCIlabels = 'Poor VCI', 'Good VCI'

    WMsize = [PoorWM, GoodWM, AvgWM]
    VPsize = [PoorVP, GoodVP, AvgVP]
    Attensize = [poorAtt, goodAtt]
    PPsize = [PoorPP, GoodPP, AvgPP]
    VCIsizes = [poorVCI, goodVCI]

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.suptitle("Cluster " + str(N), fontsize=30)
    axs[0, 0].pie(WMsize, labels=WMlabels, autopct=make_autopct(WMsize))
    axs[0, 0].set_title('Working Memory')
    axs[0, 1].pie(Attensize, labels=Attlabels, autopct=make_autopct(Attensize))
    axs[0, 1].set_title('Attention')
    axs[0, 2].pie(VPsize, labels=VPlabels, autopct=make_autopct(VPsize))
    axs[0, 2].set_title('Visual Processing')
    axs[1, 0].pie(PPsize, labels=PPlabels, autopct=make_autopct(PPsize))
    axs[1, 0].set_title('Phonological Processing')
    axs[1, 1].pie(VCIsizes, labels=VCIlabels, autopct=make_autopct(VCIsizes))
    axs[1, 1].set_title('WISC-VCI')
    axs[-1, -1].axis('off')

    plt.savefig(f'./plots/{todays_date}/four_reading_components_{N}.png')
    plt.close()
    plt.show()

    return None


def main(data_loc, k, run_elbow_method, run_silhouette_coeff, run_component_score, visualise_cluster_PCA,
         create_radar_profiles, run_4_component_analysis, verbose):

    todays_date = date.today()
    if not os.path.exists(f'./plots/{todays_date}'):
        os.makedirs(f'./plots/{todays_date}')
    print(f'Saving the plots to ./plots/{todays_date}')

    data = pd.read_csv(data_loc).drop('ID', axis=1)

    if verbose:
        print(data.head(5))
        print(f'The columns in the data are {data.columns}')
        print(f'The data has {data.shape[0]} rows')

    print('Standardising the data')
    standardised_data = StandardScaler().fit_transform(data)
    standardised_dataframe = pd.DataFrame(standardised_data)
    print(standardised_dataframe.head(5))

    if run_elbow_method:
        clusters_to_try = [3, 4, 5]
        for c in clusters_to_try:
            elbow_method(data, c, todays_date)

    if run_silhouette_coeff:
        clusters_to_try = [3, 4, 5]
        for c in clusters_to_try:
            silhouette_coefficient(data, c, todays_date)

    if k != None:
        clustering_df = clustering(k, data, verbose, init_method='k-means++')

        if run_component_score:
            for i in range(0, k):
                component_scores(clustering_df, i, todays_date)

        if visualise_cluster_PCA:
            cluster_by_PCA(clustering_df, k, todays_date)

        if create_radar_profiles:
            for i in range(0, k):
                radar_profiles(clustering_df, i, todays_date)

        if run_4_component_analysis:
            for i in range(0, k):
                four_reading_components(clustering_df, i, todays_date)

if __name__ == '__main__':
    data_loc = 'IQTable5.csv'
    k = 5  # if you run want to run clustering select number (int) if not, replace with None

    run_elbow_method = True
    run_silhouette_coeff = True
    run_component_score = True
    visualise_cluster_PCA = True
    create_radar_profiles = True
    run_4_component_analysis = True
    verbose = True

    main(data_loc=data_loc,
         k=k,
         run_elbow_method=run_elbow_method,
         run_silhouette_coeff=run_silhouette_coeff,
         run_component_score=run_component_score,
         visualise_cluster_PCA=visualise_cluster_PCA,
         create_radar_profiles=create_radar_profiles,
         run_4_component_analysis=run_4_component_analysis,
         verbose=verbose
         )
