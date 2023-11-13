import networkx as nx
import numpy as np

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from ..esdl.article import Article


def cluster_louvain(articles: List[Article], embed_field_name: str, similarity_threshold: float = 0.84):
    embeddings = []
    [embeddings.append(x.data[embed_field_name]) for x in articles]
    embeddings = np.array(embeddings)
    labels = [0] * len(embeddings)
    x = cosine_similarity(embeddings, embeddings)
    similarity_matrix = x > similarity_threshold
    G = nx.from_numpy_array(similarity_matrix)
    communities = nx.algorithms.community.louvain_communities(G, resolution=0.1)
    for community in communities:
        initial_member = min(community)
        for member in community:
            labels[member] = initial_member

    clusters = {}
    for a, lbl in zip(articles, labels):
        if lbl not in clusters:
            clusters[lbl] = [a]
        else:
            clusters[lbl].append(a)
    clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    consistent = {}
    for k in clusters.keys():
        articles: List[Article] = clusters[k]
        articles.sort(key=lambda article: (article.mediaReach, article.created), reverse=True)
        consistent[articles[0].uuid] = articles
    return consistent


def cluster_print(clusters: Dict[int, List[Article]]):
    for k in clusters.keys():
        articles: List[Article] = clusters[k]
        print(f"Cluster [{articles[0].title}]")
        for x, a in enumerate(articles):
            if x == len(articles) - 1:
                print(f'\t+---{a}')
                print('')
            else:
                print(f'\t|---{a}')


def compare_clusterings(a_first, a_second, cf_name='First Clustering', cs_name='Second Clustering'):
    diff = list(set(a_first) - set(a_second))
    if diff:
        print(f"{cf_name}[{len(a_first)}] {len(diff)} cluster articles are missing in {cs_name}[{len(a_second)}]")
        for x, a in enumerate(diff):
            if x == len(diff) - 1:
                print(f'\t+---{a}')
            else:
                print(f'\t|---{a}')
        return -1
    else:
        delta = len(a_first) - len(a_second)
        if delta == 0:
            print(f"Clusters are the same!")
        else:
            print(f"{cs_name} cluster [{a_first[0].title}] has all articles that are in {cf_name}!")
    return delta


def clusters_compare(c_first, c_second, fc_name, sc_name):

    for k in c_first.keys():
        if k not in c_second:
            print(f"{sc_name} is missing cluster [{k}]")
            continue

        a_first: List[Article] = c_first[k]
        a_second: List[Article] = c_second[k]

        if len(a_first) < 2 and len(a_second) < 2:
            continue

        print(f"Compare cluster [{a_first[0].title}] uuid [{k}]")
        ret = compare_clusterings(a_first, a_second, fc_name, sc_name)
        if ret == 0:
            print('---------------------------------------------------------------------------------------------------'
                  '---------------------------------------------------------------------------------------------------')
            print('')
            continue
        compare_clusterings(a_second, a_first, sc_name, fc_name)
        print('---------------------------------------------------------------------------------------------------'
              '---------------------------------------------------------------------------------------------------')
        print('')
