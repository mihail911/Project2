import numpy as np
import collections
import itertools
import matplotlib.pyplot as plt
import time

from data_utils import makeHeatMap
from scipy.sparse import csr_matrix



# Part 1
def load_data():

    # Data50.csv
    articleIDs = []
    wordIDs = []
    wordCounts = []

    with open("p2_data/data50.csv") as f:
        for line in f:
            articleID, wordID, wordCount = line.split(",")
            articleIDs.append(int(articleID))
            wordIDs.append(int(wordID))
            wordCounts.append(int(wordCount))

            # if int(articleID) > maxArticleID:
            #     maxArticleID = int(articleID)
            # elif int(wordID) > maxWordID:
            #     maxWordID = int(maxWordID)

    maxWordID = max(wordIDs)
    maxArticleID = max(articleIDs)
    # Note 0-indexed so article 1 at index 0
    article_word_mat = csr_matrix((maxArticleID, maxWordID), dtype=np.int32)

    # Populate matrix
    for art, word, count in zip(articleIDs, wordIDs, wordCounts):
        article_word_mat[art-1, word-1] += count


    # Label.csv
    articleID_groupID = {}
    group_to_articles = collections.defaultdict(list)
    articleID = 1
    with open("p2_data/label.csv") as f:
        for line in f:
            group = int(line)
            group_to_articles[group].append(articleID)

            articleID_groupID[articleID] = int(line)
            articleID += 1

    # Group.csv
    id_group_name = {}
    groupID = 1
    with open("p2_data/groups.csv") as f:
        for line in f:
            id_group_name[groupID] = line
            groupID += 1

    return id_group_name, group_to_articles, articleID_groupID, article_word_mat


def jaccard_sim(x, y):
    min_elems = np.minimum(x, y)
    max_elems = np.maximum(x, y)

    return np.sum(min_elems)/float(np.sum(max_elems))


def cosine_sim(x, y):
    return np.dot(x, y) / float((np.linalg.norm(x) * np.linalg.norm(y)))


def l2_sim(x, y):
    return -1.0 * np.linalg.norm(x-y)


# Compute 20x20 group similarity metric, given 'sim' metric
def group_sim(group_to_articles, article_word_mat, sim):
    group_sim_mat = np.zeros((20, 20))
    groups = group_to_articles.keys()
    # Iterate over all pairs of groups
    start_time = time.time()

    for group1, group2 in itertools.product(groups, groups):
        articles1 = group_to_articles[group1]
        articles2 = group_to_articles[group2]
        # All pairs of articles from the two groups
        article_cross = list(itertools.product(articles1, articles2))
        avg_sim = 0.0
        for art1, art2 in article_cross:
            art1_vec = article_word_mat[art1-1,:].toarray().flatten()
            art2_vec = article_word_mat[art2-1,:].toarray().flatten()
            avg_sim += sim(art1_vec, art2_vec)

        avg_sim /= len(article_cross)

        print "Time elapsed: ", str(time.time() - start_time)
        print "Group {0}, Group {1} complete".format(str(group1), str(group2))
        group_sim_mat[group1-1, group2-1] = avg_sim

    return group_sim_mat


def alt_group_sim(group_to_articles, article_word_mat, sim):
    # Compute article with max Jaccard similarity -- note we have to update both arrays
    # Article that has max J sim for given article
    article_max = np.zeros(61037)
    # Actual value of max similarity for each article
    sim_max = np.zeros(61037)

    all_articles = set(range(1, 1001))
    # Compute similarities once
    groups = group_to_articles.keys()
    for group in groups:
        articles = set(group_to_articles[group])
        # Iterate over all articles in current group
        for art in articles:
            art_vec = article_word_mat[art-1, :].toarray().flatten()
            curr_art_max = article_max[art-1]
            curr_sim_max = sim_max[art-1]
            for other_art in all_articles:
                # Only consider if other article not in current group
                if other_art not in articles:
                    other_art_vec = article_word_mat[other_art-1, :].toarray().flatten()

                    curr_sim = sim(art_vec, other_art_vec)
                    if curr_sim > curr_sim_max:
                        curr_art_max = other_art
                        curr_sim_max = curr_sim

            article_max[art-1] = curr_art_max
            sim_max[art-1] = curr_sim_max

    print "Finished computing similarity arrays..."

    start_time = time.time()
    # Now populate similarity matrix
    group_sim_mat = np.zeros((20, 20))
    for group1, group2 in itertools.product(groups, groups):
        articles1 = set(group_to_articles[group1])
        articles2 = set(group_to_articles[group2])

        num_sim = 0.0
        for art1 in articles1:
            most_similar = article_max[art1-1]
            if most_similar in articles2:
                num_sim += 1

        print "Time elapsed: ", str(time.time() - start_time)
        print "Group {0}, Group {1} complete".format(str(group1), str(group2))
        group_sim_mat[group1-1, group2-1] = num_sim

    return group_sim_mat


def jaccard_heatmap(group_to_articles, article_word_mat, names):
    group_sim_mat = group_sim(group_to_articles, article_word_mat, jaccard_sim)
    makeHeatMap(group_sim_mat, names, plt.cm.Blues, "jaccard_heatmap")


def jaccard_heatmap_alt(group_to_articles, article_word_mat, names):
    group_sim_mat = alt_group_sim(group_to_articles, article_word_mat, jaccard_sim)
    makeHeatMap(group_sim_mat, names, plt.cm.Blues, "jaccard_heatmap_alt")


def cosine_heatmap(group_to_articles, article_word_mat, names):
    group_sim_mat = group_sim(group_to_articles, article_word_mat, cosine_sim)
    makeHeatMap(group_sim_mat, names, plt.cm.Blues, "cosine_heatmap")


def l2_heatmap(group_to_articles, article_word_mat, names):
    group_sim_mat = group_sim(group_to_articles, article_word_mat, l2_sim)
    makeHeatMap(group_sim_mat, names, plt.cm.Blues, "l2_heatmap")


if __name__ == "__main__":
    id_group_name, group_to_articles, articleID_groupID, article_word_mat = load_data()

    group_names = id_group_name.values()

    #jaccard_heatmap(group_to_articles, article_word_mat, group_names)
    #cosine_heatmap(group_to_articles, article_word_mat, group_names)
    #l2_heatmap(group_to_articles, article_word_mat, group_names)
    jaccard_heatmap_alt(group_to_articles, article_word_mat, group_names)
