from gtnlplib import constants

def word_feats(words,y,y_prev,m):
    """This function should return at most two features:
    - (y,constants.CURR_WORD_FEAT,words[m])
    - (y,constants.OFFSET)

    :param words: list of word tokens
    :param y: current feature
    :param y_prev: previous feature (ignored)
    :param m: index of current word
    :returns: dict of features, containing a single feature and a count of 1
    :rtype: dict

    """
    feature_vector = dict()
    feature_vector[(y, constants.OFFSET)] = 1
    if m < len(words):
        feature_vector[(y, constants.CURR_WORD_FEAT, words[m])] = 1
    return feature_vector

def word_suff_feats(words,y,y_prev,m):
    """This function should return all the features returned by word_feats,
    plus an additional feature for each token, indicating the final two characters.

    You may call word_feats in this function.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    feature_vector = word_feats(words, y, y_prev, m)
    feature_vector[(y, constants.SUFFIX_FEAT, words[m][-2:])] = 1
    return feature_vector
    
def word_neighbor_feats(words,y,y_prev,m):
    """compute features for the current word being tagged, its predecessor, and its successor.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    feature_vector = word_feats(words, y, y_prev, m)
    if m == len(words):
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
    elif len(words) == 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    elif m == 0:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    elif m == len(words) - 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    else:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    return feature_vector
    
def word_feats_competitive_en(words,y,y_prev,m):
    feature_vector = word_feats(words, y, y_prev, m)
    for i in range(len(words[m])):
        feature_vector[(y, constants.SUFFIX_FEAT, words[m][-1 * i:])] = 1
    if m == len(words):
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
    elif len(words) == 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    elif m == 0:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    elif m == len(words) - 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    else:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    return feature_vector
    
def word_feats_competitive_ja(words,y,y_prev,m):
    feature_vector = word_feats(words, y, y_prev, m)
    for i in range(len(words[m])):
        feature_vector[(y, constants.SUFFIX_FEAT, words[m][-1 * i:])] = 1
    if m == len(words):
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
    elif len(words) == 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    elif m == 0:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    elif m == len(words) - 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    else:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    return feature_vector

def hmm_feats(words,y,y_prev,m):
    feature_vector = dict()
    feature_vector = { (y, constants.PREV_TAG_FEAT, y_prev) : 1 }
    if m < len(words):
        feature_vector[(y, constants.CURR_WORD_FEAT, words[m])] = 1
    return feature_vector

def hmm_feats_competitive_en(words,y,y_prev,m):
    feature_vector = hmm_feats(words, y, y_prev, m)
    for i in range(len(words[m])):
        feature_vector[(y, constants.SUFFIX_FEAT, words[m][-1 * i:])] = 1
    if m == len(words):
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
    elif len(words) == 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    elif m == 0:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    elif m == len(words) - 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    else:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    return feature_vector

def hmm_feats_competitive_ja(words,y,y_prev,m):
    feature_vector = hmm_feats(words, y, y_prev, m)
    for i in range(len(words[m])):
        feature_vector[(y, constants.SUFFIX_FEAT, words[m][-1 * i:])] = 1
    if m == len(words):
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
    elif len(words) == 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    elif m == 0:
        feature_vector[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    elif m == len(words) - 1:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
    else:
        feature_vector[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        feature_vector[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    return feature_vector