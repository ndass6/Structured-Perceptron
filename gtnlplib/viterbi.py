import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG
import numpy as np

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

def viterbi_step(tag, m, words, feat_func, weights, prev_scores):
    """
    Calculate the best path score and back pointer for a given node in the trellis

    :param tag: The tag for which we want to calculate the best path
    :param m: index of the token for which we want to calculate the best tag
    :param words: the list of tokens to tag
    :param feat_func: A function of (words, curr_tag, prev_tag, curr_index) that produces features 
    :param weights: A defaultdict that maps features to numeric score. Should not key error for indexing into keys that do not exist.
    :param prev_scores: a dict, in which keys are tags for token m-1 and values are viterbi scores
    :returns: tuple of (best_score, best_feature), where
        best_score   -- The highest score of any sequence of tags
        best_feature -- The feature in the previous layer of the trellis corresponding
            to the best score

    :rtype: tuple

    """
    
    scores = {}

    for prev_tag in prev_scores:
        feats = feat_func(words, tag, prev_tag, m)
        score = 0
        for feat in feats:
            score += feats[feat] * weights[feat]
        scores[prev_tag] = prev_scores[prev_tag] + score

    best_score = max(scores.values())
    best_tag = argmax(scores)
    
    return best_score, best_tag

def build_trellis(tokens,feat_func,weights,all_tags):
    """Construct a trellis for the hidden Markov model. Output is a list of dicts.

    :param tokens: list of word tokens to be tagged
    :param feat_func: feature function (words, tag, prev_tag, index)
    :param weights: defaultdict of weights
    :param all_tags: list/set of all possible tags
    :returns: list of dicts, length = len(words)
    first dict should represent score from start to token 1, 
    then score from token 1 to token 2,
    etc until token M
    :rtype: list of dicts

    """
    
    trellis = [None]*(len(tokens))

    # build the first column separately
    trellis[0] = {}
    for tag in all_tags:
            trellis[0][tag] = viterbi_step(tag, 0, tokens, feat_func, weights, {START_TAG : 0})
    
    # iterate over the remaining columns
    for m in range(1,len(tokens)):
        trellis[m] = {}
        for tag in all_tags:
            trellis[m][tag] = viterbi_step(tag, m, tokens, feat_func, weights, { key : trellis[m - 1][key][0] for key in trellis[m - 1] })
        
    return trellis


def viterbi_tagger(tokens,feat_func,weights,all_tags):
    """Tag the given words using the viterbi algorithm
        Parameters:
        tokens     -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights   -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags  -- A set of all possible tags

        Returns:
        tags       -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
    """
    
    trellis = build_trellis(tokens,feat_func,weights,all_tags)

    best_score, last_tag = viterbi_step(END_TAG, len(tokens), tokens, feat_func, weights, { key : trellis[-1][key][0] for key in trellis[-1] })

    output = [last_tag]
    for v_m in reversed(trellis[1:]):
        output.insert(0, v_m[output[0]][1])

    return output, best_score