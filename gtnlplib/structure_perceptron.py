from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """compute the structure perceptron update for a single instance

    :param tokens: tokens to tag 
    :param tags: gold tags
    :param weights: weights
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :param tagger: function from (tokens,feat_func,weights,all_tags) --> tag sequence
    :param all_tags: list of all candidate tags
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """

    update = defaultdict(float)
    predicted_tags, score = tagger(tokens, feat_func, weights, all_tags)
    for pos, predicted_tag in enumerate(predicted_tags):
        predicted_features = feat_func(tokens, predicted_tag, predicted_tags[pos - 1] if pos > 0 else constants.START_TAG, pos)
        actual_features = feat_func(tokens, tags[pos], tags[pos - 1] if pos > 0 else constants.START_TAG, pos)

        for key in actual_features:
            update[key] += actual_features[key]

        for key in predicted_features:
            update[key] -= predicted_features[key]

    return update
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron

    :param labeled instances: list of (token-list, tag-list) tuples, each representing a tagged sentence
    :param feat_func: function from list of words and index to dict of features
    :param tagger: function from list of words, features, weights, and candidate tags to list of tags
    :param N_its: number of training iterations
    :param all_tags: optional list of candidate tags. If not provided, it is computed from the dataset.
    :returns: weight dictionary
    :returns: list of weight dictionaries at each iteration
    :rtype: defaultdict, list

    """
    """
    You can almost copy-paste your perceptron.estimate_avg_perceptron function here. 
    The key differences are:
    (1) the input is now a list of (token-list, tag-list) tuples
    (2) call sp_update to compute the update after each instance.
    """

    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)

    weights = defaultdict(float, { ('NOUN',constants.OFFSET) : 1e-3})
    avg_weights = defaultdict(float)
    w_sum = defaultdict(float)
    weight_history = []

    t = 0.0
    for it in xrange(N_its):
        for (tokens, tags) in labeled_instances:
            update = sp_update(tokens, tags, weights, feat_func, tagger, all_tags)
            for key in update:
                w_sum[key] += t * update[key]
                weights[key] += update[key]
            t += 1
        for key in weights:
            avg_weights[key] = weights[key] - 1 / t * w_sum[key]
        weight_history.append(avg_weights.copy())
        print it,
    if ('--END--', '**OFFSET**') in avg_weights:
        avg_weights.pop(('--END--', '**OFFSET**'))
    return avg_weights, weight_history