from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE, START_TAG, END_TAG

import operator
from collections import defaultdict
import tempfile
import matplotlib.pyplot as plt

argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

def classifier_tagger(tokens,feat_func,weights,all_tags):
    """function that tags a sequence of tokens using a classifier, with the given feature function and list of weights

    :param tokens: list of tokens to tag
    :param feat_func: feature function
    :param weights: defaultdict of weights
    :param all_tags: list of all possible tags
    :returns: list of predicted tags, score of best tag sequence
    :rtype: list, float

    """

    tags = []
    score = 0.0
    for pos, token in enumerate(tokens):
        scores = {}
        for tag in all_tags:
            scores[tag] = 0.0
            features = feat_func(tokens, tag, "IGNORE", pos)
            for feature in features:
                scores[tag] += features[feature] * weights[feature]
        tags.append(argmax(scores))
        score += scores[argmax(scores)]
    return tags, score

def compute_features(tokens,tags,feat_func):
    """compute dict of features and counts for a token and tag sequence

    :param tokens: list of tokens
    :param tags: list of tags
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :returns: dict of features and counts over entire sequence
    :rtype: defaultdict

    """
    features = defaultdict(float)
    for pos in range(len(tags) + 1):
        if pos == 0:
            word_features = feat_func(tokens, tags[pos], START_TAG, pos)
        elif pos == len(tokens):
            word_features = feat_func(tokens, END_TAG, tags[pos - 1], pos)
        else:
            word_features = feat_func(tokens, tags[pos], tags[pos - 1], pos)
        for word_feature in word_features:
            features[word_feature] += 1
    return features

def eval_tagging_model(testfile,tagger_func,features,weights,all_tags,output_file=None):
    tagger = lambda words, all_tags : tagger_func(words,
                                                  features,
                                                  weights,
                                                  all_tags)[0]
    confusion = eval_tagger(tagger,
                            output_file,
                            testfile=testfile,
                            all_tags=all_tags)
    return scorer.accuracy(confusion)

def apply_tagging_model(testfile,tagger_func,features,weights,all_tags,output_file):
    tagger = lambda words, all_tags : tagger_func(words,
                                                  features,
                                                  weights,
                                                  all_tags)[0]
    apply_tagger(tagger,
                 outfilename=output_file,
                 testfile=testfile,
                 all_tags=all_tags)


def plot_learning_curve(testfile, tagger_func, features, weight_hist, all_tags, lineformat='b-'):
    accs = []
    for weights in weight_hist:
        accs.append(eval_tagging_model(testfile,tagger_func,features,weights,all_tags))
    lines = plt.plot(accs,lineformat)
    return lines
    
def apply_tagger(tagger,outfilename=None,all_tags=None,trainfile=TRAIN_FILE,testfile=DEV_FILE,kaggle=False):
    if all_tags is None:
       all_tags = set()

       for i,(words, tags) in enumerate(preproc.conll_seq_generator(trainfile)):
           for tag in tags:
               all_tags.add(tag)
        
    with open(outfilename,'w') as outfile:
        idx = 1
        if kaggle:
            outfile.write("Id,Prediction")
        for words,_ in preproc.conll_seq_generator(testfile):
            pred_tags = tagger(words,all_tags)
            if kaggle:
                for tag in pred_tags:
                    outfile.write("\n")
                    outfile.write(str(idx) + "," + tag)
                    idx += 1
            else:
                for i,tag in enumerate(pred_tags):
                    print >>outfile, tag
                print >>outfile, ""

def eval_tagger(tagger,outfilename=None,all_tags=None,trainfile=TRAIN_FILE,testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger

    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    def get_outfile():
        if outfilename is not None:
            return open(outfilename,'w')
        else:
            return tempfile.NamedTemporaryFile('w',delete=False)
        
    with get_outfile() as outfile:
        apply_tagger(tagger,outfile.name,all_tags,trainfile,testfile)
        confusion = scorer.get_confusion(testfile,outfile.name) #run the scorer on the prediction file

    return confusion


