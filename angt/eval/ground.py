from angt import STOPWORD_PATH
from nltk import stem
stemmer = stem.PorterStemmer()


def get_head(verbose, include_n_lines):
    strings = ""
    strings += "filename\t"
    if include_n_lines:
        strings += "n_lines\t"
    strings += "prec\trecall\tf1\tprec_gt\trecall_gt\tf1_gt"
    if verbose:
        strings += "\tmicro_prec\tmicro_recall\tf1\tmicro_prec_gt\tmicro_recall_gt\tmicro_f1_gt"
    return strings


def get_stop_words():
    stop_words = list()
    with open(STOPWORD_PATH, "r") as fin:
        for line in fin.readlines():
            s = line.strip()
            stop_words.append(s)
    return stop_words


def stemming(words):
    if type(words) == str:
        words = words.strip().split()
    #stemmed = [stemmer.stem(w) for w in words]
    stemmed = [w for w in words if w.strip() != ""]
    return stemmed


def count_grounded(facts, fresult):
    g_count = 0
    w_count = 0
    f_count = 0
    lines_count = 0
    fact_dict = dict()
    stop_words = set(get_stop_words())
    assert len(facts) == len(fresult)

    res_set = set()

    for i, (fact, result) in enumerate(zip(facts, fresult)):
        fact = set(fact.strip().split())
        #fact = fact.strip().split()
        id = result[0]
        que = set(result[-2].strip().split())
        #que = result[-2].split()
        res = result[-1]

        lines_count += 1

        if res in res_set:
            continue
        else:
            res_set.add(res)
        #res = res.split()
        res = set(res.strip().split())


        fact_dict[id] = fact
        grounded = []
        qued = []
        for x in stemming(res):
            w_count += 1
            if x not in stop_words:
                if x not in stemming(que):
                    if x in stemming(fact):
                        grounded.append(x)
                        g_count += 1
                else:
                    qued.append(x)

        for x in stemming(fact):
            if x not in stop_words:
                f_count += 1
    if w_count == 0:
        precision = 0.0
    else:
        precision = g_count / w_count
    if f_count == 0:
        recall = 0.0
    else:
        recall = g_count / f_count
    if precision == 0.0 and recall == 0.0:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return fact_dict, precision, recall, f1, g_count, w_count, f_count

