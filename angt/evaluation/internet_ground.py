#!/usr/bin/env python3
from parlai.core.params import ParlaiParser, print_announcements

import os

from parlai_internal.scripts.utils.text import preprocess_text
from collections import Counter, OrderedDict
import jsonlines
from tqdm import tqdm

from parlai.core.script import ParlaiScript, register_script


def extract_cells(path_in, path_hash=None):
    if path_hash:
        if os.path.exists(path_hash):
            with open(path_hash) as fr:
                keys = [line.strip("\n") for line in fr.readlines()]
    else:
        keys = None

    cells = dict()
    with open(path_in, encoding="utf-8") as fr:
        for line in fr.readlines():
            c = line.strip("\n").split("\t")
            k = c[0]
            if keys:
                if k in keys:
                    cells[k] = c[1:]
            else:
                cells[k] = c[1:]
    return cells


def load_key_file(key_file):
    with open(key_file) as fr:
        keys = [line.strip("\n") for line in fr.readlines()]
    return set(keys)


def preprocess_text_list(textlist, stopwords, to_set=False):
    list_of_tokens = []
    for text in textlist:
        tokens = preprocess_text(text, stopwords)
        if to_set is True:
            tokens = set(tokens)
        list_of_tokens.append(list(tokens))
    return list_of_tokens


def calc_count(gold_responses_, ret_result_, context_, stopwords, remove_context_, is_gold_multi=True):
    if is_gold_multi is True:
        gold_responses_ = preprocess_text_list(gold_responses_, stopwords, True)
    else:
        gold_responses_ = preprocess_text(gold_responses_, stopwords)

    ret_result_ = preprocess_text_list(ret_result_, stopwords, True)
    context_keyword_ = set(preprocess_text(context_, stopwords))

    if remove_context_:
        new_ret_keyword = []
        for ret in ret_result_:
            new_ret = [tok for tok in ret if tok not in context_keyword_]
            new_ret_keyword.append(new_ret)
        ret_result_ = new_ret_keyword

    def concat_and_set(tokens_list):
        all_tokens = []
        for tokens in tokens_list:
            for tok in tokens:
                all_tokens.append(tok)
        all_tokens = set(all_tokens)
        return all_tokens
    if is_gold_multi:
        gold_resp_tokens = concat_and_set(gold_responses_)
    else:
        gold_resp_tokens = set(gold_responses_)
    gold_resp_token_cnt = Counter(gold_resp_tokens)
    gold_num = get_total_cnt_sum(gold_resp_token_cnt)

    pred_cnt = Counter([w for ret in ret_result_ for w in ret])
    pred_num = get_total_cnt_sum(pred_cnt)

    common_cnt = gold_resp_token_cnt & pred_cnt
    common_num = get_total_cnt_sum(common_cnt)

    return common_num, pred_num, gold_num


def get_total_cnt_sum(counter):
    return sum([cnt for cnt in counter.values()])


def load_vis_file(vis_file, key_file):
    objects = OrderedDict()
    keys = load_key_file(key_file)
    print("vis_file: ", vis_file)
    with jsonlines.open(vis_file, "r") as reader:
        for obj in reader:
            if obj['hash_id'] in keys:
                objects[obj['hash_id']] = obj
    return objects


def load_ref_file(ref_file, key_file):
    objects = extract_cells(ref_file, key_file)
    refs = {}
    for k in objects:
        strings = objects[k]
        refs[k] = [elem.split("|")[1] for elem in strings]
    return refs


def get_refs(refs, vshuman, n_refs):
    new_refs = []
    for i in range(n_refs):
        idx = i % len(refs)
        if idx == vshuman:
            idx = (idx + 1) % len(refs)
        new_refs.append(refs[idx])
    return new_refs


def run_on_single_file(vis_file, key_file, ref_file, stopwords, resp_num, vshuman, ret_num, remove_context_):

    refs_dict = load_ref_file(ref_file, key_file)
    ex_dict = load_vis_file(vis_file, key_file)

    common_cnt, pred_cnt, target_cnt = 0., 0., 0.

    for hash_id in tqdm(ex_dict, total=len(ex_dict)):
        golds = refs_dict[hash_id]
        cxt = ex_dict[hash_id].get('context', "")
        preds = ex_dict[hash_id]['top_ret_text']

        golds = get_refs(golds, vshuman, resp_num)
        preds = preds[:ret_num]
        com_cnt, p_cnt, t_cnt = calc_count(golds, preds, cxt, stopwords, remove_context_)
        common_cnt += com_cnt
        pred_cnt += p_cnt
        target_cnt += t_cnt

    recall = common_cnt / target_cnt
    precision = common_cnt / pred_cnt
    f1_score = 2 * precision * recall / (precision + recall)

    whole_words_set = set()
    for hash_id in ex_dict:
        preds = ex_dict[hash_id]['top_ret_text']
        preds_ = preds[:ret_num]
        all_words_set = set()
        for set_keywords in preprocess_text_list(preds_, stopwords, True):
            all_words_set = all_words_set.union(set(set_keywords))
        whole_words_set = whole_words_set.union(all_words_set)

    num_distct_words_in_ret = len(whole_words_set)

    return recall, precision, f1_score, num_distct_words_in_ret


def setup_args(parser=None):
    if parser is None:
        parser = ArgumentParser('Evaluate a retrieval model')
    parser.add_argument('--vis-submit', type=str)
    parser.add_argument("--resp_num", default=6)
    parser.add_argument("--vshuman", default=1, type=int)
    parser.add_argument("--ret_num", default=5)
    parser.add_argument('--ref_file', type=str, default="test.refs")
    parser.add_argument('--key_file', type=str, default="test.2k.txt")
    parser.add_argument('--report', type=str, default="report.tsv")
    parser.add_argument('--remove-context', type='bool', default=True)
    parser.add_argument(
        "--stopwords",
        default="/home/acha21/codes/ParlAI_dev/parlai_internal/scripts/stopwords_700+.txt",
    )
    parser.add_argument("--home", help="home directory")
    return parser


def run(args):

    ref_file = os.path.join(args.home, 'parlai_internal/scripts/', args.ref_file)
    key_file = os.path.join(args.home, 'parlai_internal/scripts/', args.key_file)

    stopwords = set([line.strip() for line in open(args.stopwords, mode="r").readlines()])
    if args.vis_submit.endswith(".jsonl"):
        recall, precision, f1_score, num_distct_words_in_ret = run_on_single_file(args.vis_submit, key_file, ref_file, stopwords, args.resp_num, args.vshuman, args.ret_num, args.remove_context)
        print(f"recall = {recall: 4.5f}")
        print(f"precision = {precision: 4.5f}")
        print(f"f1_score = {f1_score: 4.5f}")
        print(f"num_distct_words_in_ret = {num_distct_words_in_ret}")
    else:
        path_report = args.report
        with open(path_report, "w") as file:
            for i, path_ in enumerate(os.listdir(args.vis_submit)):
                file_path = os.path.join(args.vis_submit, path_)
                recall, precision, f1_score, num_distct_words_in_ret = run_on_single_file(file_path, key_file, ref_file, stopwords, args.resp_num, args.vshuman, args.ret_num, args.remove_context)
                print(f"evaluation finished on {file_path}")
                line = [file_path] + [str(v) for v in [recall, precision, f1_score, num_distct_words_in_ret]]
                line = "\t".join(line)
                if i == 0:
                    head = "\t".join(["file_path", 'recall', 'precision', 'f1_score', 'num_distct_words_in_ret'])
                    file.write(head + "\n")
                    print(head)
                file.write(line+"\n")
                print(line)
        print(path_report, "is written.")


if __name__ == "__main__":
    args = EvalKgcRet.setup_args()
    
