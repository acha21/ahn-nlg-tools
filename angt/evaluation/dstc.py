# author: Xiang Gao @ Microsoft Research, Oct 2018
import os
import sys
import argparse
from angt import KNOWLEDGE_PATH, NLG_TOOLS_DIR
import jsonlines
from nltk.tokenize import TweetTokenizer
from .preprocessing import clean_str
from .metrics import unicode, nlp_metrics
from .ground import count_grounded_v2, count_grounded_v1

tw_tokenizer = TweetTokenizer()


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


def extract_hyp_refs(
    raw_hyp, raw_ref, path_hash, fld_out, n_refs=6, clean=False, vshuman=-1
):
    cells_hyp = extract_cells(raw_hyp, path_hash)
    cells_ref = extract_cells(raw_ref, path_hash)
    if not os.path.exists(fld_out):
        os.makedirs(fld_out)

    def _clean(s):
        if clean:
            return clean_str(s)
        else:
            return s

    keys = sorted(cells_hyp.keys())
    with open(fld_out + "/hash.txt", "w", encoding="utf-8") as f:
        f.write(unicode("\n".join(keys)))

    lines = [_clean(cells_hyp[k][-1]) for k in keys]
    path_hyp = fld_out + "/hyp.txt"
    with open(path_hyp, "w", encoding="utf-8") as f:
        f.write(unicode("\n".join(lines)))

    lines = []
    for _ in range(n_refs):
        lines.append([])
    for k in keys:
        refs = cells_ref[k]
        for i in range(n_refs):
            idx = i % len(refs)
            if idx == vshuman:
                idx = (idx + 1) % len(refs)
            lines[i].append(_clean(refs[idx].split("|")[1]))

    path_refs = []
    for i in range(n_refs):
        path_ref = fld_out + "/ref%i.txt" % i
        with open(path_ref, "w", encoding="utf-8") as f:
            f.write(unicode("\n".join(lines[i])))
        path_refs.append(path_ref)

    return path_hyp, path_refs


def read_hyp(
    raw_hyp, path_hash, clean=False
):
    cells_hyp = extract_cells(raw_hyp, path_hash)
    keys = sorted(cells_hyp.keys())

    def _clean(s):
        if clean:
            return clean_str(s)
        else:
            return s

    def _tokenize(strings):
        #return " ".join(_clean(strings).split())
        return " ".join(tw_tokenizer.tokenize(_clean(strings)))
        #return " ".join(tw_tokenizer.tokenize(_clean(strings)))

    #lines = #[[k] + cells_hyp[k][:-2] + [_tokenize(cells_hyp[k][-2])] + [_tokenize(cells_hyp[k][-1])] for k in keys]
    lines = [[k] + cells_hyp[k][:-2] + [_tokenize(cells_hyp[k][-2])] + [_tokenize(cells_hyp[k][-1])] for k in keys]
    return lines


def get_fact_dict(knowledge_file, clean):
    fact = {}
    processed_doc = {}

    def _clean(s):
        if clean:
            return clean_str(s)
        else:
            return s

    def _tokenize(strings):
        return " ".join(tw_tokenizer.tokenize(_clean(strings)))

    with jsonlines.open(knowledge_file, "r") as fin:
        for obj in fin:
            hash_id = obj['hash_id']
            given_doc = obj['given_doc']
            conv_id = obj['conv_id']
            if conv_id not in processed_doc:
                processed_doc[conv_id] = _tokenize(given_doc)
            fact[hash_id] = processed_doc[conv_id]
    return fact


def eval_one_system(
    submitted,
    keys,
    multi_ref,
    n_refs=6,
    n_lines=None,
    clean=False,
    vshuman=-1,
    PRINT=True,
    eval_ground=False,
    eval_gen=True,
    eval_ground_version="v2",
):
    print("evaluating %s" % submitted)

    fld_out = submitted.replace(".txt", "")
    if clean:
        fld_out += "_cleaned"

    path_hyp, path_refs = extract_hyp_refs(
        submitted, multi_ref, keys, fld_out, n_refs, clean=clean, vshuman=vshuman
    )

    if n_lines is None:
        n_lines = len(open(path_hyp, encoding="utf-8").readlines())

    result = [n_lines]

    if eval_ground:
        if eval_ground_version == 'v2':
            count_grounded = count_grounded_v2
        elif eval_ground_version == 'v1':
            count_grounded = count_grounded_v1
        else:
            count_grounded = count_grounded_v2
        print("eval_ground_version = " + eval_ground_version)
        if NLG_TOOLS_DIR == "":
            raise ValueError("NLG_TOOLS_DIR is not set.")
        facts_dic = get_fact_dict(KNOWLEDGE_PATH, True)
        fresult = read_hyp(submitted, keys, True)
        facts = []
        for line_res in fresult:
            hash_key = line_res[0]
            facts.append(facts_dic[hash_key])
        fact_dict, precision, recall, f1, g_count, w_count, f_count = \
            count_grounded(facts, fresult)

        if PRINT:
            print('grounded words: {}'.format(g_count))
            print('words: {}'.format(w_count))
            print('fact_len: {}\n'.format(f_count))
            print('precision: {:.2f}%'.format(precision * 100))
            print('recall: {:.2f}%'.format(recall * 100))
            print('f1: {:.2f}%'.format(f1 * 100))

        result += [precision, recall, f1, f_count, w_count, g_count]

    if eval_gen:
        nist, bleu, meteor, entropy, div, avg_len = nlp_metrics(
            path_refs, path_hyp, fld_out, n_lines=n_lines
        )

        if PRINT:
            print("n_lines = " + str(n_lines))
            print("NIST = " + str(nist))
            print("BLEU = " + str(bleu))
            print("METEOR = " + str(meteor))
            print("entropy = " + str(entropy))
            print("diversity = " + str(div))
            print("avg_len = " + str(avg_len))

        result += nist + bleu + [meteor] + entropy + div + [avg_len]

    return result


def eval_all_systems(
    files,
    keys,
    multi_ref,
    path_report=None,
    n_refs=6,
    n_lines=None,
    clean=False,
    vshuman=False,
    eval_ground=False,
    eval_gen=True,
    eval_ground_version='v2',
):
    # evaluate all systems (*.txt) in each folder `files`
    report_out = []

    columns = ["fname", "n_lines"]
    if eval_ground:
        columns += ['precision', 'recall', 'f1', 'f_count', 'w_count', 'g_count']
    if eval_gen:
        columns += ["nist%i" % i for i in range(1, 4 + 1)]  + ["bleu%i" % i for i in range(1, 4 + 1)] + \
              ["meteor"] + ["entropy%i" % i for i in range(1, 4 + 1)] + ["div1", "div2", "avg_len"]

    head = "\t".join(columns)
    report_out.append(head)
    if path_report:
        with open(path_report, "w") as f:
            f.write(head + "\n")

    for fl in files:
        if fl.endswith(".txt"):
            submitted = fl
            results = eval_one_system(
                submitted,
                keys=keys,
                multi_ref=multi_ref,
                n_refs=n_refs,
                clean=clean,
                n_lines=n_lines,
                vshuman=vshuman,
                PRINT=False,
                eval_ground=eval_ground,
                eval_gen=eval_gen,
                eval_ground_version=eval_ground_version,
            )
            if path_report:
                with open(path_report, "a") as f:
                    f.write(",".join(map(str, [submitted] + results)) + "\n")

            report_out.append("\t".join(map(str, [submitted] + results)))
        else:
            for fname in os.listdir(fl):
                if fname.endswith(".txt"):
                    submitted = fl + "/" + fname
                    results = eval_one_system(
                        submitted,
                        keys=keys,
                        multi_ref=multi_ref,
                        n_refs=n_refs,
                        clean=clean,
                        n_lines=n_lines,
                        vshuman=vshuman,
                        PRINT=False,
                        eval_ground=eval_ground,
                        eval_gen=eval_gen,
                        eval_ground_version=eval_ground_version,
                    )

                    if path_report:
                        with open(path_report, "a") as f:
                            f.write(",".join(map(str, [submitted] + results)) + "\n")

                    report_out.append("\t".join(map(str, [submitted] + results)))
    if path_report:
        print("report saved to: " + path_report, file=sys.stderr)

    return report_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "submitted"
    )  # if 'all' or '*', eval all teams listed in dstc/teams.txt
    # elif endswith '.txt', eval this single file
    # else, eval all *.txt in folder `submitted_fld`

    parser.add_argument(
        "--clean", "-c", action="store_true"
    )  # whether to clean ref and hyp before eval
    parser.add_argument(
        "--n_lines", "-n", type=int, default=-1
    )  # eval all lines (default) or top n_lines (e.g., for fast debugging)
    parser.add_argument("--n_refs", "-r", type=int, default=6)  # number of references
    parser.add_argument(
        "--vshuman", "-v", type=int, default="1"
    )  # when evaluating against human performance (N in refN.txt that should be removed)
    # in which case we need to remove human output from refs`
    parser.add_argument("--refs", "-g", default="data/eval/keys/test.refs")
    parser.add_argument("--keys", "-k", default="data/eval/keys/test.2k.txt")
    parser.add_argument("--teams", "-i", type=str, default="dstc/teams.txt")
    parser.add_argument("--report", "-o", type=str)
    parser.add_argument("--ground", "-u", action='store_true')
    parser.add_argument("--no_eval_gen", default=False, action='store_true')
    parser.add_argument("--eval_ground_version", "-e", type=str, default='v2')

    args = parser.parse_args()
    print("Args: %s\n" % str(args), file=sys.stderr)

    if args.n_lines < 0:
        n_lines = None  # eval all lines
    else:
        n_lines = args.n_lines  # just eval top n_lines

    if args.submitted.endswith(".txt"):
        eval_one_system(
            args.submitted,
            keys=args.keys,
            multi_ref=args.refs,
            clean=args.clean,
            n_lines=n_lines,
            n_refs=args.n_refs,
            vshuman=args.vshuman,
            eval_ground=args.ground,
            eval_gen=not args.no_eval_gen,
            eval_ground_version=args.eval_ground_version,
        )
    else:
        fname_report = "report_ref%i" % args.n_refs
        if args.clean:
            fname_report += "_cleaned"
        fname_report += ".tsv"
        if args.submitted == "all" or args.submitted == "*":
            files = ["dstc/" + line.strip("\n") for line in open(args.teams)]
            path_report = "dstc/" + fname_report
        else:
            files = [args.submitted]
            path_report = args.submitted + "/" + fname_report
        if args.report != None:
            path_report = args.report
        eval_all_systems(
            files,
            keys=args.keys,
            multi_ref=args.refs,
            path_report=path_report,
            clean=args.clean,
            n_lines=n_lines,
            n_refs=args.n_refs,
            vshuman=args.vshuman,
            eval_ground=args.ground,
            eval_gen=not args.no_eval_gen,
            eval_ground_version=args.eval_ground_version,
        )
