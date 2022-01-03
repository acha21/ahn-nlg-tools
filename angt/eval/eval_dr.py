import argparse
from .dstc import extract_cells
from .preprocessing import clean_str
from .util import *
import tempfile
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.nist_score import sentence_nist

from functools import partial

smooth_bleu = partial(sentence_bleu, smoothing_function=SmoothingFunction().method7)



third_party_path = "data/eval/"


def parse_args():
    parser = argparse.ArgumentParser()
    folder_parser = parser.add_mutually_exclusive_group(required=True)
    folder_parser.add_argument("--single_system", "-S", type=str, required=False)
    folder_parser.add_argument(
        "--systems",
        "-T",
        type=str,
        required=False,
        help="dir where submissions of multiple systems exists, "
             "every submission file of the same system should be in a same directory",
    )
    parser.add_argument(
        "--dataset", "-d", choices=["reddit", "holle"], default="reddit"
    )
    parser.add_argument(
        "--mode_split",
        "-m",
        type=str,
        help="argument for reader.read()",
        default="official:test",
    )
    parser.add_argument("--n_refs", "-r", type=int, default=5)  # number of references
    parser.add_argument(
        "--n_lines", "-n", type=int, default=-1
    )  # eval all lines (default) or top n_lines (e.g., for fast debugging)
    parser.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help="whether to clean ref and hyp before eval",
    )
    parser.add_argument("--report", "-o", type=str, default=None)
    parser.add_argument(
        "--no_verbose", "-v", action="store_false", dest="verbose", default=True
    )
    parser.add_argument("--keys", default="data/eval/keys/reddit.div_auto.keys")
    parser.add_argument("--refs", "-g", default="data/eval/keys/test.refs")
    parser.add_argument(
        "--by_product_path",
        "-b",
        default=None,
        help="by default, subs[0]'s path is used "
             "if `tmp` is used, we save them in /tmp, "
             "else user defined path is used",
    )
    # test codes
    # cmd = "-S test_fixtures/reddit/eval_div/systems/sys1 " \
    #       "--keys test_fixtures/reddit/eval_div/test.key --refs test_fixtures/reddit/eval_div/test.refs " \
    #       "--report test_fixtures/reddit/eval_div/report.tsv"

    # cmd = "-T test_fixtures/reddit/eval_div/systems " \
    #       "--keys test_fixtures/reddit/eval_div/test.key --refs test_fixtures/reddit/eval_div/test.refs " \
    #       "--report test_fixtures/reddit/eval_div/report.tsv"
    # return parser.parse_args(cmd.split())
    return parser.parse_args()


def calc_metric(refs_lines, hyps_lines, metric_func):
    num_submitted_hyps = len(hyps_lines[0])
    hyp_num = len(hyps_lines)
    num_queries = len(refs_lines[0])
    ref_num = len(refs_lines)

    for hyp_i, hyps in enumerate(hyps_lines):
        assert num_submitted_hyps == len(hyps)

    for ref_j, refs in enumerate(refs_lines):
        assert num_queries == len(refs)

    refs_lines = np.array(refs_lines)
    refs_lines = np.transpose(refs_lines)
    hyps_lines = np.array(hyps_lines)
    hyps_lines = np.transpose(hyps_lines)

    num_eval_exs = min([num_submitted_hyps, num_queries])
    scores = np.zeros((num_submitted_hyps, hyp_num, ref_num))

    for ex_index in range(num_eval_exs):
        hyps = hyps_lines[ex_index]
        refs = refs_lines[ex_index]
        for hyp_i in range(hyp_num):
            hyp = hyps[hyp_i]
            for ref_j in range(ref_num):
                ref = refs[ref_j]
                hyp_tokens = hyp.split()
                if len(hyp_tokens) <= 5:
                    hyp_tokens += [""] * (5 - len(hyp_tokens))
                scores[ex_index][hyp_i][ref_j] = metric_func([ref.split()], hyp_tokens)

    precisions = np.max(scores, axis=2)
    recalls = np.max(scores, axis=1)

    avg_precisions = np.mean(precisions, axis=1)
    avg_recalls = np.mean(recalls, axis=1)
    f1_list = []
    for i in range(num_submitted_hyps):
        p, r = avg_precisions[i], avg_recalls[i]
        f1_list.append(f1_score(p, r))

    metrics = dict(
        nlines=num_eval_exs,
        F1=np.mean(f1_list),
        prec=np.mean(avg_precisions),
        recall=np.mean(avg_recalls),
        F1_list=np.array(f1_list),
        prec_list=avg_precisions,
        recall_list=avg_recalls,
    )

    return metrics


def calc_bleu(refs_lines, hyps_lines):
    return calc_metric(refs_lines, hyps_lines, smooth_bleu)


def nlp_metrics(refs_lines, hyps_lines):
    nist = calc_metric(refs_lines, hyps_lines, sentence_nist)
    bleu = calc_bleu(refs_lines, hyps_lines)
    return nist, bleu


def extract_refs(raw_ref, path_hash, out_dir, keys, n_refs=6, clean=False):
    # path_hash: target hash key provided by separated file
    def _clean(s):
        if clean:
            return clean_str(s)
        else:
            return s

    cells_ref = extract_cells(raw_ref, path_hash)
    lines = []
    for _ in range(n_refs):
        lines.append([])
    for k in keys:
        refs = cells_ref[k]
        for i in range(n_refs):
            idx = i % len(refs)
            cleaned = _clean(refs[idx].split("|")[1])
            if len(cleaned) == 0:
                cleaned = refs[idx].split("|")[1]
            lines[i].append(cleaned)

    path_refs = []
    for i in range(n_refs):
        path_ref = out_dir + "/ref%i.txt" % i
        with open(path_ref, "w", encoding="utf-8") as f:
            f.write(unicode("\n".join(lines[i])))
        path_refs.append(path_ref)

    return path_refs, lines


def extract_hyp(raw_hyp, path_hash, out_dir, clean=False):
    def _clean(s):
        if clean:
            return clean_str(s)
        else:
            return s

    cells_hyp = extract_cells(raw_hyp, path_hash)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    keys = sorted(cells_hyp.keys())
    with open(out_dir + "/hash.txt", "w", encoding="utf-8") as f:
        f.write(unicode("\n".join(keys)))

    lines = [_clean(cells_hyp[k][-1]) for k in keys]
    path_hyp = out_dir + "/hyp.txt"
    with open(path_hyp, "w", encoding="utf-8") as f:
        f.write(unicode("\n".join(lines)))

    return path_hyp, keys, lines


def eval_one_system(
        subs,
        keys_from_file,
        multi_ref,
        n_refs=6,
        n_lines=None,
        clean=False,
        PRINT=True,
        by_product_path=None,
):
    print("evaluating %s" % ",".join(subs))
    keys, prev_keys = None, None
    hyp_lines = []
    for _ in range(len(subs)):
        hyp_lines.append([])

    path_hyps = []

    for i, submitted in enumerate(subs):
        dirname_for_this_submit = os.path.basename(submitted).replace(".txt", "")
        if clean:
            dirname_for_this_submit += "_cleaned"

        path_hyp, keys, hyps = extract_hyp(
            submitted,
            keys_from_file,
            os.path.join(by_product_path, dirname_for_this_submit),
            clean=clean,
        )
        n_lines = len(hyps)
        if i == 0:
            prev_keys = keys
        else:
            assert prev_keys == keys

        for hyp in hyps:
            hyp_lines[i].append(hyp)

        path_hyps.append(path_hyp)

    ref_dir = by_product_path + "/refs"
    os.makedirs(ref_dir, exist_ok=True)
    path_refs, refs = extract_refs(
        multi_ref, keys_from_file, ref_dir, keys, n_refs, clean=clean
    )
    print("By-product files")
    print("References: ", path_refs)
    print("Hypothesese: ", path_hyps)
    nist, bleu = nlp_metrics(refs, hyp_lines)

    filenames = "~".join([os.path.basename(subs[0]), os.path.basename(subs[-1])])
    num_hyps = len(subs)

    if PRINT:
        print(f" - Submissions to evaluate: {len(subs)} files - ", filenames)
        print(" - n_lines = " + str(n_lines))
        print(" - NIST = " + str(nist))
        print(" - BLEU = " + str(bleu))
        print()

    result = [
        filenames,
        n_lines,
        num_hyps,
        nist["prec"],
        nist["recall"],
        nist["F1"],
        bleu["prec"],
        bleu["recall"],
        bleu["F1"],
    ]

    return result


def print_result(results, filepath):
    report_out = []
    head = "\t".join(
        ["fname", "n_lines", "num_hyps"]
        + ["prec_nist", "recall_nist", "f1_nist"]
        + ["prec_bleu", "recall_bleu", "f1_bleu"]
    )

    report_out.append(head)
    with open(filepath, "w") as f:
        f.write(head + "\n")
        for result in results:
            row = "\t".join(map(str, result))
            f.write(row + "\n")


def set_byproduct_path(by_product_path, submissions_filepaths):
    if by_product_path == "tmp":
        by_product_path = tempfile.TemporaryDirectory()
    elif by_product_path is None:
        by_product_path = os.path.dirname(submissions_filepaths[0])
    else:
        os.makedirs(args.by_product_path, exist_ok=True)
        by_product_path = by_product_path
    return by_product_path


def main(args):
    print("Args: %s\n" % str(args), file=sys.stderr)

    if args.n_lines < 0:
        n_lines = None  # eval all lines
    else:
        n_lines = args.n_lines  # just eval top n_lines

    def get_submission_full_paths(target_dir_):
        subfilepaths = []
        for subfilename in os.listdir(target_dir_):
            subfilepath = os.path.join(target_dir_, subfilename)
            if os.path.exists(subfilepath) and os.path.isfile(subfilepath):
                subfilepaths.append(subfilepath)
        return subfilepaths

    if args.single_system:
        target_dir = args.single_system
        subfilepaths = get_submission_full_paths(target_dir)
        by_product_path = set_byproduct_path(args.by_product_path, subfilepaths)

        result = eval_one_system(
            subfilepaths,
            keys_from_file=args.keys,
            multi_ref=args.refs,
            clean=args.clean,
            n_lines=n_lines,
            n_refs=args.n_refs,
            by_product_path=by_product_path,
            PRINT=args.verbose,
        )
        print_result([result], args.report)
        print(f"By-products are saved in {by_product_path}")
        if args.report:
            print(f"We wrote the result in {args.report}.")
    else:
        target_dir = args.systems
        results = []
        dirnames = []

        if os.path.exists(target_dir) is False:
            raise NotADirectoryError(f"{target_dir} is not appropriate.")

        for sys_dirname in sorted(os.listdir(target_dir)):
            sys_fullpath = os.path.join(target_dir, sys_dirname)
            dirnames.append(sys_dirname)

            submission_files = get_submission_full_paths(sys_fullpath)
            by_product_path = set_byproduct_path(args.by_product_path, submission_files)
            result = eval_one_system(
                submission_files,
                keys_from_file=args.keys,
                multi_ref=args.refs,
                clean=args.clean,
                n_lines=n_lines,
                n_refs=args.n_refs,
                by_product_path=by_product_path,
                PRINT=args.verbose,
            )
            results.append(result)

        for i, dirname in enumerate(dirnames):
            results[i][0] = dirname
        print_result(results, args.report)
        if args.report:
            print(f"We wrote the result in {args.report}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
