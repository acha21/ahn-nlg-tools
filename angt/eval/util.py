import io
import os
import sys

import numpy as np  # type: ignore

py_version = sys.version.split(".")[0]
if py_version == "2":
    open = io.open
else:
    unicode = str


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)


old_settings = np.seterr(all="print")


def str2bool(s):
    # to avoid issue like this: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if s.lower() in ["t", "true", "1", "y"]:
        return True
    elif s.lower() in ["f", "false", "0", "n"]:
        return False
    else:
        raise ValueError


def f1_score(p, r):
    try:
        if p == 0.0 and r == 0.0:
            return 0.0
        f1 = 2.0 * p * r / float(p + r)
    except ZeroDivisionError:
        f1 = 1e-20

    return f1
