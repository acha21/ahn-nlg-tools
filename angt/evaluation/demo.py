# evaluation
import os
from angt.evaluation.metrics import nlp_metrics
from angt.evaluation.preprocessing import clean_str
from angt import NLG_TOOLS_DIR
print(NLG_TOOLS_DIR)
demo_path = os.path.join(NLG_TOOLS_DIR, "data/eval/demo")
nist, bleu, meteor, entropy, diversity, avg_len = nlp_metrics(
    path_refs=[os.path.join(demo_path, "ref0.txt"), os.path.join(demo_path, "ref1.txt")],
    path_hyp=os.path.join(demo_path, "hyp.txt")
)

print(nist)
print(bleu)
print(meteor)
print(entropy)
print(diversity)
print(avg_len)

# tokenization

s = " I don't know:). how about this?https://github.com/golsun/deep-RL-time-series"
print(clean_str(s))
