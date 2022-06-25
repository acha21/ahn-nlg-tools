# ahn-nlg-tools
Tools for developing natural language generation systems

This package includes the followings.
 1) Evaluation tool kit used in DSTC7 (angt/evaluation/dstc.py)
 2) Evaluation code (angt/evaluation/ground.py) for [**CbR task**](https://arxiv.org/pdf/1906.02738.pdf)
 3) Evaluation code (angt/evaluation/internet_ground.py) for evaluating the model's ability of utilizing knowledge in the KB. 

Specifically, 
The codes of 1) and 2) can evaluate Relevance(BLEU, NIST, METEOR, Ent-n, and Div-n) and [**Groundness**](https://arxiv.org/pdf/1906.02738.pdf), respectively.

The code 3) evaluate a multiple generated responses by measuring similarity with a single reference response.  
 
## Install

```bash
git clone git@github.com:acha21/ahn-nlg-tools.git
cd ahn-nlg-tools
python setup.py develop
```
Please install prerequisites as following installation [**instructions**](angt/README.md) for DSTC7 and CbR.
