#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    NLG tools for acha21
"""
import os

NLG_TOOLS_DIR = os.path.dirname(os.path.realpath(__file__))

STOPWORD_PATH = os.path.join(NLG_TOOLS_DIR, "data/cmr/stopwords_700+.txt")

KNOWLEDGE_PATH = os.path.join(NLG_TOOLS_DIR, "data/eval/keys/reddit_wiki_doc.jsonl")

__all__ = ['evaluation', 'STOPWORD_PATH', 'KNOWLEDGE_PATH', 'NLG_TOOLS_DIR']

from . import evaluation