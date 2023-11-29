"""create spacy docbin data from entity-ruler patterns"""

import random
from pathlib import Path

import pandas as pd
import spacy
import srsly
from neuroNLP.custom_pipes import *
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

N = 50000
M = 40000
ROOT = Path("/home/hwatkins/Desktop")
SCIMODEL = ROOT / "neuroNLP/packages/en_core_sci_lg/en_core_sci_lg-0.5.0"
REPORTS = ROOT / "neuroData/cleaned_report_data/processed_reports.csv"
PATTERNS = ROOT / "neuroData/domain_patterns/terminology_data/patterns.json"
section_cls = (
    ROOT
    / "neuroNLP/packages"
    / "en_tok2vec_section_cls-1.0"
    / "en_tok2vec_section_cls"
    / "en_tok2vec_section_cls-1.0"
)
segmenter = (
    ROOT
    / "neuroNLP/packages"
    / "en_tok2vec_senter-1.0"
    / "en_tok2vec_senter"
    / "en_tok2vec_senter-1.0"
)
df = pd.read_csv(REPORTS, low_memory=False)
# texts = df.loc[
#    (df["report_body"].str.len() > 30) & (df["report_body"].str.len() < 5000),
#    "report_body",
# ]

texts = df.loc[
    (df["Narrative"].str.len() > 30) & (df["Narrative"].str.len() < 5000),
    "Narrative",
]

sample_texts = texts.sample(N).tolist()
sectioner = Sectioner(segmenter, section_cls)
sections = sectioner(sample_texts, batch_size=128)
body_texts = [
    i["BODY"] for i in tqdm(sections) if "BODY" in i.keys() if len(i["BODY"]) > 30
]
sci_nlp = spacy.load(SCIMODEL, exclude=["ner"])

all_patterns = srsly.read_json(PATTERNS)
token_patterns = []
phrase_patterns = [
    {"label": key, "pattern": t} for key, terms in all_patterns.items() for t in terms
]
ruler = sci_nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
ruler.add_patterns(phrase_patterns)

print("processing texts")

random.shuffle(sample_texts)

body_texts = body_texts[:M]

docs = [d for d in tqdm(sci_nlp.pipe(body_texts, batch_size=128, n_process=16))]
example = docs[0]

print(example)
print([(e.text, e.label_) for e in example.ents])
print(example[10].tag_)
print(example[10].dep_)
print(list(example.sents))

length = len(docs)

l, r = int(length * 0.9), int(length * 0.95)
train_docs = docs[:l]
dev_docs = docs[l:r]
test_docs = docs[r:]

train_bin = DocBin(docs=train_docs)
train_bin.to_disk(ROOT / "neuroData/full_model_data/train.spacy")
dev_bin = DocBin(docs=dev_docs)
dev_bin.to_disk(ROOT / "neuroData/full_model_data/dev.spacy")
test_bin = DocBin(docs=test_docs)
test_bin.to_disk(ROOT / "neuroData/full_model_data/test.spacy")
