import argparse
from pathlib import Path

import spacy
import srsly
from neuradicon.custom_pipes import *
from sklearn.metrics import (confusion_matrix, multilabel_confusion_matrix,
                             precision_recall_fscore_support)
from spacy.tokens import DocBin

# model types, senter, textcat, textcat_multi, ner, relex, negex


def get_cat_label(doc):
    cat_dict = doc.cats
    return max(cat_dict, key=cat_dict.get)


def get_sent_label(doc):
    return [w.is_sent_start for w in doc]


def get_relex_label(doc):
    pass


def get_negex_label(doc):
    return [e._.is_negated for e in doc.ents]


def score_cat(gold_labels, pred_labels):
    labels = sorted(list(set(gold_labels)))
    prfs = precision_recall_fscore_support(
        gold_labels, pred_labels, average=None, labels=labels
    )
    micro = precision_recall_fscore_support(
        gold_labels, pred_labels, average="micro", labels=labels
    )
    macro = precision_recall_fscore_support(
        gold_labels, pred_labels, average="macro", labels=labels
    )
    weighted = precision_recall_fscore_support(
        gold_labels, pred_labels, average="weighted", labels=labels
    )
    conf_mat = confusion_matrix(gold_labels, pred_labels, labels=labels)
    output = {}
    keys = ["precision", "recall", "fscore", "support"]
    macro_spec = 0
    macro_spec_tn = 0
    macro_spec_fp = 0
    all_specs = []
    for i, label in enumerate(labels):
        output[label] = {keys[j]: val[i] for j, val in enumerate(prfs)}
        label_tn = (
            conf_mat.sum() - conf_mat[i].sum() - conf_mat[:, i].sum() + conf_mat[i, i]
        )
        label_fp = conf_mat[:, i].sum() - conf_mat[i, i]
        label_spec = label_tn / (label_tn + label_fp)
        output[label]["specificity"] = label_spec
        macro_spec_tn += label_tn
        macro_spec_fp += label_fp
        all_specs.append(label_spec)
    all_specs = np.array(all_specs)
    output["micro"] = dict(zip(keys, micro))
    output["micro"]["support"] = prfs[-1].sum()
    output["macro"] = dict(zip(keys, macro))
    output["macro"]["support"] = prfs[-1].sum()
    output["weighted"] = dict(zip(keys, weighted))
    output["weighted"]["support"] = prfs[-1].sum()
    output["micro"]["specificity"] = macro_spec_tn / (macro_spec_tn + macro_spec_fp)
    output["macro"]["specificity"] = all_specs.mean()
    output["weighted"]["specificity"] = (all_specs * prfs[-1]).sum() / prfs[-1].sum()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("data_path")
    parser.add_argument("model_type")
    args = parser.parse_args()

    nlp = spacy.load(args.model_path)
    doc_bin = DocBin().from_disk(args.data_path)
    gold_labels, pred_labels = [], []
    for gold_doc in doc_bin.get_docs(nlp.vocab):
        if args.model_type == "textcat":
            pred_doc = nlp(gold_doc.text)
            gold_labels.append(get_cat_label(gold_doc))
            pred_labels.append(get_cat_label(pred_doc))
        elif args.model_type == "senter":
            # senter = nlp.get_pipe("senter")
            gold_labels.extend(get_sent_label(gold_doc))
            pred_doc = nlp(gold_doc.text)  # senter(gold_doc)
            pred_labels.extend(get_sent_label(pred_doc))
        elif args.model_type == "negex":
            pred_doc = Doc(
                nlp.vocab,
                words=[t.text for t in gold_doc],
                spaces=[t.whitespace_ for t in gold_doc],
            )
            gold_doc.ents = [
                Span(doc=gold_doc, start=e.start, end=e.end, label="DESCRIPTOR")
                for e in gold_doc.ents
            ]
            pred_doc.ents = gold_doc.ents
            negex = nlp.get_pipe("negex")
            gold_labels.extend(get_negex_label(gold_doc))
            # gold_doc._.neg = []
            pred_doc = negex(pred_doc)
            pred_labels.extend(get_negex_label(pred_doc))

    if args.model_type in ["textcat", "senter", "negex"]:
        print(score_cat(gold_labels, pred_labels))


if __name__ == "__main__":
    main()
