import random
from pathlib import Path

import spacy
import typer
from neuradicon.custom_pipes import *
from spacy.tokens import Doc, DocBin, Span
from spacy.training.example import Example
from srsly import write_json
from tqdm import tqdm

activated = spacy.prefer_gpu()


def main(
    trained_pipeline: str = typer.Argument(..., exists=True, dir_okay=False),
    test_data: str = typer.Argument(..., exists=True, dir_okay=False),
    output: str = typer.Argument(..., exists=True, dir_okay=False),
):
    nlp = spacy.load(trained_pipeline, exclude=["ner"])
    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    examples = []
    for gold in tqdm(docs):
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        gold.ents = [
            Span(doc=gold, start=e.start, end=e.end, label="DESCRIPTOR")
            for e in gold.ents
        ]
        pred.ents = gold.ents
        for name, proc in nlp.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, gold))

    negation_detector = nlp.get_pipe("negex")
    scores = negation_detector.score(examples)

    print("Results of the model:")
    for key, val in scores.items():
        print("{} : {}".format(key, val))
    write_json(output, scores)


if __name__ == "__main__":
    typer.run(main)
