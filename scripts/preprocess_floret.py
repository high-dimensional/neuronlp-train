import re
from itertools import islice
from pathlib import Path

import spacy
import typer
from srsly import read_jsonl
from tqdm import tqdm


def main(
    dataset: Path,
    output_file: Path,
    max_texts: int,
    n_process: int = 8,
    batch_size: int = 128,
):
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 10**4

    data = read_jsonl(dataset)

    with open(output_file, "w") as output_fileh:
        texts = (
            re.sub("\s+", " ", line["text"].strip())
            for line in islice(iter(data), max_texts)
        )
        for doc in tqdm(nlp.pipe(texts, n_process=n_process, batch_size=batch_size)):
            for sent in doc.sents:
                output_fileh.write(" ".join([t.text for t in sent]) + "\n")


if __name__ == "__main__":
    typer.run(main)
