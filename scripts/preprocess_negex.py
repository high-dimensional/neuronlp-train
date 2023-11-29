from pathlib import Path

import spacy
import srsly
import typer
from spacy.tokens import Doc, DocBin
from spacy.util import get_words_and_spaces


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Argument(..., dir_okay=False),
):
    nlp = spacy.blank("en")
    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"], store_user_data=True)
    Doc.set_extension("neg", default=[])
    for eg in srsly.read_jsonl(input_path):
        if eg["answer"] != "accept":
            continue
        tokens = [token["text"] for token in eg["tokens"]]
        words, spaces = get_words_and_spaces(tokens, eg["text"])
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        ents = []
        negs = []
        for s in eg.get("spans", []):
            new_ent = doc.char_span(s["start"], s["end"], label="DESCRIPTOR")
            ents.append(new_ent)
            if s["label"] == "DENIAL":
                negs.extend([w.i for w in new_ent])
        doc.ents = ents
        doc._.neg = negs
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)
    print(f"Processed {len(doc_bin)} documents: {output_path.name}")


if __name__ == "__main__":
    typer.run(main)
