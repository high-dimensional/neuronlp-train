"""Preprare data for hierarchical binary classifiers"""
import random
from pathlib import Path

import pandas as pd
import srsly

# %%
ROOT = Path("/home/hwatkins/Desktop/neuroData/normality_data/multilabel_normality")
ROOT2 = Path("/home/hwatkins/Desktop/neuroData")
d1 = list(srsly.read_jsonl(ROOT / "test.jsonl"))
d2 = list(srsly.read_jsonl(ROOT / "dev.jsonl"))
d3 = list(srsly.read_jsonl(ROOT / "train.jsonl"))
data = d1 + d2 + d3
# %%
records = [
    {
        "text": i["text"],
        "NORMAL": i["cats"]["Normal"] > 0.5,
        "MISSING": i["cats"]["Missing"] > 0.5,
        "COMPARATIVE": i["cats"]["Comparative"] > 0.5,
        "ABNORMAL": i["cats"]["Abnormal"] > 0.5,
        "NORMALFORAGE": i["cats"]["Normal_for_age"] > 0.5,
    }
    for i in data
]
df = pd.DataFrame.from_records(records)

# %%
miss_pos = df[df["MISSING"]]
miss_pos["IS_MISSING"] = True
miss_neg = df[~df["MISSING"]]
miss_neg["IS_MISSING"] = False
missing_df = pd.concat([miss_neg.sample(len(miss_pos)), miss_pos])
# %%
comp_pos = miss_neg[miss_neg["COMPARATIVE"]]
comp_pos["IS_COMPARATIVE"] = True
comp_neg = miss_neg[~miss_neg["COMPARATIVE"]]
comp_neg["IS_COMPARATIVE"] = False
comp_df = pd.concat([comp_neg.sample(len(comp_pos)), comp_pos])
# %%
norm_pos = miss_neg[miss_neg["NORMAL"] & ~miss_neg["ABNORMAL"]]
norm_pos["IS_NORMAL"] = True
norm_neg = miss_neg[
    (miss_neg["ABNORMAL"] & ~miss_neg["NORMAL"]) | miss_neg["NORMALFORAGE"]
]
norm_neg["IS_NORMAL"] = False
norm_df = pd.concat([norm_neg.sample(len(norm_pos)), norm_pos])
# %%
agenorm_pos = norm_neg[norm_neg["NORMALFORAGE"]]
agenorm_pos["IS_NORMAL_FOR_AGE"] = True
agenorm_neg = norm_neg[~norm_neg["NORMALFORAGE"]]
agenorm_neg["IS_NORMAL_FOR_AGE"] = False
agenorm_df = pd.concat([agenorm_neg.sample(len(agenorm_pos)), agenorm_pos])


# %%
def make_split(stratified_df, col):
    local_df = stratified_df[["text", col]]
    all_reports = [
        {"text": t, "label": col, "answer": "accept" if v else "reject"}
        for _, t, v in local_df.to_records()
    ]
    random.shuffle(all_reports)
    length = len(all_reports)
    print(length)
    l, r = int(0.9 * length), int(0.95 * length)
    train, dev, test = all_reports[:l], all_reports[l:r], all_reports[r:]
    return train, dev, test


# %%
missloc = "missing_data"
misstrain, missdev, misstest = make_split(missing_df, "IS_MISSING")
srsly.write_jsonl(ROOT2 / missloc / "train.jsonl", misstrain)
srsly.write_jsonl(ROOT2 / missloc / "dev.jsonl", missdev)
srsly.write_jsonl(ROOT2 / missloc / "test.jsonl", misstest)
# %%
comploc = "comparitive_data/comparative_data_v2"
comptrain, compdev, comptest = make_split(comp_df, "IS_COMPARATIVE")
srsly.write_jsonl(ROOT2 / comploc / "train.jsonl", comptrain)
srsly.write_jsonl(ROOT2 / comploc / "dev.jsonl", compdev)
srsly.write_jsonl(ROOT2 / comploc / "test.jsonl", comptest)
# %%
normloc = "normality_data/binary_normality_v2"
normtrain, normdev, normtest = make_split(norm_df, "IS_NORMAL")
srsly.write_jsonl(ROOT2 / normloc / "train.jsonl", normtrain)
srsly.write_jsonl(ROOT2 / normloc / "dev.jsonl", normdev)
srsly.write_jsonl(ROOT2 / normloc / "test.jsonl", normtest)
# %%
agenormloc = "normality_data/age_normality"
agenormtrain, agenormdev, agenormtest = make_split(agenorm_df, "IS_NORMAL_FOR_AGE")
srsly.write_jsonl(ROOT2 / agenormloc / "train.jsonl", agenormtrain)
srsly.write_jsonl(ROOT2 / agenormloc / "dev.jsonl", agenormdev)
srsly.write_jsonl(ROOT2 / agenormloc / "test.jsonl", agenormtest)
