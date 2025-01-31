import pandas as pd
import os
from datetime import datetime

def preprocess_mind(mind_path, output_path, dataset_name="MIND-small"):
    # Load behaviors.tsv and news.tsv
    behaviors = pd.read_csv(os.path.join(mind_path, "behaviors.tsv"), sep="\t", 
                            names=["impression_id", "user_id", "time", "history", "impressions"])
    news = pd.read_csv(os.path.join(mind_path, "news.tsv"), sep="\t", 
                       names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])

    # Apply 10-core filter (as in the paper)
    user_counts = behaviors["user_id"].value_counts()
    behaviors = behaviors[behaviors["user_id"].isin(user_counts[user_counts >= 10].index)]
    news_counts = behaviors["impressions"].str.split("-").str[0].value_counts()
    valid_news = news_counts[news_counts >= 10].index
    behaviors["impressions"] = behaviors["impressions"].apply(lambda x: [imp for imp in x.split() if imp.split("-")[0] in valid_news])

    # Split into training/validation/test (adjust for full dataset)
    if dataset_name == "MIND-small":
        # Use predefined splits: MINDsmall_train (train) and MINDsmall_dev (val)
        train_behaviors = behaviors
        val_behaviors = pd.read_csv(os.path.join(mind_path.replace("train", "dev"), "behaviors.tsv"), sep="\t")
    else:
        # For full dataset: Split first 3 days for train, last 2 days for val/test (paper’s setup)
        behaviors["timestamp"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
        train_behaviors = behaviors[behaviors["timestamp"].dt.day <= 3]
        val_test = behaviors[behaviors["timestamp"].dt.day > 3]
        val_behaviors, test_behaviors = train_test_split(val_test, test_size=0.5)

    # Merge with news data and save
    df_train = _merge_and_save(train_behaviors, news, os.path.join(output_path, f"{dataset_name}_train.csv"))
    df_val = _merge_and_save(val_behaviors, news, os.path.join(output_path, f"{dataset_name}_val.csv"))
    if dataset_name != "MIND-small":
        df_test = _merge_and_save(test_behaviors, news, os.path.join(output_path, f"{dataset_name}_test.csv"))

def _merge_and_save(behaviors, news, output_file):
    # Explode impressions and process
    behaviors = behaviors.explode("impressions")
    behaviors[["photo_id", "clicked"]] = behaviors["impressions"].str.split("-", expand=True)
    behaviors["clicked"] = behaviors["clicked"].astype(int)
    
    # Merge with news data
    df = pd.merge(behaviors, news, left_on="photo_id", right_on="news_id", how="left")
    df["play_rate"] = 0.0  # Placeholder for play_rate (not in MIND)
    df["pctr"] = df.groupby("photo_id")["clicked"].transform("mean")  # Historical CTR
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    return df