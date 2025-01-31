import pandas as pd
import os
from datetime import datetime

def preprocess_mind(mind_path, output_path):
    # Load behaviors.tsv and news.tsv
    behaviors = pd.read_csv(os.path.join(mind_path, "behaviors.tsv"), sep="\t", header=None, 
                            names=["impression_id", "user_id", "time", "history", "impressions"])
    news = pd.read_csv(os.path.join(mind_path, "news.tsv"), sep="\t", header=None,
                       names=["news_id", "category", "subcategory", "title", "abstract", "url", 
                              "title_entities", "abstract_entities"])

    # Split impressions into individual rows
    behaviors["impressions"] = behaviors["impressions"].str.split(" ")
    behaviors = behaviors.explode("impressions").reset_index(drop=True)
    behaviors[["photo_id", "clicked"]] = behaviors["impressions"].str.split("-", expand=True)
    behaviors["clicked"] = behaviors["clicked"].astype(int)

    # Convert time to timestamp (relative hours)
    min_time = pd.to_datetime(behaviors["time"].min(), format="%m/%d/%Y %I:%M:%S %p")
    behaviors["timestamp"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
    behaviors["timelevel"] = ((behaviors["timestamp"] - min_time).dt.total_seconds() / 3600).astype(int)

    # Merge with news data
    df = pd.merge(behaviors, news, left_on="photo_id", right_on="news_id", how="left")

    # Create required columns (placeholders for play_rate and pctr)
    df["play_rate"] = 0.0  # Placeholder (MIND lacks playtime data)
    df["pctr"] = df.groupby("photo_id")["clicked"].transform("mean")  # Historical CTR as pCTR

    # Rename columns to match codebase
    df.rename(columns={"clicked": "click_rate"}, inplace=True)

    # Save preprocessed data
    output_file = os.path.join(output_path, "MIND_preprocessed.csv")
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# Example usage
preprocess_mind(
    mind_path="TaFR/data/MIND/MINDsmall_train",
    output_path="TaFR/data/MIND/preprocessed"
)