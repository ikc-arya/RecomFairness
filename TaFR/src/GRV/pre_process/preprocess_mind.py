import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

def preprocess_mind(mind_path, output_path, dataset_name="MIND-small"):
    # Load behaviors.tsv and news.tsv
    print(f"Loading data from: {mind_path}")
    try:
        behaviors = pd.read_csv(
            os.path.join(mind_path, "behaviors.tsv"), 
            sep="\t", 
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"]
        )
        news = pd.read_csv(
            os.path.join(mind_path, "news.tsv"), 
            sep="\t", 
            header=None,
            names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        )
        print(f"Original behaviors shape: {behaviors.shape}, news shape: {news.shape}")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Apply 10-core filter
    user_counts = behaviors["user_id"].value_counts()
    behaviors = behaviors[behaviors["user_id"].isin(user_counts[user_counts >= 10].index)]
    valid_news = behaviors["impressions"].str.split("-").str[0].value_counts().loc[lambda x: x >= 10].index
    behaviors["impressions"] = behaviors["impressions"].apply(lambda x: [imp for imp in str(x).split() if imp.split("-")[0] in valid_news])
    print(f"After 10-core filter: {behaviors.shape}")

    # After 10-core filter, validate impressions
    behaviors["impressions"] = behaviors["impressions"].apply(
        lambda x: [imp for imp in str(x).split() if "-" in imp]
    )
    # Remove empty impression lists
    behaviors = behaviors[behaviors["impressions"].apply(len) > 0]

    # Split into train/val/test
    os.makedirs(output_path, exist_ok=True)
    if dataset_name == "MIND-small":
        train_behaviors = behaviors
        dev_path = mind_path.replace("MINDsmall_train", "MINDsmall_dev")
        val_behaviors = pd.read_csv(os.path.join(dev_path, "behaviors.tsv"), sep="\t", 
                                   names=["impression_id", "user_id", "time", "history", "impressions"])
        print(f"Validation shape: {val_behaviors.shape}")
    else:
        behaviors["timestamp"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        train_behaviors = behaviors[behaviors["timestamp"].dt.day <= 3]
        val_test = behaviors[behaviors["timestamp"].dt.day > 3]
        val_behaviors, test_behaviors = train_test_split(val_test, test_size=0.5)
        print(f"Train: {train_behaviors.shape}, Val: {val_behaviors.shape}, Test: {test_behaviors.shape}")

    # Process and save
    df_train = _merge_and_save(train_behaviors, news, os.path.join(output_path, f"{dataset_name}_train.csv"))
    df_val = _merge_and_save(val_behaviors, news, os.path.join(output_path, f"{dataset_name}_val.csv"))
    if dataset_name != "MIND-small":
        df_test = _merge_and_save(test_behaviors, news, os.path.join(output_path, f"{dataset_name}_test.csv"))
    print(f"Preprocessing complete! Files saved to {output_path}")

def _merge_and_save(behaviors, news, output_file):
    # Explode impressions and inspect
    behaviors = behaviors.explode("impressions").reset_index(drop=True)
    print("Sample impressions after explode:")
    print(behaviors["impressions"].head(10))

    # Check for malformed entries
    malformed = behaviors[~behaviors["impressions"].str.contains("-", na=False)]
    if not malformed.empty:
        print(f"Found {len(malformed)} malformed impressions:")
        print(malformed["impressions"].head())

    # Filter out malformed entries
    behaviors = behaviors[behaviors["impressions"].str.contains("-", na=False)]

    # Split into photo_id and clicked
    split_impressions = behaviors["impressions"].str.split("-", expand=True)
    behaviors["photo_id"] = split_impressions[0]
    behaviors["clicked"] = split_impressions[1].astype(int)

    # Merge with news data
    df = pd.merge(behaviors, news, left_on="photo_id", right_on="news_id", how="left")
    
    # Add placeholders for play_rate and pctr
    df["play_rate"] = 0.0
    df["pctr"] = df.groupby("photo_id")["clicked"].transform("mean")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    return df

# Example usage
preprocess_mind(
    mind_path="TaFR/data/MIND/MINDsmall_train",
    output_path="TaFR/data/MIND/preprocessed",
    dataset_name="MIND-small"
)