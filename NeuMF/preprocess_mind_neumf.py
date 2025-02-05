# The NeuMF implementation expects the data to be prepared as follows:
# - A tab-separated file with columns: UserId, ItemId, Label
# - Labels indicate implicit feedback (e.g., 1 for interaction, 0 for no interaction)
# - User and Item IDs should be mapped to contiguous integers.
# - The dataset should be split into training and testing sets.

import pandas as pd

class NeuMFPreprocessor:
    def __init__(self, train_behavior_path, train_news_path, val_behavior_path, val_news_path):
        self.train_behavior_path = train_behavior_path
        self.train_news_path = train_news_path
        self.val_behavior_path = val_behavior_path
        self.val_news_path = val_news_path

    def load_data(self):
        train_behaviors = pd.read_csv(self.train_behavior_path, sep='\t', header=None,
                                       names=['ImpressionId', 'UserId', 'Time', 'History', 'Impressions'])
        train_news = pd.read_csv(self.train_news_path, sep='\t', header=None,
                                  names=['NewsId', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
        val_behaviors = pd.read_csv(self.val_behavior_path, sep='\t', header=None,
                                     names=['ImpressionId', 'UserId', 'Time', 'History', 'Impressions'])
        val_news = pd.read_csv(self.val_news_path, sep='\t', header=None,
                                names=['NewsId', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
        return train_behaviors, train_news, val_behaviors, val_news

    def generate_interactions(self, behaviors):
        interactions = []
        for _, row in behaviors.iterrows():
            user_id = row['UserId']
            impressions = row['Impressions'].split(' ')
            for impression in impressions:
                news_id, label = impression.split('-')
                interactions.append((user_id, news_id, int(label)))
        return pd.DataFrame(interactions, columns=['UserId', 'ItemId', 'Label'])

    def encode_ids(self, df):
        user_mapping = {user: idx for idx, user in enumerate(df['UserId'].unique())}
        item_mapping = {item: idx for idx, item in enumerate(df['ItemId'].unique())}
        df['UserId'] = df['UserId'].map(user_mapping)
        df['ItemId'] = df['ItemId'].map(item_mapping)
        return df, len(user_mapping), len(item_mapping)

    def preprocess(self):
        train_behaviors, train_news, val_behaviors, val_news = self.load_data()
        all_data = self.generate_interactions(pd.concat([train_behaviors, val_behaviors]))
        all_data, num_users, num_items = self.encode_ids(all_data)
        all_data.to_csv("preprocessedData/neumf_data.csv", index=False)
        print("Preprocessing complete. Data file saved.")

if __name__ == "__main__":
    preprocessor = NeuMFPreprocessor("../TaFR/data/MIND/MINDsmall_train/behaviors.tsv", 
                                     "../TaFR/data/MIND/MINDsmall_train/news.tsv", 
                                     "../TaFR/data/MIND/MINDsmall_dev/behaviors.tsv", 
                                     "../TaFR/data/MIND/MINDsmall_dev/news.tsv")
    preprocessor.preprocess()



