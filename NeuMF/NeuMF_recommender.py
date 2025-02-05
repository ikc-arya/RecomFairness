import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Define NeuMF model
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8):
        super(NeuMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        interaction = torch.cat([user_embedded, item_embedded], dim=-1)
        return self.mlp(interaction).squeeze()

class Evaluator:
    def __init__(self, all_items):
        self.all_items = set(all_items)  # Set of all item IDs in the dataset

    def hit_ratio(self, ranked_list, true_item):
        return int(true_item in ranked_list)

    def ndcg(self, ranked_list, true_item):
        if true_item in ranked_list:
            index = ranked_list.index(true_item)
            return 1 / np.log2(index + 2)
        return 0

    def coverage(self, all_ranked_lists, k):
        recommended_items = set(item for ranked_list in all_ranked_lists for item in ranked_list[:k])
        return len(recommended_items) / len(self.all_items)

    def novelty_coverage(self, all_ranked_lists, k):
        recommended_items = set(item for ranked_list in all_ranked_lists for item in ranked_list[:k])
        return len(recommended_items) / (k * len(all_ranked_lists))

    def evaluate(self, predictions, ground_truth, k_values=[5, 10]):
        metrics = {f'HR@{k}': [] for k in k_values}
        metrics.update({f'NDCG@{k}': [] for k in k_values})
        
        for k in k_values:
            for ranked_list, true_item in zip(predictions, ground_truth):
                metrics[f'HR@{k}'].append(self.hit_ratio(ranked_list[:k], true_item))
                metrics[f'NDCG@{k}'].append(self.ndcg(ranked_list[:k], true_item))

        results = {}
        for k in k_values:
            results[f'HR@{k}'] = np.mean(metrics[f'HR@{k}'])
            results[f'NDCG@{k}'] = np.mean(metrics[f'NDCG@{k}'])
            results[f'Cov@{k}'] = self.coverage(predictions, k)
            results[f'N_Cov@{k}'] = self.novelty_coverage(predictions, k)

        return results

if __name__ == "__main__":
    # Load preprocessed validation data
    try:
        val_data = pd.read_csv("preprocessedData/neumf_data.csv")
        print(val_data.head())  # Check if the data is loaded correctly
    except FileNotFoundError:
        print("Error: The file 'preprocessedData/neumf_data.csv' was not found.")
        exit()

    if val_data.empty:
        print("Error: The validation data is empty.")
        exit()

    users = val_data['UserId'].values
    items = val_data['ItemId'].values
    labels = val_data['Label'].values
    num_users = users.max() + 1
    num_items = items.max() + 1
    
    # Define evaluator
    evaluator = Evaluator(all_items=set(items))
    
    # Load pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(num_users, num_items).to(device)

    model.eval()

    predictions = []
    ground_truth = []
    with torch.no_grad():
        for user, item, label in zip(users, items, labels):
            user_tensor = torch.tensor([user], dtype=torch.long, device=device)
            item_tensor = torch.tensor([item], dtype=torch.long, device=device)
            pred = model(user_tensor, item_tensor).cpu().numpy().item()
            predictions.append((item, pred))
            ground_truth.append(item if label == 1 else 0)  # Use 0 for non-relevant items
    
    # Construct user_predictions
    user_predictions = {}
    for (user, (item, score)), true_item in zip(zip(users, predictions), ground_truth):
        if user not in user_predictions:
            user_predictions[user] = []
        user_predictions[user].append((item, score))
    
    print(f"User predictions constructed: {len(user_predictions)} users")

    # Sort and clean ground truth
    sorted_predictions = []
    cleaned_ground_truth = []
    user_ground_truth = {user: item if label == 1 else 0 for user, item, label in zip(users, items, labels)}

    for user, item_scores in user_predictions.items():
        item_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_predictions.append([item for item, _ in item_scores])
        cleaned_ground_truth.append(user_ground_truth[user])
    
    # Evaluate using the validation set
    results = evaluator.evaluate(sorted_predictions, cleaned_ground_truth)
    print("Evaluation Results:", results)

'''
Output:
   User predictions constructed: 94057 users
Evaluation Results: {
    'HR@5'      : 0.05436065364619327, 
    'NDCG@5'    : 0.03684368660691825, 
    'Cov@5'     : 0.20622721883096923, 
    'N_Cov@5'   : 0.009985434364268476, 
    'HR@10'     : 0.07335977120256866, 
    'NDCG@10'   : 0.04297634949031268, 
    'Cov@10'    : 0.3149180975802556, 
    'N_Cov@10'  : 0.007624100279617679
    }
'''