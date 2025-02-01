# The GRU4Rec implementation expects the data in the following format:
# A tab-separated file (.tsv).
# Columns:
# SessionId: A unique identifier for each session.
# ItemId: A unique identifier for each item (e.g., news article).
# Time: A timestamp for the interaction.

import pandas as pd

train_behaviors_path = "/Users/melodiediana/Desktop/TU Wien/ExpDesign/GRU4Rec_PyTorch_Official-master/MINDsmall_train/behaviors.tsv"
dev_behaviors_path = "/Users/melodiediana/Desktop/TU Wien/ExpDesign/GRU4Rec_PyTorch_Official-master/MINDsmall_dev/behaviors.tsv"

# Load file
train_behaviors = pd.read_csv(train_behaviors_path, sep='\t', header=None, names=['ImpressionId', 'UserId', 'Time', 'History', 'Impressions'])

# Parse interactions for training
train_sessions = []
for idx, row in train_behaviors.iterrows():
    session_id = idx  # Use row index as session ID
    timestamp = row['Time']  # Timestamp of interaction
    items = row['Impressions'].split(' ')  # Split impressions into individual items
    
    for item in items:
        train_sessions.append([session_id, item, timestamp])

# Convert to DataFrame
train_sessions_df = pd.DataFrame(train_sessions, columns=['SessionId', 'ItemId', 'Time'])

# Convert timestamp to Unix format
train_sessions_df['Time'] = pd.to_datetime(train_sessions_df['Time'])
train_sessions_df['Time'] = train_sessions_df['Time'].apply(lambda x: int(x.timestamp()))

# Save preprocessed train data
train_output_path = "/Users/melodiediana/Desktop/TU Wien/ExpDesign/GRU4Rec_PyTorch_Official-master/preprocessed_train.tsv"
train_sessions_df.to_csv(train_output_path, sep='\t', index=False)
print(f"Preprocessed train data saved to {train_output_path}")

# Repeat for dev behaviors file
dev_behaviors = pd.read_csv(dev_behaviors_path, sep='\t', header=None, names=['ImpressionId', 'UserId', 'Time', 'History', 'Impressions'])

dev_sessions = []
for idx, row in dev_behaviors.iterrows():
    session_id = idx
    timestamp = row['Time']
    items = row['Impressions'].split(' ')
    
    for item in items:
        dev_sessions.append([session_id, item, timestamp])

dev_sessions_df = pd.DataFrame(dev_sessions, columns=['SessionId', 'ItemId', 'Time'])
dev_sessions_df['Time'] = pd.to_datetime(dev_sessions_df['Time'])
dev_sessions_df['Time'] = dev_sessions_df['Time'].apply(lambda x: int(x.timestamp()))

dev_output_path = "/Users/melodiediana/Desktop/TU Wien/ExpDesign/GRU4Rec_PyTorch_Official-master/preprocessed_dev.tsv"
dev_sessions_df.to_csv(dev_output_path, sep='\t', index=False)
print(f"Preprocessed dev data saved to {dev_output_path}")

