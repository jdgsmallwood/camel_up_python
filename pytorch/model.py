import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 


NUM_EPOCHS = 20_000
LEARNING_RATE = 0.1
DROPOUT = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
BATCH_SIZE = 64

SEED = 1337

torch.manual_seed(SEED)
torch.cuda.empty_cache()

# Step 1: Load the data from a CSV file
file_path = 'pytorch/data/camel_up_simulations.csv'  # Update with your CSV file path
data = pd.read_csv(file_path)
columns = data.columns
data = data.to_numpy()

group1 = data[:, :5]   # Columns 0 to 4
group2 = data[:, 5:10]  # Columns 5 to 9
group3 = data[:, 10:15] # Columns 10 to 14
group4 = data[:, 15:20] # Columns 10 to 14

# Step 4: Prepare an empty list to store permuted data
all_permuted = []
np.random.seed(SEED)
# Step 5: Generate 5 different permutations
for _ in range(5):
    # Generate a random permutation of 5 indices
      # Ensure reproducibility with different seeds
    permutation = np.random.permutation(5)
    
    # Apply the same permutation to each group
    group1_permuted = group1[:, permutation]
    group2_permuted = group2[:, permutation]
    group3_permuted = group3[:, permutation]
    group4_permuted = group4[:, permutation]
    
    # Concatenate the permuted groups back into a single array
    permuted = np.hstack((group1_permuted, group2_permuted, group3_permuted, group4_permuted))
    
    # Append the permuted data and the corresponding y values to the list
    all_permuted.append(permuted)
    

# Step 6: Combine all permuted data into a single array
data_new = np.vstack(all_permuted)
data = pd.DataFrame(data_new, columns = columns)

# Step 2: Assume the first 15 columns are inputs and the last 5 are outputs
X = data.iloc[:, :15].values  # Input features
y = data.iloc[:, 15:].values   # Output probabilities


# Step 3: Split the data into training and temp sets (80% train, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Step 4: Split the temp set into validation and test sets (50% of temp for each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# convert to tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
#train_dataset = TensorDataset(X_train_tensor, y_train_tensor)#

X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
#val_dataset = TensorDataset(X_val_tensor, y_val_tensor)


X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class Block(nn.Module):

    def __init__(self, input_num, output_num):
        super().__init__()
        self.linear = nn.Linear(input_num, output_num)
        self.dropout = nn.Dropout(DROPOUT)
        self.batch_norm = nn.BatchNorm1d(output_num)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class ProbabilityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding = nn.Embedding(17, 5)
        self.blocks = nn.ModuleList([
            Block(15 - 5 + 5 * 5, 256),
            Block(256,128),
            Block(128,64),
            Block(64,32),
        ])
        
        self.fc4 = nn.Linear(32, 5)   


    def forward(self, x, y=None):
        position1 = self.position_embedding(x[:, 1].long())
        position2 = self.position_embedding(x[:, 2].long())
        position3 = self.position_embedding(x[:, 3].long())
        position4 = self.position_embedding(x[:, 4].long())
        position0 = self.position_embedding(x[:, 0].long())

        x = torch.cat((position0, position1, position2, position3, position4, x[:,5:]), dim=1)
        
        for block in self.blocks:
            x = block(x)
        x = torch.softmax(self.fc4(x), dim=1) 
        
        if y is None:
            loss = None
        else:
            loss = criterion(x, y)

        return x, loss

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    DATA = {
        "train": (X_train_tensor, y_train_tensor),
        "val": (X_val_tensor, y_val_tensor),
        "test": (X_test_tensor, y_test_tensor),
    }

    x, y = DATA[split]
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = ProbabilityPredictor()
m = model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Print the shapes of the datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)



def save_checkpoint(model, optimizer, epoch, loss, file_path='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

checkpoint_path = 'model_checkpoint.pth'
final_model_path = 'final_trained_model.pth'
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()  
    
    optimizer.zero_grad()  
    predictions, loss = model(X_train_tensor, y_train_tensor)
    loss.backward()  
    optimizer.step()  
    
    if epoch % 100 == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            save_checkpoint(model, optimizer, epoch, best_val_loss, file_path=checkpoint_path)


model.eval()
predictions, loss = model(X_train_tensor, y_train_tensor)

print(predictions[1005, :])
print(y_train_tensor[1005, :])
print(loss)


final_model_path = 'final_trained_model.pth'

torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")