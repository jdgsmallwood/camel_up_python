import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np 


class CustomDataset(Dataset):
    def __init__(self, X, dice, stack, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.dice = torch.tensor(dice, dtype=torch.float32)
        self.stack = torch.tensor(stack, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.dice[idx], self.stack[idx], self.y[idx]

NUM_EPOCHS = 200_000
LEARNING_RATE = 0.0001
DROPOUT = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 100
BATCH_SIZE = 4096

SEED = 1337

torch.manual_seed(SEED)
torch.cuda.empty_cache()

file_path = 'pytorch/data/camel_up_simulations.csv'
data = pd.read_csv(file_path)
columns = data.columns
data = data.to_numpy()

group1 = data[:, :5]   
group2 = data[:, 5:10] 
group3 = data[:, 10:15]
group4 = data[:, 15:20]

all_permuted = []
np.random.seed(SEED)
for _ in range(5):
    permutation = np.random.permutation(5)
    
    # Apply the same permutation to each group
    group1_permuted = group1[:, permutation]
    group2_permuted = group2[:, permutation]
    group3_permuted = group3[:, permutation]
    group4_permuted = group4[:, permutation]
    
    permuted = np.hstack((group1_permuted, group2_permuted, group3_permuted, group4_permuted))
    all_permuted.append(permuted)
    

data_new = np.vstack(all_permuted)
data = pd.DataFrame(data_new, columns = columns)
data = pd.get_dummies(data, columns=data.columns[:5])  # positions
#data = pd.get_dummies(data, columns=data.columns[5:10])  # stack

dice = data.iloc[:, 0:5].values
stack = data.iloc[:, 5:10:].values
#stack = stack.reshape(stack.shape[0], 5, 6)
y = data.iloc[:, 10:15].values
positions = data.iloc[:, 15:].values
positions = positions.reshape(positions.shape[0], 5, 17)


X_train, X_temp, dice_train, dice_temp, stack_train, stack_temp, y_train, y_temp = train_test_split(
    positions, dice, stack, y, test_size=0.2, random_state=SEED
)
X_val, X_test, dice_val, dice_test, stack_val, stack_test, y_val, y_test = train_test_split(
    X_temp, dice_temp, stack_temp, y_temp, test_size=0.5, random_state=SEED
)

train_dataset = CustomDataset(X_train, dice_train, stack_train, y_train)
val_dataset = CustomDataset(X_val, dice_val, stack_val, y_val)
test_dataset = CustomDataset(X_test, dice_test, stack_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


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

        # CNN for 5x17 matrix input
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # CNN for 5x6 matrix input
        # self.cnn2 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 2), padding=(1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 2), padding=(1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(16),
        #     nn.Flatten()
        # )
        self.dense_stack = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.dense1 = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(640, 1024), 
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 5)  # Output for 5-way softmax
        )
    
    def forward(self, positions, dice, stack, y=None):
        
        positions = positions.unsqueeze(1)  # (B, 1, 5, 17)
        cnn1_out = self.cnn1(positions)
        
        # stack = stack.unsqueeze(1)   # (B, 1, 5, 6)
        # cnn2_out = self.cnn2(stack)

        stack = self.dense_stack(stack)
        
        dense1_out = self.dense1(dice)  # (B, 1, 5)
        
        combined = torch.cat([cnn1_out, stack, dense1_out], dim=1)
        
        output = self.fc(combined)
        output = F.softmax(output, dim=1)
        
        if y is None:
            loss = None
        else:
            loss = criterion(output, y)

        return output, loss


def get_batch(split):
    DATA = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    x,dice, stack, y = next(iter(DATA[split]))
    x, dice, stack, y = x.to(device), dice.to(device), stack.to(device), y.to(device)
    return x, dice, stack, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, dice, stack, Y = get_batch(split)
            _, loss = model(X, dice, stack, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = ProbabilityPredictor()
m = model.to(device)

#criterion = nn.CrossEntropyLoss() 
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

checkpoint_path = 'model_checkpoint_cnn.pth'
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()  
    positions_train, dice_train, stack_train, y = get_batch('train')
    optimizer.zero_grad()  
    predictions, loss = model(positions_train, dice_train, stack_train, y)
    loss.backward()  
    optimizer.step()  
    
    if epoch % 1000 == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            save_checkpoint(model, optimizer, epoch, best_val_loss, file_path=checkpoint_path)


model.eval()
predictions, loss = model(positions_train, dice_train, stack_train, y)

print(F.softmax(predictions[25, :]))
print(y[25, :])
print(loss)


final_model_path = 'final_trained_model_cnn.pth'

torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")