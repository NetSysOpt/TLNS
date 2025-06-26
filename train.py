import os
import torch
import torch_geometric
import random
import time
import numpy as np
import argparse
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import DotProductSimilarity

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--instance', type=str, default='CA')
parser.add_argument('--batchsize', type=int, default=5)
parser.add_argument('--loss', type=str, default='IL')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--architecture', type=str, default='SGT')
parser.add_argument('--radius', type=int, default=75)
args = parser.parse_args()

TaskName = args.instance

weight_norm = args.weightnorm
loss_function = args.loss
architecture = args.architecture

# set folder
train_task = f'{TaskName}_{loss_function}_{architecture}_train'
if not os.path.isdir(f'./train_logs'):
    os.mkdir(f'./train_logs')
if not os.path.isdir(f'./train_logs/{train_task}'):
    os.mkdir(f'./train_logs/{train_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain')
if not os.path.isdir(f'./pretrain/{train_task}'):
    os.mkdir(f'./pretrain/{train_task}')
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}.log', 'wb')

# set params
LEARNING_RATE = 0.001
NB_EPOCHS = 9999
BATCH_SIZE = args.batchsize
NUM_WORKERS = 0
WEIGHT_NORM = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR_BG = f'./dataset/{TaskName}/r{args.radius}/BG'
DIR_SOL = f'./dataset/{TaskName}/r{args.radius}/solution'
sample_names = os.listdir(DIR_BG)
sample_files = [(os.path.join(DIR_BG, name), os.path.join(DIR_SOL, name).replace('bg', 'sol')) for name in sample_names]

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

random.shuffle(sample_files)
train_files = sample_files[: int(0.90 * len(sample_files))]
valid_files = sample_files[int(0.90 * len(sample_files)):]

if architecture == 'GCN':
    from GCN import GNNPolicy
    PredictModel = GNNPolicy().to(DEVICE)
elif architecture == 'GAT':
    from GCN import GATPolicy
    PredictModel = GATPolicy().to(DEVICE)
elif architecture == 'SGT':
    from GCN import SGT
    PredictModel = SGT(dropout=args.dropout, alpha=args.alpha, batch_size=BATCH_SIZE).to(DEVICE)
from GCN import GraphDataset

train_data = GraphDataset(train_files)
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=NUM_WORKERS, drop_last=True)
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUM_WORKERS, drop_last=True)





def train(model, data_loader, loss_type='CL', optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        if loss_type == 'CL':
            for step, batch in enumerate(data_loader):
                batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
                policy = model(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features,
                    batch.ncons
                )
                n_samples = len(batch)
                policy = policy.sigmoid()

                temperature = 0.07
                infoNCE_loss_function = losses.NTXentLoss(temperature=temperature,
                                                          distance=DotProductSimilarity()).to(DEVICE)
                anchor_positive = []
                anchor_negative = []
                positive_idx = []
                negative_idx = []
                total_sample = n_samples

                embeddings = torch.reshape(policy, (n_samples, int(policy.shape[0] / n_samples)))

                for i in range(n_samples):
                    sol = batch.sols[i]
                    p = [sol[idx][0] for idx in range(len(sol)) if sol[idx][1] > 0.6 * sol[0][1]]
                    n = [sol[idx][0] for idx in range(len(sol)) if sol[idx][1] < 0.1 * sol[0][1]]
                    for j in range(len(p)):
                        anchor_positive.append(i)
                        positive_idx.append(total_sample)
                        embeddings = torch.cat(
                            [embeddings, torch.tensor([p[j]]).to(DEVICE)])
                        total_sample += 1
                    for j in range(len(n)):
                        anchor_negative.append(i)
                        negative_idx.append(total_sample)
                        embeddings = torch.cat(
                            [embeddings, torch.tensor([n[j]]).to(DEVICE)])
                        total_sample += 1
                indices = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE),
                           torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))

                loss = infoNCE_loss_function(embeddings, indices_tuple=indices) * n_samples

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                mean_loss += loss.item()
                n_samples_processed += batch.num_graphs
        elif loss_type == 'IL':
            for step, batch in enumerate(data_loader):
                policy = model(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features,
                    batch.ncons
                )

                sol = [batch.sols[j][0][0] for j in range(len(batch))]
                sol = torch.FloatTensor(sol).view(-1).to(DEVICE)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1).cuda())
                loss = criterion(policy, sol) * len(sol)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                mean_loss += loss.item()
                n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed

    return mean_loss


optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)

best_val_loss = 99999
for epoch in range(NB_EPOCHS):

    begin = time.time()
    train_loss = train(PredictModel, train_loader, loss_function, optimizer, weight_norm)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
    valid_loss = train(PredictModel, valid_loader, loss_function, None, weight_norm)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}, time: {time.time() - begin:.3f}")
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(PredictModel.state_dict(), model_save_path + f'{loss_function}_{architecture}_best.pth')
    torch.save(PredictModel.state_dict(), model_save_path + f'{loss_function}_{architecture}_last.pth')
    st = f'@epoch{epoch}   Train loss:{train_loss}   Valid loss:{valid_loss}    TIME:{time.time() - begin}\n'
    log_file.write(st.encode())
    log_file.flush()
print('done')





