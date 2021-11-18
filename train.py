import pickle
import torch
import torch.nn as nn
import numpy as np
import math
import time
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from dataprocess import EvictionrateRegDataset, EvictionrateNodeDataset, EvictionrateClusterDataset, collate_fn_reg, collate_fn_node, collate_fn_cluster
from model import LR, LSTM_M, STTransformer
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_squared_error
from evaluation import test_cluster_reg, test_cluster_lr, test_cluster_ml, test_node, test_node_reg


def remove_feature(node_t_batch, nodes_st_batch, predict_window):
    index_list = np.array([[11, 12, 13], [17, 18, 19], [23, 24, 25]])
    for t in range(1, predict_window+1):
        remove_indexes = list(np.concatenate(index_list[:, -t:]))
        node_t_batch[:, -t, remove_indexes] = 0
        nodes_st_batch[:, :, -t, remove_indexes] = 0
    return node_t_batch, nodes_st_batch


def evaluate_node(eval_model, valid_loader, predict_window, hour):
    total_loss = 0.
    outputs = []
    targets_s = []
    with torch.no_grad():
        for batch_index, batch_data in enumerate(valid_loader):
            (node_t_batch, nodes_st_batch) = batch_data
            targets = node_t_batch[:, -1, -predict_window:].clone()
            # remove predicted feature of last predict_window timesteps
            node_t_batch, nodes_st_batch = remove_feature(
                node_t_batch, nodes_st_batch, predict_window)
            node_t_batch = node_t_batch.view(node_t_batch.shape[0], -1)
            output = eval_model.predict(node_t_batch.detach().cpu().numpy())
            outputs.append(output[0])
            # print(output[0],targets.detach().cpu().numpy()[0])
            targets_s.append(targets.detach().cpu().numpy()[0][hour])
        total_loss = mean_squared_error(outputs, targets_s)
    return total_loss


def train_node_reg(train_data_path, time_window=48, predict_window=3,
                   batch_size=1, seed=0, valid_split=0.2, shuffle_dataset=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EvictionrateRegDataset(
        processed_data_path=train_data_path, time_window=time_window)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split*dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=collate_fn_reg)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=collate_fn_reg)
    tot_train_feature = []
    tot_train_feature_targets = []
    criterion = nn.MSELoss()
    print(len(train_loader))
    for batch_index, batch_data in enumerate(train_loader):
        (node_t_batch, nodes_st_batch) = batch_data
        targets = node_t_batch[:, -1, -predict_window:].clone()
        # remove predicted feature of last predict_window timesteps
        node_t_batch, nodes_st_batch = remove_feature(
            node_t_batch, nodes_st_batch, predict_window)
        node_t_batch = node_t_batch.view(node_t_batch.shape[0], -1)
        tot_train_feature.append(node_t_batch)
        tot_train_feature_targets.append(targets)
    print('Done prepration')
    tot_train_feature = torch.cat(
        tot_train_feature, dim=0).detach().cpu().numpy()
    tot_train_feature_targets = torch.cat(
        tot_train_feature_targets, dim=0).detach().cpu().numpy()

    for t in range(predict_window):
        print(f'processing time {t}')
        gbdt_model_save_path = 'save/'+'GBDT_node_{}h_0.pth'.format(t)
        gbdt = GradientBoostingRegressor(random_state=seed)
        gbdt.fit(tot_train_feature, tot_train_feature_targets[:, t])
        val_loss_gbdt = 0.
        for batch_data in validation_loader:
            val_loss_gbdt += test_node_reg(gbdt, batch_data,
                                           criterion, device, t)
        # val_loss_gbdt = evaluate_node(gbdt, validation_loader, predict_window, t)
        print(f'{t}h val_loss gbdt {val_loss_gbdt}')
        pickle.dump(gbdt, open(gbdt_model_save_path, 'wb'))

        rf_model_save_path = 'save/'+'RF_node_{}h_0.pth'.format(t)
        rf = RandomForestRegressor(max_depth=2, random_state=seed)
        rf.fit(tot_train_feature, tot_train_feature_targets[:, t])
        val_loss_rf = 0.
        for batch_data in validation_loader:
            val_loss_rf += test_node_reg(rf, batch_data,
                                         criterion, device, t)
        # val_loss_rf = evaluate_node(rf, validation_loader, predict_window, t)
        print(f'{t}h val_loss rf {val_loss_rf}')
        pickle.dump(rf, open(rf_model_save_path, 'wb'))

        svr_model_save_path = 'save/'+'SVR_node_{}h_0.pth'.format(t)
        svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        svr.fit(tot_train_feature, tot_train_feature_targets[:, t])
        val_loss_svr = 0.
        for batch_data in validation_loader:
            val_loss_svr += test_node_reg(svr, batch_data,
                                          criterion, device, t)
        # val_loss_svr = evaluate_node(svr, validation_loader, predict_window, t)
        print(f'{t}h val_loss svr {val_loss_svr}')
        pickle.dump(svr, open(svr_model_save_path, 'wb'))


def train_node_ml(train_data_path, model_name='LR', time_window=48, predict_window=3,
                  batch_size=16, epochs=100, lr=1e-4, seed=0, valid_split=0.2, shuffle_dataset=False):
    model_save_path = 'save/'+'{}_node_{}.pth'.format(model_name, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = EvictionrateNodeDataset(
        processed_data_path=train_data_path, time_window=time_window)
    print('dataset class done')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split*dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=collate_fn_node)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=collate_fn_node)
    criterion = nn.MSELoss()
    node_dim = dataset.get_node_dim()
    feature_dim = dataset.get_feature_dim()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if model_name == 'STTransformer':
        model = STTransformer(node_dim=node_dim, feature_dim=feature_dim,
                              time_window=time_window, predicted_len=predict_window).to(device)
    elif model_name == 'LR':
        model = LR(feature_dim=feature_dim,
                   time_window=time_window, predicted_len=predict_window).to(device)
    elif model_name == 'LSTM':
        model = LSTM_M(feature_dim=feature_dim,
                       time_window=time_window, predicted_len=predict_window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.
        start_time = time.time()

        for batch_index, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            (nodes_st_batch, nodes_count_batch,
             spot_count_batch, cluster_target_batch) = batch_data
            batch_size = spot_count_batch.shape[0]
            predicted_eviction_batch = []
            for ibatch in range(batch_size):
                predict_rates = []
                spotNode_count = int(nodes_count_batch[ibatch].cpu().item())
                for node_idx in range(spotNode_count):
                    if model_name == 'STTransformer':
                        node_t = nodes_st_batch[ibatch,
                                                node_idx, :, :].unsqueeze(0)
                        nodes_st = nodes_st_batch[ibatch].unsqueeze(0)
                        output = model(node_t, nodes_st)
                        predict_rates.append(output)
                    elif model_name == 'LR':
                        node_t = nodes_st_batch[ibatch, node_idx, :, :].view(
                            nodes_st_batch.shape[0], -1)
                        output = model(node_t)
                        predict_rates.append(output[0])
                    elif model_name == 'LSTM':
                        node_t = nodes_st_batch[ibatch,
                                                node_idx, :, :].unsqueeze(0)
                        output = model(node_t)
                        predict_rates.append(output[0])
                predict_rates = torch.stack(
                    predict_rates).squeeze().permute(1, 0)  # 3*spot_node_dim
                predicted_eviction_rates = []
                spot_sum = spot_count_batch[ibatch].sum()
                for t in range(predict_window):
                    evicted_count = spot_count_batch[ibatch,
                                                     :spotNode_count]*predict_rates[t]
                    eviction_rate = torch.div(evicted_count.sum(), spot_sum)
                    predicted_eviction_rates.append(eviction_rate)
                predicted_eviction_rates = torch.as_tensor(
                    np.array(predicted_eviction_rates))  # 3*1
                predicted_eviction_batch.append(predicted_eviction_rates)
            predicted_eviction_batch = torch.stack(
                predicted_eviction_batch).to(device)  # B*3
            loss = criterion(predicted_eviction_batch, cluster_target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(train_loader) / (5*batch_size))
            if batch_index % log_interval == 0 and batch_index > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | {:5.2f} ms | '
                      'loss {:5.5f} | ppl {:8.2f}'.format(
                          epoch, batch_index, len(
                              train_loader), scheduler.get_last_lr()[0],
                          elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

        if(epoch % 5 == 0):
            val_loss = np.mean(test_node(
                model, model_name, validation_loader, predict_window, criterion, device))
            torch.save(model.state_dict(), model_save_path)
            model.train()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                                                                          val_loss, math.exp(val_loss)))
            print('-' * 89)

        scheduler.step()


def train_cluster_reg(train_data_path, time_window=48, predict_window=3,
                      batch_size=1, seed=0, valid_split=0.2, shuffle_dataset=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = EvictionrateClusterDataset(
        processed_data_path=train_data_path, time_window=time_window)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split*dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=collate_fn_cluster)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=collate_fn_cluster)
    tot_train_feature = []
    tot_train_feature_targets = []

    for batch_index, batch_data in enumerate(train_loader):
        (cluster_t_batch, target_batch) = batch_data
        cluster_t_batch = cluster_t_batch.view(cluster_t_batch.shape[0], -1)
        tot_train_feature.append(cluster_t_batch)
        tot_train_feature_targets.append(target_batch)
        print(f'the batch index is {batch_index}')

    tot_train_feature = torch.cat(
        tot_train_feature, dim=0).detach().cpu().numpy()
    tot_train_feature_targets = torch.cat(
        tot_train_feature_targets, dim=0).detach().cpu().numpy()

    for t in range(predict_window):
        gbdt_model_save_path = 'save/'+'GBDT_cluster_{}h_0.pth'.format(t)
        gbdt = GradientBoostingRegressor(random_state=seed)
        gbdt.fit(tot_train_feature, tot_train_feature_targets[:, t])
        val_loss_gbdt = test_cluster_reg(gbdt, validation_loader, t)
        print(f'{t}h val_loss gbdt {val_loss_gbdt}')
        pickle.dump(gbdt, open(gbdt_model_save_path, 'wb'))

        rf_model_save_path = 'save/'+'RF_cluster_{}h_0.pth'.format(t)
        rf = RandomForestRegressor(max_depth=2, random_state=seed)
        rf.fit(tot_train_feature, tot_train_feature_targets[:, t])
        val_loss_rf = test_cluster_reg(rf, validation_loader, t)
        print(f'{t}h val_loss rf {val_loss_rf}')
        pickle.dump(rf, open(rf_model_save_path, 'wb'))

        svr_model_save_path = 'save/'+'SVR_cluster_{}h_0.pth'.format(t)
        svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        svr.fit(tot_train_feature, tot_train_feature_targets[:, t])
        val_loss_svr = test_cluster_reg(svr, validation_loader, t)
        print(f'{t}h val_loss svr {val_loss_svr}')
        pickle.dump(svr, open(svr_model_save_path, 'wb'))


def train_cluster_ml(train_data_path, model_name='LR', time_window=48, predict_window=3,
                     batch_size=32, epochs=300, lr=0.01, seed=0, valid_split=0.2, shuffle_dataset=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_save_path = 'save/'+'{}_cluster_{}.pth'.format(model_name, seed)
    dataset = EvictionrateClusterDataset(
        processed_data_path=train_data_path, time_window=time_window)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split*dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=collate_fn_cluster)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=collate_fn_cluster)
    feature_dim = dataset.get_feature_dim()

    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'LR':
        model = LR(feature_dim=feature_dim,
                   time_window=time_window, predicted_len=predict_window).to(device)
    elif model_name == 'LSTM':
        model = LSTM_M(feature_dim=feature_dim,
                       time_window=time_window, predicted_len=predict_window).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.f1.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.
        start_time = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            (cluster_t_batch, target_batch) = batch_data
            if model_name == 'LR':
                cluster_t_batch = cluster_t_batch.view(
                    cluster_t_batch.shape[0], -1)
            elif model_name == 'LSTM':
                cluster_t_batch = cluster_t_batch.view(
                    cluster_t_batch.shape[0], time_window, -1)
            targets = target_batch
            output = model(cluster_t_batch)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            log_interval = int(len(train_loader) / 5)
            if batch_index % log_interval == 0 and batch_index > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | {:5.2f} ms  '
                      'loss {:5.5f} '.format(
                          epoch, batch_index, len(
                              train_loader), scheduler.get_last_lr()[0],
                          elapsed * 1000 / log_interval,
                          cur_loss, ))
                total_loss = 0
                start_time = time.time()

        if(epoch % 5 == 0):
            if model_name == 'LR':
                val_loss = test_cluster_lr(model, validation_loader, criterion)
            elif model_name == 'LSTM':
                val_loss = test_cluster_ml(
                    model, validation_loader, time_window, criterion)
            torch.save(model.state_dict(), model_save_path)
            model.train()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} '.format(epoch, (time.time() - epoch_start_time),
                                                                                       val_loss, ))
            print('-' * 89)

        scheduler.step()
