import torch
import torch.nn as nn
import numpy as np
import math
import pickle
from model import STTransformer, LR, LSTM_M
from dataprocess import EvictionrateClusterDataset, EvictionrateNodeDataset, collate_fn_cluster, collate_fn_node
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_squared_error


def test_node(eval_model, model_name, batch_data, predict_window, criterion, device):
    eval_model.eval()  # Turn on the evaluation mode
    loss = np.zeros((predict_window,))
    # spot_count_batch: B*N
    with torch.no_grad():
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
                    output = eval_model(node_t, nodes_st)
                    predict_rates.append(output)
                elif model_name == 'LR':
                    node_t = nodes_st_batch[ibatch, node_idx, :, :].view(
                        nodes_st_batch.shape[0], -1)
                    output = eval_model(node_t)
                    predict_rates.append(output[0])
                elif model_name == 'LSTM':
                    node_t = nodes_st_batch[ibatch,
                                            node_idx, :, :].unsqueeze(0)
                    output = eval_model(node_t)
                    predict_rates.append(output[0])
            predict_rates = torch.stack(
                predict_rates).squeeze().permute(1, 0)
            predicted_eviction_rates = []
            spot_sum = spot_count_batch[ibatch].sum().cpu().item()
            for t in range(predict_window):
                evicted_count = spot_count_batch[ibatch,
                                                 :spotNode_count]*predict_rates[t]
                eviction_rate = evicted_count.sum().cpu().item()/spot_sum
                predicted_eviction_rates.append(eviction_rate)
            predicted_eviction_rates = torch.as_tensor(
                np.array(predicted_eviction_rates))
            predicted_eviction_batch.append(predicted_eviction_rates)
        predicted_eviction_batch = torch.stack(
            predicted_eviction_batch).to(device)
        for t in range(predict_window):
            loss[t] = criterion(predicted_eviction_batch[:, t],
                                cluster_target_batch[:, t]).cpu().item()
        return loss


def test_cluster_reg(eval_model, test_loader, hour):
    total_loss = 0.
    outputs = []
    targets_s = []
    for batch_index, batch_data in enumerate(test_loader):
        (cluster_t_batch, target_batch) = batch_data
        if target_batch.detach().cpu().numpy()[0][hour] == -1:
            continue
        cluster_t_batch = cluster_t_batch.view(cluster_t_batch.shape[0], -1)
        output = eval_model.predict(cluster_t_batch.detach().cpu().numpy())
        outputs.append(output[0])
        targets_s.append(target_batch.detach().cpu().numpy()[0][hour])
    total_loss = mean_squared_error(outputs, targets_s, squared=False)
    return total_loss


def test_cluster_ml(eval_model, test_loader, time_window, criterion):
    eval_model.eval()
    loss = 0.
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_index, batch_data in enumerate(test_loader):
            (cluster_t_batch, target_batch) = batch_data
            cluster_t_batch = cluster_t_batch.view(
                cluster_t_batch.shape[0], time_window, -1)
            output = eval_model(cluster_t_batch)
            loss += criterion(output, target_batch).cpu().item()
    loss = np.sqrt(loss/len(test_loader))
    return loss


def test_cluster_lr(eval_model, test_loader, criterion):
    eval_model.eval()
    loss = 0.
    with torch.no_grad():
        for batch_index, batch_data in enumerate(test_loader):
            (cluster_t_batch, target_batch) = batch_data
            cluster_t_batch = cluster_t_batch.view(
                cluster_t_batch.shape[0], -1)
            output = eval_model(cluster_t_batch)
            loss += criterion(output, target_batch).cpu().item()
    loss = np.sqrt(loss/len(test_loader))
    return loss


def test_node_reg(eval_model, batch_data, criterion, device, t):
    (nodes_st_batch, nodes_count_batch,
     spot_count_batch, cluster_target_batch) = batch_data
    batch_size = spot_count_batch.shape[0]
    predicted_eviction_batch = []
    for ibatch in range(batch_size):
        predicted_eviction_rates = []
        spotNode_count = int(nodes_count_batch[ibatch].cpu().item())
        evicted_count = 0
        spot_sum = spot_count_batch[ibatch].sum().cpu().item()
        for node_idx in range(spotNode_count):
            node_t = nodes_st_batch[ibatch, node_idx, :, :].unsqueeze(0)
            node_t = node_t.view(node_t.shape[0], -1)
            output = eval_model.predict(node_t.detach().cpu().numpy())
            evicted_count += output[0] * \
                spot_count_batch[ibatch, node_idx].cpu().numpy()
        predict_rate = evicted_count/spot_sum
        predicted_eviction_rates.append(predict_rate)
        predicted_eviction_rates = torch.as_tensor(
            np.array(predicted_eviction_rates))  # 3*1
        predicted_eviction_batch.append(predicted_eviction_rates)
    predicted_eviction_batch = torch.as_tensor(
        np.array(predicted_eviction_batch)).to(device)  # B*3
    loss = criterion(predicted_eviction_batch,
                     cluster_target_batch[:, t]).cpu().item()
    # print(loss)
    return loss


def evaluate_cluster_baselines(baseline_cluster_path, prediction_hours=3,
                               batch_size=1, time_window=48, seed=0):
    # 'test/baseline_cluster.pkl'
    cluster_model_loss = {}
    torch.manual_seed(seed)
    np.random.seed(seed)
    reg_model_list = ['GBDT', 'RF', 'SVR']
    ml_model_list = ['LR', 'LSTM']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EvictionrateClusterDataset(
        processed_data_path=baseline_cluster_path, time_window=prediction_hours)
    dataset_size = len(dataset)
    feature_dim = dataset.get_feature_dim()
    indices = list(range(dataset_size))
    test_sampler = SubsetRandomSampler(indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler, collate_fn=collate_fn_cluster)
    for model_name in reg_model_list:
        tmp_result_list = []
        for t in range(prediction_hours):
            path = 'save/'+model_name+'_cluster_{}h_0.pth'.format(t)
            load_model = pickle.load(open(path, 'rb'))
            loss = test_cluster_reg(load_model, test_loader, t)
            print(f'{model_name} with predict {t} hour has loss {loss}')
            tmp_result_list.append(loss)
        cluster_model_loss[model_name] = tmp_result_list

    criterion = nn.MSELoss()
    for model_name in ml_model_list:
        model = None
        if model_name == 'LSTM':
            model = LSTM_M(feature_dim=feature_dim,
                           time_window=time_window, predicted_len=prediction_hours).to(device)
            model.load_state_dict(torch.load(
                'save/{}_cluster_0.pth'.format(model_name)))
            loss = test_cluster_ml(
                model, test_loader, prediction_hours, criterion)
        elif model_name == 'LR':
            model = LR(feature_dim=feature_dim,
                       time_window=time_window, predicted_len=prediction_hours).to(device)
            model.load_state_dict(torch.load(
                'save/{}_cluster_0.pth'.format(model_name)))
            loss = test_cluster_lr(model, test_loader, criterion)
        cluster_model_loss[model_name] = loss
        print(f'{model_name} has loss {loss}')
    return cluster_model_loss


def evaluate_node_baselines(baseline_node_path, node_dim=440, prediction_hours=3,
                            batch_size=1, time_window=48, seed=0):
    # 'test/processed_test_data.pkl'
    torch.manual_seed(seed)
    np.random.seed(seed)
    node_model_loss = {}
    reg_model_list = ['GBDT', 'RF', 'SVR']
    ml_model_list = ['LR', 'LSTM', 'STTransformer']
    dataset = EvictionrateNodeDataset(
        processed_nodes_data_path=baseline_node_path, time_window=prediction_hours)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_sampler = SubsetRandomSampler(indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler, collate_fn=collate_fn_node)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    feature_dim = dataset.get_feature_dim()
    # node_dim = dataset.get_node_dim()

    for model_name in reg_model_list:
        tmp_result_list = []
        for t in range(prediction_hours):
            path = 'save/'+model_name+'_node_{}h_0.pth'.format(t)
            load_model = pickle.load(open(path, 'rb'))
            loss = 0
            for batch_index, batch_data in enumerate(test_loader):
                loss += test_node_reg(load_model, batch_data,
                                      criterion, device, t)
            loss = math.sqrt(loss/len(test_loader))
            tmp_result_list.append(loss)
            print(f'{model_name} with predict {t} hour has loss {loss}')
        node_model_loss[model_name] = tmp_result_list

    for model_name in ml_model_list:
        model = None
        loss = np.zeros((prediction_hours,))
        if model_name == 'LSTM':
            model = LSTM_M(feature_dim=feature_dim,
                           time_window=time_window, predicted_len=prediction_hours).to(device)
            model.load_state_dict(torch.load(
                'save/{}_node_0.pth'.format(model_name)))
            for batch_index, batch_data in enumerate(test_loader):
                loss_ml = test_node(
                    model, model_name, batch_data, prediction_hours, criterion, device)
                loss += loss_ml
            loss = np.sqrt(loss/len(test_loader))
        elif model_name == 'LR':
            model = LR(feature_dim=feature_dim,
                       time_window=time_window, predicted_len=prediction_hours).to(device)
            model.load_state_dict(torch.load(
                'save/{}_node_0.pth'.format(model_name)))
            for batch_index, batch_data in enumerate(test_loader):
                loss_lr = test_node(
                    model, model_name, batch_data, prediction_hours, criterion, device)
                loss += loss_lr
            loss = np.sqrt(loss/len(test_loader))
        else:
            model = STTransformer(node_dim=node_dim, feature_dim=feature_dim,
                                  time_window=time_window, predicted_len=prediction_hours).to(device)
            model.load_state_dict(torch.load(
                'save/{}_0.pth'.format(model_name)))
            for batch_index, batch_data in enumerate(test_loader):
                loss_st = test_node(
                    model, model_name, batch_data, prediction_hours, criterion, device)
                loss += loss_st
            loss = np.sqrt(loss/len(test_loader))
        node_model_loss[model_name] = loss
        print(f'{model_name} has loss {loss}')
    return node_model_loss

