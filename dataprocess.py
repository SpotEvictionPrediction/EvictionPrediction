from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import pickle
from multiprocessing import Pool
from itertools import repeat


def combine_tables(file_data_path, file_rate_path, file_gt_path, save_to_path):
    df = pd.read_csv(file_data_path)
    df_rate = pd.read_csv(file_rate_path)
    df_gt = pd.read_csv(file_gt_path)
    df = df.fillna(0)
    df_rate = df_rate.fillna(0)
    df_gt = df_gt.fillna(0)
    table1 = pd.pivot_table(df_rate, values='EvictedSpotVMCount', index=['ClusterId', 'SnapshotTS', 'NodeId'],
                            columns=['PredHorizon'], aggfunc=np.sum, fill_value=0)
    table1.rename(columns={'PredHorizon_1.0_h': 'PredHorizon_1h_Count',
                           'PredHorizon_2.0_h': 'PredHorizon_2h_Count',
                           'PredHorizon_3.0_h': 'PredHorizon_3h_Count',
                           'PredHorizon_-1.0_h': 'PredHorizon_-1h_Count',
                           'PredHorizon_-2.0_h': 'PredHorizon_-2h_Count',
                           'PredHorizon_-3.0_h': 'PredHorizon_-3h_Count'}, inplace=True)

    table2 = pd.pivot_table(df_rate, values='EvictedSpotVMCore', index=['ClusterId', 'SnapshotTS', 'NodeId'],
                            columns=['PredHorizon'], aggfunc=np.sum, fill_value=0)
    table2.rename(columns={'PredHorizon_1.0_h': 'PredHorizon_1h_Core',
                           'PredHorizon_2.0_h': 'PredHorizon_2h_Core',
                           'PredHorizon_3.0_h': 'PredHorizon_3h_Core',
                           'PredHorizon_-1.0_h': 'PredHorizon_-1h_Core',
                           'PredHorizon_-2.0_h': 'PredHorizon_-2h_Core',
                           'PredHorizon_-3.0_h': 'PredHorizon_-3h_Core'}, inplace=True)

    total = pd.merge(df, table1, how="outer", on=[
                     'ClusterId', 'SnapshotTS', 'NodeId'])
    total = pd.merge(total, table2, how="outer", on=[
                     'ClusterId', 'SnapshotTS', 'NodeId'])
    total = pd.merge(total, df_gt, how="outer", on=[
                     'ClusterId', 'SnapshotTS', 'NodeId'])
    total = total.fillna(0)
    total['SnapshotTS'] = pd.to_datetime(total['SnapshotTS'])
    total.to_csv(save_to_path, index=False)


def convert2cluster_table(combine_path, cluster_evictionrate_path, result_path):
    df_nodes = pd.read_csv(combine_path)
    df_cluster_rate = pd.read_csv(cluster_evictionrate_path)
    df_nodes['SnapshotTS'] = pd.to_datetime(df_nodes['SnapshotTS'])
    df_cluster_rate['SnapshotTS'] = pd.to_datetime(
        df_cluster_rate['SnapshotTS'])
    df_cluster_agg = df_nodes.groupby(['ClusterId', 'SnapshotTS']).sum().drop(columns=['evictedRateN3h', 'evictedRateN2h', 'evictedRateN1h',
                                                                                       'evictedRate1h', 'evictedRate2h', 'evictedRate3h'])
    total = pd.merge(df_cluster_agg, df_cluster_rate,
                     how="outer", on=['ClusterId', 'SnapshotTS'])
    total = total.fillna(0)
    # 'test/combine_cluster.csv' or 'data/combine_cluster.csv'
    total.to_csv(result_path, index=False)


def convert2node_table(combine_path, cluster_evictionrate_path, result_path):
    df_nodes = pd.read_csv(combine_path)
    df_cluster_rate = pd.read_csv(cluster_evictionrate_path)
    df_nodes['SnapshotTS'] = pd.to_datetime(df_nodes['SnapshotTS'])
    df_cluster_rate['SnapshotTS'] = pd.to_datetime(
        df_cluster_rate['SnapshotTS'])
    total = pd.merge(df_nodes, df_cluster_rate, how="outer",
                     on=['ClusterId', 'SnapshotTS'])
    total = total.fillna(0)
    total.to_csv(result_path, index=False)


def process_node_reg(cluster, combine_nodes_path, sliding_window):
    df = pd.read_csv(combine_nodes_path)
    df['SnapshotTS'] = pd.to_datetime(df['SnapshotTS'])
    maxtime = df['SnapshotTS'].max()-pd.Timedelta(hours=5)
    mintime = df['SnapshotTS'].min()+pd.Timedelta(hours=5)
    feature_dim = len(df.columns)-9
    time_horizon = (maxtime-mintime)/pd.Timedelta(hours=1)
    node_dim = df.groupby(['ClusterId', 'SnapshotTS'])['spotVMCount'].apply(
        lambda x: (x > 0).sum()).reset_index(name='spotVMNodeCount')['spotVMNodeCount'].max()

    subdata = []
    subnonSpotNode_sum_max = np.zeros((feature_dim,))
    subspotNodeCount = []
    df_cluster = df.loc[df['ClusterId'] == cluster]
    print(f'cluster Id is {cluster} with node dim {node_dim}')
    for t in range(0, int(time_horizon-sliding_window), 4):
        time_start = mintime+pd.Timedelta(hours=t)
        time_end = time_start+pd.Timedelta(hours=sliding_window-1)
        df_slice = df_cluster[(df_cluster['SnapshotTS'] >= time_start) & (
            df_cluster['SnapshotTS'] <= time_end)]
        node_list = df_slice[(df_slice['SnapshotTS'] ==
                              time_end)]['NodeId'].unique()
        spotNode_list = df_slice[(df_slice['spotVMCount'] > 0) & (
            df_slice['SnapshotTS'] == time_end)]['NodeId'].unique()
        nonSpotNode_list = list(set(node_list)-set(spotNode_list))
        df_nonSpotNode = df_slice[(df_slice['NodeId'].isin(nonSpotNode_list))]
        nodes_st = np.zeros((node_dim+1, sliding_window, feature_dim))
        for node_idx, spotNode in enumerate(spotNode_list):
            for it in range(sliding_window):
                cur_time = time_start+pd.Timedelta(hours=it)
                nodes_st[node_idx, it, :] = df_slice[(df_slice['SnapshotTS'] == cur_time) & (df_slice['NodeId'] == spotNode)][['NodeCore', 'NodeMemoryInGB', 'spotVMCount',
                                                                                                                              'onDemandVMCount', 'spotVMCore', 'onDemandVMCore',
                                                                                                                               'spotVMMemoryInGB', 'onDemandVMMemoryInGB', 'PredHorizon_-1h_Count',
                                                                                                                               'PredHorizon_-2h_Count', 'PredHorizon_-3h_Count', 'PredHorizon_1h_Count',
                                                                                                                               'PredHorizon_2h_Count', 'PredHorizon_3h_Count', 'PredHorizon_-1h_Core',
                                                                                                                               'PredHorizon_-2h_Core', 'PredHorizon_-3h_Core', 'PredHorizon_1h_Core',
                                                                                                                               'PredHorizon_2h_Core', 'PredHorizon_3h_Core', 'evictedRateN3h',
                                                                                                                               'evictedRateN2h', 'evictedRateN1h', 'evictedRate1h',
                                                                                                                               'evictedRate2h', 'evictedRate3h']].iloc[0].astype(float)
        for it in range(sliding_window):
            cur_time = time_start+pd.Timedelta(hours=it)
            nodes_st[-1, it, :] = df_nonSpotNode[(df_nonSpotNode['SnapshotTS'] == cur_time)][['NodeCore', 'NodeMemoryInGB', 'spotVMCount',
                                                                                             'onDemandVMCount', 'spotVMCore', 'onDemandVMCore',
                                                                                              'spotVMMemoryInGB', 'onDemandVMMemoryInGB', 'PredHorizon_-1h_Count',
                                                                                              'PredHorizon_-2h_Count', 'PredHorizon_-3h_Count', 'PredHorizon_1h_Count',
                                                                                              'PredHorizon_2h_Count', 'PredHorizon_3h_Count', 'PredHorizon_-1h_Core',
                                                                                              'PredHorizon_-2h_Core', 'PredHorizon_-3h_Core', 'PredHorizon_1h_Core',
                                                                                              'PredHorizon_2h_Core', 'PredHorizon_3h_Core', 'evictedRateN3h',
                                                                                              'evictedRateN2h', 'evictedRateN1h', 'evictedRate1h',
                                                                                              'evictedRate2h', 'evictedRate3h']].astype(float).sum()
            subnonSpotNode_sum_max = np.maximum(
                subnonSpotNode_sum_max, nodes_st[-1, it, :])
        subdata.append(nodes_st)
        subspotNodeCount.append(len(spotNode_list))
    return (subdata, subspotNodeCount, subnonSpotNode_sum_max)


def process_node(cluster, combine_nodes_path, sliding_window):
    df = pd.read_csv(combine_nodes_path)
    df['SnapshotTS'] = pd.to_datetime(df['SnapshotTS'])
    maxtime = df['SnapshotTS'].max()-pd.Timedelta(hours=5)
    mintime = df['SnapshotTS'].min()+pd.Timedelta(hours=5)
    feature_dim = len(df.columns)-9
    time_horizon = (maxtime-mintime)/pd.Timedelta(hours=1)
    node_dim = df.groupby(['ClusterId', 'SnapshotTS'])['spotVMCount'].apply(
        lambda x: (x > 0).sum()).reset_index(name='spotVMNodeCount')['spotVMNodeCount'].max()

    subdata = []
    subnonSpotNode_sum_max = np.zeros((feature_dim,))
    subspotNodeCount = []
    subspotVMCount = []
    targetEvictionRate = []
    df_cluster = df.loc[df['ClusterId'] == cluster]
    print(f'cluster Id is {cluster} with node dim {node_dim}')
    for t in range(0, int(time_horizon-sliding_window), 4):
        time_start = mintime+pd.Timedelta(hours=t)
        time_end = time_start+pd.Timedelta(hours=sliding_window-1)
        df_slice = df_cluster[(df_cluster['SnapshotTS'] >= time_start) & (
            df_cluster['SnapshotTS'] <= time_end)]
        node_list = df_slice[(df_slice['SnapshotTS'] ==
                              time_end)]['NodeId'].unique()
        spotNode_list = df_slice[(df_slice['spotVMCount'] > 0) & (
            df_slice['SnapshotTS'] == time_end)]['NodeId'].unique()
        nonSpotNode_list = list(set(node_list)-set(spotNode_list))
        df_nonSpotNode = df_slice[(df_slice['NodeId'].isin(nonSpotNode_list))]
        nodes_st = np.zeros((node_dim+1, sliding_window, feature_dim))
        spot_count = np.zeros((node_dim+1,))
        target_cluster_rate = np.zeros((6,))
        target_cluster_rate = df_slice[(df_slice['SnapshotTS'] == time_end)][['evictedRateN3h_y',
                                                                              'evictedRateN2h_y', 'evictedRateN1h_y', 'evictedRate1h_y',
                                                                             'evictedRate2h_y', 'evictedRate3h_y']].iloc[0].astype(float)
        for node_idx, spotNode in enumerate(spotNode_list):
            for it in range(sliding_window):
                cur_time = time_start+pd.Timedelta(hours=it)
                nodes_st[node_idx, it, :] = df_slice[(df_slice['SnapshotTS'] == cur_time) & (df_slice['NodeId'] == spotNode)][['NodeCore', 'NodeMemoryInGB', 'spotVMCount',
                                                                                                                              'onDemandVMCount', 'spotVMCore', 'onDemandVMCore',
                                                                                                                               'spotVMMemoryInGB', 'onDemandVMMemoryInGB', 'PredHorizon_-1h_Count',
                                                                                                                               'PredHorizon_-2h_Count', 'PredHorizon_-3h_Count', 'PredHorizon_1h_Count',
                                                                                                                               'PredHorizon_2h_Count', 'PredHorizon_3h_Count', 'PredHorizon_-1h_Core',
                                                                                                                               'PredHorizon_-2h_Core', 'PredHorizon_-3h_Core', 'PredHorizon_1h_Core',
                                                                                                                               'PredHorizon_2h_Core', 'PredHorizon_3h_Core', 'evictedRateN3h_x',
                                                                                                                               'evictedRateN2h_x', 'evictedRateN1h_x', 'evictedRate1h_x',
                                                                                                                               'evictedRate2h_x', 'evictedRate3h_x']].iloc[0].astype(float)
                spot_count[node_idx] = df_slice[(df_slice['SnapshotTS'] == cur_time) & (
                    df_slice['NodeId'] == spotNode)]['spotVMCount'].iloc[0].astype(float)

        for it in range(sliding_window):
            cur_time = time_start+pd.Timedelta(hours=it)
            nodes_st[-1, it, :] = df_nonSpotNode[(df_nonSpotNode['SnapshotTS'] == cur_time)][['NodeCore', 'NodeMemoryInGB', 'spotVMCount',
                                                                                             'onDemandVMCount', 'spotVMCore', 'onDemandVMCore',
                                                                                              'spotVMMemoryInGB', 'onDemandVMMemoryInGB', 'PredHorizon_-1h_Count',
                                                                                              'PredHorizon_-2h_Count', 'PredHorizon_-3h_Count', 'PredHorizon_1h_Count',
                                                                                              'PredHorizon_2h_Count', 'PredHorizon_3h_Count', 'PredHorizon_-1h_Core',
                                                                                              'PredHorizon_-2h_Core', 'PredHorizon_-3h_Core', 'PredHorizon_1h_Core',
                                                                                              'PredHorizon_2h_Core', 'PredHorizon_3h_Core', 'evictedRateN3h_x',
                                                                                              'evictedRateN2h_x', 'evictedRateN1h_x', 'evictedRate1h_x',
                                                                                              'evictedRate2h_x', 'evictedRate3h_x']].astype(float).sum()
            subnonSpotNode_sum_max = np.maximum(
                subnonSpotNode_sum_max, nodes_st[-1, it, :])
        subdata.append(nodes_st)
        subspotNodeCount.append(len(spotNode_list))
        subspotVMCount.append(spot_count)
        targetEvictionRate.append(target_cluster_rate)
    return (subdata, subspotNodeCount, subnonSpotNode_sum_max, subspotVMCount, targetEvictionRate)


def save_processed_data_node_reg(combine_nodes_path, save_path, sliding_window=48):
    df = pd.read_csv(combine_nodes_path)
    feature_dim = len(df.columns)-9
    data = []
    spotNodeCount = []
    nonSpotNode_sum_max = np.zeros((feature_dim,))
    cluster_list = list(df['ClusterId'].unique())
    pool = Pool()
    results = pool.starmap(process_node_reg, zip(
        cluster_list, repeat(combine_nodes_path), repeat(sliding_window)))

    for result in results:
        (idata, ispotNodeCount, inonSpotNode_sum_max) = result
        data += idata
        spotNodeCount += ispotNodeCount
        nonSpotNode_sum_max = np.maximum(
            nonSpotNode_sum_max, inonSpotNode_sum_max)

    open_file = open(save_path, "wb")
    pickle.dump((data, spotNodeCount, nonSpotNode_sum_max), open_file)
    open_file.close()


def save_processed_node_data(combine_nodes_path, save_path, sliding_window=48):
    df = pd.read_csv(combine_nodes_path)
    feature_dim = len(df.columns)-9
    data = []
    spotNodeCount = []
    spotVMCount = []
    targetEvictionRate = []
    nonSpotNode_sum_max = np.zeros((feature_dim,))
    cluster_list = list(df['ClusterId'].unique())
    pool = Pool()
    results = pool.starmap(process_node, zip(
        cluster_list, repeat(combine_nodes_path), repeat(sliding_window)))

    for result in results:
        (idata, ispotNodeCount, inonSpotNode_sum_max,
         ispotVMCount, itargetEvictionRate) = result
        data += idata
        spotNodeCount += ispotNodeCount
        nonSpotNode_sum_max = np.maximum(
            nonSpotNode_sum_max, inonSpotNode_sum_max)
        spotVMCount += ispotVMCount
        targetEvictionRate += itargetEvictionRate

    open_file = open(save_path, "wb")
    pickle.dump((data, spotNodeCount, nonSpotNode_sum_max,
                spotVMCount, targetEvictionRate), open_file)
    open_file.close()

def process_cluster(combine_cluster_path, cluster_list=None, sliding_window=48):
    df = pd.read_csv(combine_cluster_path)
    df['SnapshotTS'] = pd.to_datetime(df['SnapshotTS'])
    maxtime = df['SnapshotTS'].max()-pd.Timedelta(hours=5)
    mintime = df['SnapshotTS'].min()+pd.Timedelta(hours=5)
    feature_dim = len(df.columns)-2  # exclude ClusterId, SnapshotTS
    time_horizon = (maxtime-mintime)/pd.Timedelta(hours=1)
    if cluster_list is None:
        cluster_list = df['ClusterId'].unique()
    data = []
    for cluster in cluster_list:
        df_cluster = df.loc[df['ClusterId'] == cluster]
        print(f'cluster Id is {cluster}')
        for t in range(int(time_horizon-sliding_window)):
            cluster_st = np.zeros((sliding_window, feature_dim))  # T * F
            time_start = mintime+pd.Timedelta(hours=t)
            time_end = time_start+pd.Timedelta(hours=sliding_window-1)
            df_slice = df_cluster[(df_cluster['SnapshotTS'] >= time_start) & (
                df_cluster['SnapshotTS'] <= time_end)]
            for it in range(sliding_window):
                cur_time = time_start+pd.Timedelta(hours=it)
                df_cur_time = df_slice[(df_slice['SnapshotTS'] == cur_time)]
                cluster_st[it, :] = df_cur_time[['NodeCore', 'NodeMemoryInGB', 'spotVMCount',
                                                'onDemandVMCount', 'spotVMCore', 'onDemandVMCore',
                                                 'spotVMMemoryInGB', 'onDemandVMMemoryInGB', 'PredHorizon_-1h_Count',
                                                 'PredHorizon_-2h_Count', 'PredHorizon_-3h_Count', 'PredHorizon_1h_Count',
                                                 'PredHorizon_2h_Count', 'PredHorizon_3h_Count', 'PredHorizon_-1h_Core',
                                                 'PredHorizon_-2h_Core', 'PredHorizon_-3h_Core', 'PredHorizon_1h_Core',
                                                 'PredHorizon_2h_Core', 'PredHorizon_3h_Core', 'evictedRateN3h',
                                                 'evictedRateN2h', 'evictedRateN1h', 'evictedRate1h',
                                                 'evictedRate2h', 'evictedRate3h']].iloc[0].astype(float)
            data.append(cluster_st)
    return data


def norm_and_remove(data, predict_window):
    feature_dim = data[0].shape[1]-6
    for idata in data:
        for idx_f in range(feature_dim):
            max_val = idata[:, idx_f].max()
            min_val = idata[:, idx_f].min()
            if max_val-min_val > 0:
                max_val = idata[:, idx_f] = (
                    idata[:, idx_f]-min_val)/(max_val-min_val)
    target = []
    index_list = np.array([[11, 12, 13], [17, 18, 19], [23, 24, 25]])
    for idata in data:
        tar = idata[-1, -predict_window:].copy()
        for t in range(1, predict_window+1):
            remove_indexes = list(np.concatenate(index_list[:, -t:]))
            idata[-t, remove_indexes] = 0
        target.append(tar)
    return (data, target)


def save_processed_cluster_data(combine_cluster_path, save_path, sliding_window=48, predict_window=3):
    data = process_cluster(combine_cluster_path,
                           cluster_list=None, sliding_window=sliding_window)
    (data, target) = norm_and_remove(data, predict_window)
    open_file = open(save_path, "wb")
    pickle.dump((data, target), open_file)
    open_file.close()


def save_processed_cluster_bucket_data(bucket_list, bucket_dict, combine_cluster_path, sliding_window=48, predict_window=3):
    for bucket_id in bucket_list:
        print(f'bucket_id is {bucket_id}')
        cluster_list = bucket_dict[bucket_id]
        save_path = 'test/baseline_cluster_{}.pkl'.format(bucket_id)
        data = process_cluster(combine_cluster_path,
                               cluster_list, sliding_window)
        (data, target) = norm_and_remove(data, predict_window)
        open_file = open(save_path, "wb")
        pickle.dump((data, target), open_file)
        open_file.close()


class EvictionrateClusterDataset(Dataset):
    def __init__(self, processed_data_path, time_window):
        super(EvictionrateClusterDataset, self).__init__()
        self.time_window = time_window
        open_file = open(processed_data_path, "rb")
        (self.data, self.target) = pickle.load(open_file)
        open_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.target[index])

    def get_feature_dim(self):
        return self.data[0].shape[1]


def collate_fn_cluster(batch):
    # normalize batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cluster_s_batch = torch.as_tensor(
        np.array([data for (data, target) in batch])).float().to(device)
    target_batch = torch.as_tensor(
        np.array([target for (data, target) in batch])).float().to(device)
    return (cluster_s_batch, target_batch)


class EvictionrateNodeDataset(Dataset):
    def __init__(self, processed_nodes_data_path, time_window):
        super(EvictionrateNodeDataset, self).__init__()
        self.time_window = time_window
        open_nodes_file = open(processed_nodes_data_path, "rb")
        (self.data, self.spotNodeCount_list, self.nonSpotNode_sum_max,
         self.spotVMCount, self.targetEvictionRate) = pickle.load(open_nodes_file)
        open_nodes_file.close()

        self.acc_spotNodeCount_list = list(np.cumsum(self.spotNodeCount_list))
        assert self.acc_spotNodeCount_list[0] > 1
        # normalization
        for idx, idata in enumerate(self.data):
            for idx_f in range(self.data[idx].shape[2]-6):
                max_val = self.data[idx][:-1, :, idx_f].max()
                if max_val > 0:
                    self.data[idx][:-1, :, idx_f] /= max_val
                self.data[idx][-1, :,
                               idx_f] /= abs(self.nonSpotNode_sum_max[idx_f])
            self.data[idx][self.data[idx] < 0] = -1

        self.index2actidx = []
        prev_spotNodeCount = 0
        idx = 0
        for index in range(sum(self.spotNodeCount_list)):
            act_idx = index-prev_spotNodeCount
            if index >= self.acc_spotNodeCount_list[idx]:
                prev_spotNodeCount = self.acc_spotNodeCount_list[idx]
                act_idx = index-prev_spotNodeCount
                idx += 1
            self.index2actidx.append((idx, act_idx))

    def __len__(self):
        return sum(self.spotNodeCount_list)

    def __getitem__(self, index):
        (idx, act_idx) = self.index2actidx[index]
        # assert (self.data[idx][act_idx][-1,-3:]>=0).all()==True
        return (self.data[idx], self.spotNodeCount_list[idx], self.spotVMCount[idx], self.targetEvictionRate[idx])

    def get_node_dim(self):
        return self.data[0].shape[0]

    def get_feature_dim(self):
        return self.data[0].shape[2]


def collate_fn_node(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_t_batch = torch.as_tensor(np.array([nodes_st for (
        nodes_st, nodes_count, spot_count, cluster_target) in batch])).float().to(device)
    nodes_st_batch = torch.as_tensor(np.array([nodes_count for (
        nodes_st, nodes_count, spot_count, cluster_target) in batch])).float().to(device)
    spot_count_batch = torch.as_tensor(np.array([spot_count for (
        nodes_st, nodes_count, spot_count, cluster_target) in batch])).float().to(device)
    cluster_target_batch = torch.as_tensor(np.array([cluster_target for (
        nodes_st, nodes_count, spot_count, cluster_target) in batch])).float().to(device)
    return (node_t_batch, nodes_st_batch, spot_count_batch, cluster_target_batch)


class EvictionrateRegDataset(Dataset):
    def __init__(self, processed_data_path, time_window):
        super(EvictionrateRegDataset, self).__init__()
        self.time_window = time_window
        open_file = open(processed_data_path, "rb")
        (self.data, self.spotNodeCount_list, self.nonSpotNode_sum_max,
         self.spotVMCount, self.targetEvictionRate) = pickle.load(open_file)
        open_file.close()
        self.acc_spotNodeCount_list = list(np.cumsum(self.spotNodeCount_list))
        assert self.acc_spotNodeCount_list[0] > 1
        # normalization
        for idx, idata in enumerate(self.data):
            for idx_f in range(self.data[idx].shape[2]-6):
                max_val = self.data[idx][:-1, :, idx_f].max()
                if max_val > 0:
                    self.data[idx][:-1, :, idx_f] /= max_val
                self.data[idx][-1, :,
                               idx_f] /= abs(self.nonSpotNode_sum_max[idx_f])
            self.data[idx][self.data[idx] < 0] = -1

        self.index2actidx = []
        prev_spotNodeCount = 0
        idx = 0
        for index in range(sum(self.spotNodeCount_list)):
            act_idx = index-prev_spotNodeCount
            if index >= self.acc_spotNodeCount_list[idx]:
                prev_spotNodeCount = self.acc_spotNodeCount_list[idx]
                act_idx = index-prev_spotNodeCount
                idx += 1
            self.index2actidx.append((idx, act_idx))

    def __len__(self):
        return sum(self.spotNodeCount_list)

    def __getitem__(self, index):
        (idx, act_idx) = self.index2actidx[index]
        # assert (self.data[idx][act_idx][-1,-3:]>=0).all()==True
        return (self.data[idx][act_idx], self.data[idx])

    def get_node_dim(self):
        return self.data[0].shape[0]

    def get_feature_dim(self):
        return self.data[0].shape[2]


def collate_fn_reg(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_t_batch = torch.as_tensor(
        np.array([node_t for (node_t, nodes_st) in batch])).float().to(device)
    nodes_st_batch = torch.as_tensor(
        np.array([nodes_st for (node_t, nodes_st) in batch])).float().to(device)
    return (node_t_batch, nodes_st_batch)
