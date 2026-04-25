import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import math
import argparse
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax as Softmax
from tqdm import tqdm
from scipy import sparse


os.environ['PYTHONHASHSEED'] = "42"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def pro_pep_matrix(pro_pep, pro_names, pep_names):
    keys1 = pro_names
    keys2 = pep_names
    values1 = range(len(pro_names))
    values2 = range(len(pep_names))
    dict1 = dict(zip(keys1, values1))
    dict2 = dict(zip(keys2, values2))
    matrix = np.zeros([len(pro_names), len(pep_names)])

    for i in range(len(pro_pep)):
        x = pro_pep[i, 0]
        y = pro_pep[i, 1]
        x1 = dict1[x]
        y1 = dict2[y]
        matrix[x1, y1] = 1
    return matrix


def convert_to_sparse_matrix(matrix):
    rows, cols = np.nonzero(matrix)
    sparse_matrix = sparse.csr_matrix((matrix[rows, cols], (rows, cols)), shape=matrix.shape)
    return sparse_matrix


def softmax(x):
    return (np.exp(x) / np.exp(x).sum())


def subgraph(graph, seed, n_neighbors, node_sele_prob):
    picked_nodes = {seed}
    last_layer_nodes = {seed}

    to_pick = 1
    for n_neighbors_current in n_neighbors:
        to_pick = to_pick * n_neighbors_current
        neighbors = graph[list(last_layer_nodes), :].nonzero()[1]

        neighbors_prob = node_sele_prob[list(neighbors)]
        neighbors = list(set(neighbors))
        n_neigbors_real = min(to_pick, len(neighbors))
        if len(neighbors_prob) == 0:
            continue
        last_layer_nodes = set(
            np.random.choice(neighbors, n_neigbors_real, replace=False))
        picked_nodes |= last_layer_nodes
    indices = list(sorted(picked_nodes - {seed}))
    return indices


def batch_select_whole(PRO_matrix, PEP_matrix, neighbor, cell_size):
    print('Batch partitioning has started. Please wait.')
    node_ids = np.random.choice(PRO_matrix.shape[1], size=PRO_matrix.shape[1], replace=False)
    n_batch = math.ceil(node_ids.shape[0] / cell_size)
    indices_ss = []

    PRO_matrix1 = PRO_matrix
    dic = {}

    for i in tqdm(range(n_batch)):
        pro_indices_all = []
        pep_indices_all = []
        if i < n_batch:
            for index, node in enumerate(node_ids[i * cell_size:(i + 1) * cell_size]):
                rna_ = PRO_matrix1[:, node].todense()
                pro_indices = subgraph(PRO_matrix.transpose(), node, neighbor, np.squeeze(np.array(rna_)))
                pep_indices = subgraph(
                    PEP_matrix.transpose(),
                    node,
                    neighbor,
                    np.squeeze(np.array(PEP_matrix[:, node].todense()))
                )
                dic[node] = {'g': pro_indices, 'p': pep_indices}
                pro_indices_all = pro_indices_all + pro_indices
                pep_indices_all = pep_indices_all + pep_indices
            node_indices_all = node_ids[i * cell_size:(i + 1) * cell_size]
        else:
            for index, node in enumerate(node_ids[i * cell_size:]):
                rna_ = PRO_matrix1[:, node].todense()
                pro_indices = subgraph(
                    PRO_matrix.transpose(),
                    node,
                    neighbor,
                    np.squeeze(np.array(rna_[:, node].todense()))
                )
                pep_indices = subgraph(
                    PEP_matrix.transpose(),
                    node,
                    neighbor,
                    np.squeeze(np.array(PEP_matrix[:, node].todense()))
                )
                dic[node] = {'g': pro_indices, 'p': pep_indices}
                pro_indices_all = pro_indices_all + pro_indices
                pep_indices_all = pep_indices_all + pep_indices
            node_indices_all = node_ids[i * cell_size:]

        pro_indices_all = list(set(pro_indices_all))
        pep_indices_all = list(set(pep_indices_all))
        h = dict()
        h['pro_index'] = pro_indices_all
        h['pep_index'] = pep_indices_all
        h['cell_index'] = node_indices_all
        indices_ss.append(h)
    return indices_ss, node_ids, dic


class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=True, distribution='uniform', **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.distribution = distribution
        self.att = None
        self.res_att = None
        self.res = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def _initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                if self.distribution == 'uniform':
                    torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if self.distribution == 'normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, node_inp, node_type, edge_index, edge_type):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, edge_type=edge_type)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type):
        data_size = edge_index_i.size(0)

        self.res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue

                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]

                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    self.res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk

                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)

        res = res_msg * Softmax(self.res_att, edge_index_i).view(-1, self.n_heads, 1)

        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))

            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        self.res = res
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm=True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        self.res_att = None
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)

    def forward(self, meta_xs, node_type, edge_index, edge_type):
        if self.conv_name == 'hgt':
            a = self.base_conv(meta_xs, node_type, edge_index, edge_type)
            self.res_att = self.base_conv.res_att
            return a
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'dense_hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type)


class GNN_from_raw_encode(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='hgt',
                 prev_norm=True, last_norm=True):
        super(GNN_from_raw_encode, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()

        for ti in range(num_types):
            self.embedding1.append(nn.Linear(in_dim[ti], 256))

        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid))

        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm))
        self.gcs.append(
            GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm))

    def encode(self, x, t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return h1

    def forward(self, node_feature, node_type, edge_index, edge_type):
        node_embedding = []
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id], t_id))

        node_embedding = torch.stack(node_embedding)
        res = torch.zeros(node_embedding.size(0), self.n_hid).to(node_feature[0].device)

        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding[idx]))

        meta_xs = self.drop(res)
        del res

        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)

        return meta_xs


class GNN_from_raw_decode(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='hgt',
                 prev_norm=True, last_norm=True):
        super(GNN_from_raw_decode, self).__init__()
        self.gcs2 = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws2 = nn.ModuleList()
        self.drop2 = nn.Dropout(dropout)
        self.embedding2 = nn.ModuleList()

        for ti in range(num_types):
            self.embedding2.append(nn.Linear(256, in_dim[ti]))

        for t in range(num_types):
            self.adapt_ws2.append(nn.Linear(n_hid, 256))

        for l in range(n_layers - 1):
            self.gcs2.append(
                GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm))
        self.gcs2.append(
            GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm))

    def forward(self, node_feature, node_type, edge_index, edge_type):
        for gc in self.gcs2:
            meta_xs = gc(node_feature, node_type, edge_index, edge_type)

        meta_xs1 = self.drop2(meta_xs)
        meta_xs2 = torch.tanh(meta_xs1)

        res = torch.zeros(meta_xs2.size(0), 256).to(node_feature[0].device)

        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = F.relu(self.adapt_ws2[t_id](meta_xs2[idx]))

        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue

            if t_id == 0:
                node_embedding0 = self.embedding2[t_id](res[idx])

            if t_id == 1:
                node_embedding1 = self.embedding2[t_id](res[idx])

            if t_id == 2:
                node_embedding2 = self.embedding2[t_id](res[idx])

        return node_embedding0, node_embedding1, node_embedding2


def matrix01(matrix):
    indices = np.array(np.nonzero(matrix)).T
    rows, cols = matrix.shape[0], matrix.shape[1]

    matrix1 = np.zeros((rows, cols), dtype=int)

    for row, col in indices:
        matrix1[row, col] = 1

    return matrix1


class NodeDimensionReduction(nn.Module):
    def __init__(self, PRO_matrix, PEP_matrix, PP_matrix, indices, cell_names, Node_Ids, n_hid, n_heads,
                 n_layers, labsm, lr, wd, device, output_dir, num_types=3, num_relations=3, epochs=1, seed=42):
        super(NodeDimensionReduction, self).__init__()
        self.PRO_matrix = PRO_matrix
        self.PEP_matrix = PEP_matrix
        self.PP_matrix = PP_matrix
        self.indices = indices
        self.cell_names = cell_names
        self.Node_Ids = Node_Ids
        self.in_dim = [PRO_matrix.shape[0], PRO_matrix.shape[1], PEP_matrix.shape[1]]
        self.n_hid = n_hid
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.labsm = labsm
        self.lr = lr
        self.wd = wd
        self.device = device
        self.epochs = epochs
        self.seed = seed
        self.output_dir = output_dir

        self.gnn1 = GNN_from_raw_encode(in_dim=self.in_dim,
                                        n_hid=self.n_hid,
                                        num_types=self.num_types,
                                        num_relations=self.num_relations,
                                        n_heads=self.n_heads,
                                        n_layers=self.n_layers,
                                        dropout=0.3).to(self.device)
        self.gnn2 = GNN_from_raw_decode(in_dim=self.in_dim,
                                        n_hid=self.n_hid,
                                        num_types=self.num_types,
                                        num_relations=self.num_relations,
                                        n_heads=self.n_heads,
                                        n_layers=self.n_layers,
                                        dropout=0.3).to(self.device)

        self.optimizer1 = torch.optim.AdamW(self.gnn1.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer2 = torch.optim.AdamW(self.gnn2.parameters(), lr=self.lr, weight_decay=self.wd)

        self.scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer1, 'min', factor=0.5, patience=5, verbose=True)
        self.scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer2, 'min', factor=0.5, patience=5, verbose=True)

    def train_model(self, n_batch):
        print('NodeDimensionReduction training has started. Please wait.')

        loss_history = {'loss_kl': [], 'loss_kl1': [], 'loss_kl2': [], 'loss_kl3': []}

        for epoch in tqdm(range(self.epochs)):
            print('\n')
            cell_embedding = np.zeros((self.PRO_matrix.shape[1], self.n_hid))
            pro_embedding = np.zeros((self.PRO_matrix.shape[0], self.n_hid))
            pep_embedding = np.zeros((self.PEP_matrix.shape[0], self.n_hid))
            cell_decode = np.zeros((self.PRO_matrix.shape[1], self.PRO_matrix.shape[0]))

            total_loss_kl = 0
            total_loss_kl1 = 0
            total_loss_kl2 = 0
            total_loss_kl3 = 0

            for batch_id in np.arange(n_batch):
                pro_index = self.indices[batch_id]['pro_index']
                cell_index = self.indices[batch_id]['cell_index']
                pep_index = self.indices[batch_id]['pep_index']

                pro_feature = self.PRO_matrix[list(pro_index),]
                cell_feature = self.PRO_matrix[:, list(cell_index)].T
                pep_feature = self.PEP_matrix[list(pep_index),]
                pro_feature = torch.tensor(np.array(pro_feature.todense()), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
                pep_feature = torch.tensor(np.array(pep_feature.todense()), dtype=torch.float32).to(self.device)
                node_feature = [cell_feature, pro_feature, pep_feature]

                pro_cell_sub = self.PRO_matrix[list(pro_index),][:, list(cell_index)]
                pep_cell_sub = self.PEP_matrix[list(pep_index),][:, list(cell_index)]
                pro_pep_sub = self.PP_matrix[list(pro_index),][:, list(pep_index)]

                pro_cell_edge_index1 = list(np.nonzero(pro_cell_sub)[0] + pro_cell_sub.shape[1]) + list(
                    np.nonzero(pro_cell_sub)[1])
                pro_cell_edge_index2 = list(np.nonzero(pro_cell_sub)[1]) + list(
                    np.nonzero(pro_cell_sub)[0] + pro_cell_sub.shape[1])
                pro_cell_edge_index = torch.LongTensor([pro_cell_edge_index1, pro_cell_edge_index2]).to(self.device)

                pep_cell_edge_index1 = list(
                    np.nonzero(pep_cell_sub)[0] + pro_cell_sub.shape[0] + pep_cell_sub.shape[1]) + list(
                    np.nonzero(pep_cell_sub)[1])
                pep_cell_edge_index2 = list(np.nonzero(pep_cell_sub)[1]) + list(
                    np.nonzero(pep_cell_sub)[0] + pro_cell_sub.shape[0] + pep_cell_sub.shape[1])
                pep_cell_edge_index = torch.LongTensor([pep_cell_edge_index1, pep_cell_edge_index2]).to(self.device)

                pro_pep_edge_index1 = list(
                    np.nonzero(pro_pep_sub)[0] + pro_cell_sub.shape[1]) + list(
                    np.nonzero(pro_pep_sub)[1] + pro_cell_sub.shape[0] + pep_cell_sub.shape[1])
                pro_pep_edge_index2 = list(
                    np.nonzero(pro_pep_sub)[1] + pro_cell_sub.shape[0] + pep_cell_sub.shape[1]) + list(
                    np.nonzero(pro_pep_sub)[0] + pro_cell_sub.shape[1])
                pro_pep_edge_index = torch.LongTensor([pro_pep_edge_index1, pro_pep_edge_index2]).to(self.device)
                edge_index = torch.cat((pro_cell_edge_index, pep_cell_edge_index, pro_pep_edge_index), dim=1)

                node_type = torch.LongTensor(np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(pro_index))) + list(
                        np.ones(len(pep_index)) * 2))).to(self.device)

                edge_type = torch.LongTensor(np.array(list(
                    np.zeros(np.nonzero(pro_cell_sub)[0].shape[0])) + list(
                    np.ones(np.nonzero(pro_cell_sub)[1].shape[0])) + list(
                    2 * np.ones(np.nonzero(pep_cell_sub)[0].shape[0])) + list(
                    3 * np.ones(np.nonzero(pep_cell_sub)[1].shape[0])) + list(
                    4 * np.ones(np.nonzero(pro_pep_sub)[0].shape[0])) + list(
                    5 * np.ones(np.nonzero(pro_pep_sub)[1].shape[0])))).to(self.device)

                node_rep = self.gnn1.forward(node_feature, node_type, edge_index, edge_type).to(self.device)

                cell_emb = node_rep[node_type == 0]
                pro_emb = node_rep[node_type == 1]
                pep_emb = node_rep[node_type == 2]

                cell_dec, pro_dec, pep_dec = self.gnn2.forward(node_rep, node_type, edge_index, edge_type)

                cell_dec = cell_dec.to(self.device)
                pro_dec = pro_dec.to(self.device)
                pep_dec = pep_dec.to(self.device)

                cell_emb1 = cell_emb.detach().cpu()
                pro_emb1 = pro_emb.detach().cpu()
                pep_emb1 = pep_emb.detach().cpu()

                cell_dec1 = cell_dec.detach().cpu()

                for k in range(len(cell_index)):
                    cell_embedding[cell_index[k]] = cell_emb1[k]

                for i in range(len(pro_index)):
                    pro_embedding[pro_index[i]] = pro_emb1[i]

                for j in range(len(pep_index)):
                    pep_embedding[pep_index[j]] = pep_emb1[j]

                for k in range(len(cell_index)):
                    cell_decode[cell_index[k]] = cell_dec1[k]

                cell_sub = np.array(cell_feature.cpu())
                pro_sub = np.array(pro_feature.cpu())
                pep_sub = np.array(pep_feature.cpu())

                cell_sub_01 = torch.tensor(matrix01(cell_sub), dtype=torch.float32).to(self.device)
                pro_sub_01 = torch.tensor(matrix01(pro_sub), dtype=torch.float32).to(self.device)
                pep_sub_01 = torch.tensor(matrix01(pep_sub), dtype=torch.float32).to(self.device)

                logp_x1 = F.log_softmax(cell_dec * cell_sub_01, dim=-1)
                p_y1 = F.softmax(cell_feature, dim=-1)

                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='batchmean')
                print('loss_cell:  %.4f' % loss_kl1)

                logp_x2 = F.log_softmax(pro_dec * pro_sub_01, dim=-1)
                p_y2 = F.softmax(pro_feature, dim=-1)

                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='batchmean')
                print('loss_pro:  %.4f' % loss_kl2)

                logp_x3 = F.log_softmax(pep_dec * pep_sub_01, dim=-1)
                p_y3 = F.softmax(pep_feature, dim=-1)

                loss_kl3 = F.kl_div(logp_x3, p_y3, reduction='batchmean')
                print('loss_pep:  %.4f' % loss_kl3)

                loss_kl = loss_kl1 + loss_kl2 + loss_kl3
                print('total_loss: %.4f' % loss_kl)

                total_loss_kl += loss_kl.item()
                total_loss_kl1 += loss_kl1.item()
                total_loss_kl2 += loss_kl2.item()
                total_loss_kl3 += loss_kl3.item()

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                loss_kl.backward()
                self.optimizer1.step()
                self.optimizer2.step()

            average_loss_kl = total_loss_kl / n_batch
            average_loss_kl1 = total_loss_kl1 / n_batch
            average_loss_kl2 = total_loss_kl2 / n_batch
            average_loss_kl3 = total_loss_kl3 / n_batch

            loss_history['loss_kl'].append(average_loss_kl)
            loss_history['loss_kl1'].append(average_loss_kl1)
            loss_history['loss_kl2'].append(average_loss_kl2)
            loss_history['loss_kl3'].append(average_loss_kl3)

            print(
                f'Epoch {epoch}, Loss: {average_loss_kl}, Loss1: {average_loss_kl1}, Loss2: {average_loss_kl2}, Loss3: {average_loss_kl3}')

            if (epoch + 1) % 10 == 0:
                torch.save(self, os.path.join(self.output_dir, f'model_epoch_{epoch}.pth'))
                torch.save(self.state_dict(), os.path.join(self.output_dir, f'model_state_dict_epoch_{epoch}.pth'))
                np.save(os.path.join(self.output_dir, f'cell_decode_epoch_{epoch}.npy'), cell_decode)

            np.save(os.path.join(self.output_dir, 'loss_history.npy'), loss_history)

        print('NodeDimensionReduction training has finished.')
        return cell_embedding, pro_embedding, pep_embedding, cell_decode


def parse_args():
    parser = argparse.ArgumentParser(description='Train NodeDimensionReduction model')
    parser.add_argument('--input_pro', type=str, required=True, help='Path to protein input CSV')
    parser.add_argument('--input_pep', type=str, required=True, help='Path to peptide input CSV')
    parser.add_argument('--output', type=str, required=True, help='Output directory')

    parser.add_argument('--cuda_visible_devices', type=str, default='6', help='CUDA device id')
    parser.add_argument('--cell_size', type=int, default=10, help='Number of cells per subgraph')
    parser.add_argument('--n_hid', type=int, default=104, help='Hidden dimension size')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of network layers')
    parser.add_argument('--labsm', type=float, default=0.1, help='Label smoothing rate')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    pro_cell = pd.read_csv(args.input_pro).fillna(0.)
    pep_cell = pd.read_csv(args.input_pep).fillna(0.)

    pro_pep = np.array(pep_cell)[:, :2]
    cell_names = pro_cell.columns.to_numpy()[1:]
    pro_names = np.array(pro_cell)[:, 0]
    pep_names = np.array(pep_cell)[:, 1]
    pro_cell = np.array(pro_cell)[:, 1:]
    pep_cell = np.array(pep_cell)[:, 2:]

    pro_pep = pro_pep_matrix(pro_pep, pro_names, pep_names)

    pro_cell = np.array(pro_cell, dtype=float)
    pep_cell = np.array(pep_cell, dtype=float)
    pro_pep = np.array(pro_pep, dtype=float)

    pro_cell = convert_to_sparse_matrix(np.array(pro_cell))
    pep_cell = convert_to_sparse_matrix(np.array(pep_cell))

    PRO_matrix = pro_cell
    PEP_matrix = pep_cell
    PP_matrix = pro_pep

    pro_num = PRO_matrix.shape[0]
    pep_num = PEP_matrix.shape[0]

    neighbor = max(pro_num, pep_num)
    indices, Node_Ids, dic = batch_select_whole(PRO_matrix, PEP_matrix, neighbor=[neighbor], cell_size=args.cell_size)
    n_batch = len(indices)

    node_model = NodeDimensionReduction(
        PRO_matrix, PEP_matrix, PP_matrix, indices, cell_names, Node_Ids,
        n_hid=args.n_hid,
        n_heads=args.nheads,
        n_layers=args.nlayers,
        labsm=args.labsm,
        lr=args.lr,
        wd=args.wd,
        device=device,
        output_dir=args.output,
        num_types=3,
        num_relations=3,
        epochs=args.epochs,
        seed=args.seed
    )

    cell_embedding, pro_embedding, pep_embedding, cell_decode = node_model.train_model(n_batch=n_batch)

    print('finish')


if __name__ == '__main__':
    main()