import networkx
import torch
from networkx import jaccard_coefficient, preferential_attachment, adamic_adar_index, resource_allocation_index, \
    pagerank, simrank_similarity
from torch_geometric.datasets import FakeDataset, SNAPDataset
from torch_geometric.transforms import LargestConnectedComponents
from SAVAGE.Data.custom_graph_dataset import CustomGraphDataset
from networkx import adjacency_matrix


class DataSyn:
    x = None
    edge_index = None


class DataWiki:
    x = None
    edge_index = None


def check_degree(source, target, adj):
    s_o = adj[source].sum()
    s_i = adj[:, source].sum()
    t_i = adj[target].sum()
    t_o = adj[:, target].sum()

    d = (s_o and s_i and t_i and t_o).bool()

    return ~d


def check_repeated(source, target, sources, targets, max_pairs):
    rep = False
    for j in range(max_pairs):
        a = sources[j]
        b = targets[j]
        if source == a and target == b:
            rep = True

    return rep


def build_data(path, data_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_name == "fake":
        dataset = FakeDataset(
            num_graphs=1, avg_num_nodes=4000, num_classes=32, is_undirected=False, task="node"
        )
        data = dataset[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "arxiv":
        data = CustomGraphDataset(root=path + "/arxiv")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "cora":
        data = CustomGraphDataset(root=path + "/cora")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "cora_ml":
        data = CustomGraphDataset(root=path + "/cora_ml")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "citeseer":
        data = CustomGraphDataset(root=path + "/citeseer")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "polblogs":
        data = CustomGraphDataset(root=path + "/polblogs")[0].to(device)
        data.x = torch.Tensor(data.x)
        data.x = data.x.to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "twitter":
        data = CustomGraphDataset(root=path + "/twitter")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "citation2":
        data = CustomGraphDataset(root=path + "/citation2")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "gplus":
        data = CustomGraphDataset(root=path + "/gplus")[0].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")
    elif data_name == "synthetic":
        tmp = torch.load(f"{path}/synthetic/synthetic.pt")
        data = DataSyn()
        data.x = tmp["x"].to(device)
        data.edge_index = tmp["edge_index"].to(device)
    elif data_name == "wiki":
        tmp = torch.load(f"{path}/wiki/wiki.pt")
        data = DataWiki()
        data.x = tmp["x"].to(device)
        data.edge_index = tmp["edge_index"].to(device)
        print(f"NUMBER OF NODES: {data.x.size(0)}")
        print(f"NUMBER OF EDGES: {data.edge_index.size(-1)}")

    return data


def initialize_adj_fn(adj, max_fn, device):
    adj_fn = torch.zeros((
        adj.size(0) + max_fn, adj.size(1) + max_fn
    ), device=device)
    # fill the original part of the matrix
    adj_fn[: adj.size(0), : adj.size(0)] = adj

    return adj_fn


def initialize_P(source, target, adj_fn, max_fn, device, random=False):

    P = torch.randn((max_fn, adj_fn.size(0)), device=device)
    # P = torch.zeros((max_fn, adj_fn.size(0)), device=device)
    # P += (torch.randn(P.size(), device=device) / 100)
    if not random:
        P[:, source] = 10
        P[:, target] = 10

    P_ori = P[:, :-max_fn]
    P_nori = P[:, -max_fn:]

    return P_ori, P_nori


def initialize_P_ablation(source, target, adj_fn, max_fn, device, init_P):

    if init_P == "ones":
        P = torch.ones((max_fn, adj_fn.size(0)), device=device)
        P += (torch.randn(P.size(), device=device) / 100)
    elif init_P == "zeros":
        P = torch.zeros((max_fn, adj_fn.size(0)), device=device)
        P += (torch.randn(P.size(), device=device) / 100)
    elif init_P == "random":
        P = torch.randn((max_fn, adj_fn.size(0)), device=device)
    elif init_P == "neg_ones":
        P = torch.ones((max_fn, adj_fn.size(0)), device=device)
        P *= -1
        P += (torch.randn(P.size(), device=device) / 100)

    if P == None:
        raise NotImplementedError

    P_ori = P[:, :-max_fn]
    P_nori = P[:, -max_fn:]

    return P_ori, P_nori



def import_initialized_P(cfg, exp, device):

    iga_init = torch.load(
        f"../support_materials/iga_init/iga_results_"
        f"{cfg.data_name}_{cfg.max_fn}_{cfg.num_exp}_{cfg.X_existent}"
    )

    P = iga_init[exp] * 10
    P += torch.randn(P.size()) / 10
    P = P.to(device)

    P_ori = P[:, :-cfg.max_fn]
    P_nori = P[:, -cfg.max_fn:]

    return P_ori, P_nori





def initialize_P_fn(max_fn, device):
    P_fn = torch.ones(max_fn, device=device) - 0.1

    return P_fn

def initialize_P_fn_ablation(max_fn, device, init_P_fn):

    if init_P_fn == "ones":
        P_fn = torch.ones(max_fn, device=device)
        P_fn -= torch.abs(torch.randn(P_fn.size(), device=device) / 100)
    elif init_P_fn == "zeros":
        P_fn = torch.zeros(max_fn, device=device) - 0.1
        P_fn += torch.abs(torch.randn(P_fn.size(), device=device) / 100)
    elif init_P_fn == "random":
        P_fn = torch.rand(max_fn, device=device)

    if P_fn == None:
        raise NotImplementedError

    return P_fn

def initialize_X_full(X, max_fn, device, existent=False):
    # initialize new features (this should be studied more)
    if not existent:
        X_fn = torch.mean(X).expand(
            max_fn, X.size(-1)) + torch.randn((max_fn, X.size(-1)), device=device)
    else:
        idx = torch.randperm(X.size(-2))[:max_fn]
        X_fn = X[idx]
    # X_fn = torch.randn(max_fn, data.x.size(-1), device=device)
    # X_fn = torch.zeros(max_fn, data.x.size(-1), device=device)
    X_full = torch.cat((X, X_fn))

    return X_full


def initialize_X_full_ablation(X, max_fn, device, init_X):

    X_tmp = X.clone()
    X_fn = torch.empty((max_fn, X_tmp.size(-1)), device=device)

    if init_X == "existent":
        idx = torch.randperm(X_tmp.size(-2))[:max_fn]
        X_fn = X_tmp[idx]
        X_fn += torch.randn(X_fn.size(), device=device) / 10
    elif init_X == "random":
        X_fn = torch.randn(X_fn.size(), device=device)
    elif init_X == "mean":
        X_fn = torch.mean(X_tmp).expand(X_fn.size()).clone()
        X_fn += torch.randn(X_fn.size(), device=device) / 10
    elif init_X == "median":
        X_fn = torch.median(X_tmp).expand(X_fn.size()).clone()
        X_fn += torch.randn(X_fn.size(), device=device) / 10
    elif init_X == "ones":
        X_fn = torch.ones(X_fn.size(), device=device)
        X_fn += torch.randn(X_fn.size(), device=device) / 10
    elif init_X == "zeros":
        X_fn = torch.zeros(X_fn.size(), device=device)
        X_fn += torch.randn(X_fn.size(), device=device) / 10

    X_full = torch.cat((X, X_fn))

    return X_full





def make_pred(model, X, adj, source, target, device):
    with torch.no_grad():
        # initial prediction
        z = model(X, adj)
        out = model.decode(
            z.squeeze(0), torch.tensor([[target], [source]]).long().to(device)
        ).view(-1)
        pred = torch.sigmoid(out)

    return pred


def compute_transferability(G, source, target, mode):

    if mode == "common_neighbour":
        metric = len(list(networkx.common_neighbors(G, source, target)))
        degree_source = G.degree(source)
        degree_target = G.degree(target)
        if metric != 0:
            metric = metric / (degree_source + degree_target)

    if mode == "jaccard":
        metric = list(jaccard_coefficient(G, [(source, target)]))[-1][-1]

    if mode == "pref_attach":
        metric = list(preferential_attachment(G, [(source, target)]))[-1][-1]

    if mode == "adamic":
        metric = list(adamic_adar_index(G, [(source, target)]))[-1][-1]

    if mode == "resource":
        metric = list(resource_allocation_index(G, [(source, target)]))[-1][-1]

    if mode == "katz":
        metric = katz(G, source, target)

    if mode == "pagerank":
        metric = 0

        personalization = {n: 0.0 for n in list(G)}
        personalization[source] = 1.0
        metric += pagerank(G, personalization=personalization)[target]

        personalization = {n: 0.0 for n in list(G)}
        personalization[target] = 1.0
        metric += pagerank(G, personalization=personalization)[source]

    if mode == "simrank":
        metric = simrank_similarity(G, source, target)

    return metric


def katz(G, source, target, l=5, beta=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj = adjacency_matrix(G).todense()
    adj = torch.tensor(adj).to(device).float()
    l2a = dict()
    adj_t = adj.clone()
    for l in range(l):
        adj_tp1 = torch.matmul(adj, adj_t)
        l2a[l] = adj_tp1
        adj_t = adj_tp1.clone()
    # l2a = {l: adj ** l for l in range(1, l + 1)}
    katz_index = sum((beta ** l) * (l2a[l][source, target] + l2a[l][target, source]) for _ in range(l))
    katz_index = katz_index.cpu().item()

    return katz_index


if __name__ == "__main__":
    check_degree(3, 10, torch.ones(20, 20))
