import os
import time
from argparse import ArgumentParser
from distutils.util import strtobool

import torch
from pytorch_lightning import seed_everything
from torch import nn
from torch_geometric.utils import to_dense_adj
from SAVAGE.Model.model import Model
from SAVAGE.Utils.utils import build_data, initialize_adj_fn, initialize_P, initialize_P_fn, initialize_X_full, \
    import_initialized_P


# TODO: should I create a link between source --> vicious nodes?


def attack(cfg):

    seed_everything(0)

    print(cfg)

    start = time.time()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_data(cfg.root_data, cfg.data_name)

    adj = to_dense_adj(data.edge_index).squeeze(0).to(device)
    assert (adj.size(-1) == data.x.size(-2))

    # load the model
    model = Model(data.x.shape[-1], 128, 64).to(device)
    model.load_state_dict(torch.load(f"../support_materials/models/{cfg.data_name}_lp_directed.pt"))

    # load pairs
    pairs = torch.load(f"../support_materials/pairs/pairs_{cfg.data_name}.pt")
    sources, targets = pairs["source"], pairs["target"]

    # BCELoss creates a criterion that measures the Binary Cross Entropy between the target and the output.
    criterion = torch.nn.BCEWithLogitsLoss()

    # initialize augmented matrix (with vicious nodes)
    adj_fn = initialize_adj_fn(adj, cfg.max_fn, device)
    X_full = initialize_X_full(data.x, cfg.max_fn, device, cfg.X_existent)

    attack_rate = 0
    mean_final_pred = 0
    mean_added_nodes = 0
    mean_added_edges = 0
    mean_original_pred = 0
    for exp in range(cfg.num_exp):

        source = sources[exp].int()
        target = targets[exp].int()

        print(f"SOURCE: {source}, TARGET: {target}")
        print(f"Starting SOURCE outcoming edges: {adj[source].sum()}")
        print(f"Starting SOURCE incoming edges: {adj[:, source].sum()}")
        print(f"Starting TARGET outcoming edges: {adj[target].sum()}")
        print(f"Starting TARGET incoming edges: {adj[:, target].sum()}")

        with torch.no_grad():
            # initial prediction
            z = model(data.x, adj)
            out = model.decode(
                z.squeeze(0), torch.Tensor([[target], [source]]).long().to(device)
            ).view(-1)
            print(f"INITIAL PREDICTION BEFORE OPTIMIZATION IS: {torch.sigmoid(out)}")
            mean_original_pred += torch.sigmoid(out).detach().item()

        # TODO: is this the correct way of initializing it (range of values)
        # initialize perturbation matrix
        if cfg.import_init:
            P_ori, P_nori = import_initialized_P(cfg, exp, device)
            print(f"LOADED IGA INITALIZED P")
        else:
            P_ori, P_nori = initialize_P(source, target, adj_fn, cfg.max_fn, device)
        P_ori = nn.Parameter(P_ori)
        P_nori = nn.Parameter(P_nori)

        # initialize mask for fake nodes
        P_fn = initialize_P_fn(cfg.max_fn, device)
        P_fn = nn.Parameter(P_fn)

        # initialize optimizer with both P and P_fn
        optimizer = torch.optim.Adam(params=[P_ori, P_nori, P_fn], lr=0.1)

        for step in range(cfg.n_steps):

            P_ori_tr = torch.tanh(P_ori)
            P_nori_tr = torch.tanh(P_nori)

            P_fn_tr = torch.clamp(P_fn, min=0, max=1)

            P_ori_tr_masked = P_fn_tr.unsqueeze(1) * P_ori_tr
            P_nori_tr_masked = P_fn_tr.unsqueeze(1) * P_nori_tr
            P_nori_tr_masked = P_nori_tr_masked * P_fn_tr.unsqueeze(0)

            P_tr_masked = torch.cat((P_ori_tr_masked, P_nori_tr_masked), dim=-1)

            A_tilde = adj_fn.clone()

            # update with perturbation matrix (fake nodes)
            A_tilde[-cfg.max_fn:, :] += P_tr_masked
            # erase inactive fake nodes
            A_tilde[source, -cfg.max_fn:] += P_fn_tr
            # turn values into 0,1 range
            A_tilde_tr = torch.clamp(A_tilde, min=0, max=1)

            # compute updated prediction
            z = model(X_full, A_tilde_tr)
            out = model.decode(
                z.squeeze(0), torch.Tensor([[target], [source]]).long().to(device)
            ).view(-1)

            # compute losses
            loss_lik = criterion(out, torch.Tensor([1]).float().to(device))
            loss_pen = torch.mean(
                torch.abs(
                    A_tilde_tr[-cfg.max_fn:, :].view(-1)
                    - adj_fn[-cfg.max_fn:, :].view(-1))
            )
            loss_fn = torch.mean(torch.abs(P_fn_tr))

            loss = loss_lik + (cfg.pen_edg * loss_pen) + (cfg.pen_fn * loss_fn)

            # print( f"LOSS at step: {i} is {loss_lik.item()=} | {loss_pen.item()=} | {loss_fn.item()=}, prediction
            # is {F.sigmoid(out).item()}, sum(P) is {torch.tanh(P).sum()}" )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"LOSS:\n LIK IS: {loss_lik}\n LOSS PEN IS: {loss_pen}\n LOSS FN IS {loss_fn}")

        # prediction part
        with torch.no_grad():

            P_ori_tr = torch.tanh(P_ori)
            P_nori_tr = torch.tanh(P_nori)

            P_fn_tr = torch.clamp(P_fn, min=0, max=1)
            # maybe this can be done later?
            P_fn_tr = torch.round(P_fn_tr)

            P_ori_tr_masked = P_fn_tr.unsqueeze(1) * P_ori_tr
            P_nori_tr_masked = P_fn_tr.unsqueeze(1) * P_nori_tr
            P_nori_tr_masked = P_nori_tr_masked * P_fn_tr.unsqueeze(0)

            P_tr_masked = torch.cat((P_ori_tr_masked, P_nori_tr_masked), dim=-1)

            A_tilde = adj_fn.clone()
            A_tilde[-cfg.max_fn:, :] += P_tr_masked
            A_tilde[source, -cfg.max_fn:] += P_fn_tr
            # if True:
            #     A_tilde[source, -cfg.max_fn:] *= 0
            A_tilde_tr = torch.clamp(A_tilde, min=0, max=1)

            A_tilde_tr = torch.round(A_tilde_tr)

            z = model(X_full, A_tilde_tr)
            out = model.decode(
                z.squeeze(0), torch.Tensor([[target], [source]]).long().to(device)
            ).view(-1)

            threshold = 0.6
            pred = torch.sigmoid(out) > threshold
            print(
                f"FINAL PREDICTION IS: {pred.item(), torch.sigmoid(out).item()}, ADDED NODES: {torch.sum(P_fn_tr).item()}, "
                f"MODIFIED EDGES: {torch.sum(~(A_tilde_tr == adj_fn)).item()}"
            )

            attack_rate += pred.int()
            mean_final_pred += torch.sigmoid(out).detach().item()
            mean_added_nodes += torch.sum(P_fn_tr).detach().item()
            mean_added_edges += torch.sum(~(A_tilde_tr == adj_fn)).detach().item()

            print("\n\n ----------------------------------")

    print("FINAL RESULTS -------------------------------------------------------------------")
    print("")
    print(f"ATTACK RATE: {attack_rate / cfg.num_exp}")
    print(f"MEAN FINAL PRED: {mean_final_pred / cfg.num_exp}")
    print(f"MEAN ADDED NODES: {mean_added_nodes / cfg.num_exp}")
    print(f"MEAN ADDED EDGES: {mean_added_edges / cfg.num_exp}")
    print(f"ELAPSED TIME: {time.time() - start}")

    os.makedirs("../support_materials/results/ablations/pen_fn/", exist_ok=True)
    with open(f"../support_materials/results/ablations/pen_fn/"
              f"{cfg.data_name}_pen_fn_{cfg.pen_fn}_n_steps_{cfg.n_steps}.txt", "w") as f:
        f.write(f"ATTACK RATE: {attack_rate / cfg.num_exp}")
        f.write("\n")
        f.write(f"MEAN FINAL PREDICTION IS: {mean_final_pred / cfg.num_exp}")
        f.write("\n")
        f.write(f"MEAN ADDED NODES: {mean_added_nodes / cfg.num_exp}")
        f.write("\n")
        f.write(f"MEAN ADDED EDGES: {mean_added_edges / cfg.num_exp}")
        f.write("\n")
        f.write(f"MEAN ORIGINAL PREDICTION IS: {mean_original_pred / cfg.num_exp}")




if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_name", default="synthetic", type=str)
    parser.add_argument("--pen_edg", default=0.1, type=float)
    parser.add_argument("--pen_fn", default=0.1, type=float)
    parser.add_argument("--max_fn", default=50, type=int)
    parser.add_argument("--num_exp", default=20, type=int)
    parser.add_argument("--n_steps", default=25, type=int)
    # parser.add_argument("--n_steps", default=200, type=int)
    parser.add_argument("--root_data", default="../support_materials/dataset", type=str)
    parser.add_argument("--X_existent", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--import_init", default=False, type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()

    attack(args)




