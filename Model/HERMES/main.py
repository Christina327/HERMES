import time
import os
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.utils.data as Data
from einops import rearrange, repeat

from utils import set_seed_all, get_metrics, TruncatedExponentialLR
from process_data import getData
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from drug_util import GraphDataset, collate
from HERMES.model import Initializer, HERMERS, Refiner, Consolidator

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 1
THRESHOLD = 30
CV_RATIO = 0.9


def load_data(dataset, use_disease=False):
    drug_fea, drug_smiles_fea, cline_fea, gene_data, synergy, *extra = \
        getData(dataset, use_disease)
    cline_fea = torch.from_numpy(cline_fea).to(DEVICE)
    if use_disease:
        disease_bert_fea, interaction = extra
        disease_fea = torch.from_numpy(disease_bert_fea).to(DEVICE)
    for row in synergy:
        row[3] = 1 if row[3] >= THRESHOLD else 0

    if use_disease:
        sim_matrices = get_sim_mat(
            drug_smiles_fea, np.array(gene_data, dtype='float32'), disease_bert_fea)
    else:
        sim_matrices = get_sim_mat(
            drug_smiles_fea, np.array(gene_data, dtype='float32'))

    ret = drug_fea, cline_fea, synergy, sim_matrices
    if use_disease:
        ret += (disease_fea, interaction)

    return ret


def _split_dataframe_to_arrays(df, ratio, rd_seed):
    shuffled_df = df.sample(frac=1, random_state=rd_seed)
    shuffled_arr = np.array(shuffled_df)
    arr_train, arr_test = np.split(
        shuffled_arr, [int(ratio*len(shuffled_arr))])
    return arr_train, arr_test


def data_split(synergy, rd_seed):
    set_seed_all(rd_seed)

    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])

    synergy_cv_pos, synergy_test_pos = _split_dataframe_to_arrays(
        synergy_pos, CV_RATIO, rd_seed)
    synergy_cv_neg, synergy_test_neg = _split_dataframe_to_arrays(
        synergy_neg, CV_RATIO, rd_seed)

    data_cv_raw = np.concatenate(
        (np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    data_test = np.concatenate(
        (np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)

    np.random.seed(rd_seed)
    np.random.shuffle(data_cv_raw)
    np.random.seed(rd_seed)
    np.random.shuffle(data_test)
    print("    number of cross:", len(data_cv_raw))
    print("    number of test: ", len(data_test))
    tensor_test = torch.from_numpy(data_test).to(DEVICE)
    label_test = torch.from_numpy(
        np.array(data_test[:, 3], dtype='float32')).to(DEVICE)

    return data_cv_raw, tensor_test, label_test


def get_sim_mat(drug_smiles_fea, cline_fea, *args):
    drug_sim_matrix = get_Cosin_Similarity(drug_smiles_fea)
    cline_sim_matrix = get_pvalue_matrix(cline_fea)
    matrices_np = [drug_sim_matrix, cline_sim_matrix]
    if len(args) > 0:
        disease_fea = args[0]
        disease_sim_matrix = get_pvalue_matrix(disease_fea)
        matrices_np.append(disease_sim_matrix)
    matries = []
    for matrix in matrices_np:
        matrix = torch.from_numpy(matrix).float().to(DEVICE)
        matries.append(matrix)
    return matries


def train(batch_tensor, batch_label, alpha, loaders, use_disease=False):
    model.train()
    if swap:
        batch_label = torch.cat((batch_label, batch_label), dim=0)

    druga_id, drugb_id, cline_id, _ = batch_tensor.unbind(1)

    optimizer.zero_grad()
    loss_train = 0
    batch_label_ls, batch_pred_ls = [], []
    for item in zip(*loaders):
        if use_disease:
            drug, (cline, ), (disease, ) = item
            pred, rec_s = model(
                drug.x, drug.edge_index, drug.batch,
                cline,
                druga_id, drugb_id, cline_id,
                disease,
            )
            rec_drug, rec_cline, rec_disease = rec_s
        else:
            drug, (cline, ) = item
            pred, rec_s = model(
                drug.x, drug.edge_index, drug.batch,
                cline,
                druga_id, drugb_id, cline_id,
            )
            rec_drug, rec_cline = rec_s

        loss_target = ce_loss_fn(pred, batch_label)
        loss_aux = ce_loss_fn(rec_drug, drug_sim_mat) + \
            ce_loss_fn(rec_cline, cline_sim_mat)
        if use_disease:
            loss_aux += ce_loss_fn(rec_disease, disease_sim_mat)

        loss = loss_target + alpha * loss_aux

        loss.backward()

        optimizer.step()

        loss_train += loss.item()
        batch_label_ls += batch_label.cpu().detach().numpy().tolist()
        batch_pred_ls += pred.cpu().detach().numpy().tolist()

    return loss_train, batch_label_ls, batch_pred_ls


def test(batch_tensor, batch_label, alpha, loaders, use_disease=False):
    model.eval()

    druga_id, drugb_id, cline_id, _ = batch_tensor.unbind(1)
    with torch.no_grad():
        for item in zip(*loaders):
            if use_disease:
                drug, (cline,), (disease,) = item
                batch_pred, rec_s = model(
                    drug.x, drug.edge_index, drug.batch, cline,
                    druga_id, drugb_id, cline_id, disease)
                rec_drug, rec_cline, rec_disease = rec_s
            else:
                drug, (cline, ) = item
                batch_pred, rec_s = model(
                    drug.x, drug.edge_index, drug.batch, cline,
                    druga_id, drugb_id, cline_id)
                rec_drug, rec_cline = rec_s

        loss_target = ce_loss_fn(batch_pred, batch_label)

        loss_aux = ce_loss_fn(rec_drug, drug_sim_mat) + \
            ce_loss_fn(rec_cline, cline_sim_mat)
        if use_disease:
            loss_aux += ce_loss_fn(rec_disease, disease_sim_mat)
        loss = loss_target + alpha * loss_aux

        batch_label = batch_label.cpu().detach().numpy()
        batch_pred = batch_pred.cpu().detach().numpy()
        return loss.item(), batch_label, batch_pred


if __name__ == '__main__':
    start_time = time.time()

    swap = False
    use_disease = True
    use_attention = False
    heads = 4

    refine_in_dim, refine_out_dim = 256, 256
    max_epoch = 3000  # 2000 or 2500
    start_update_epoch = 999
    print_interval = 200

    num_split = 5
    dataset_name = 'ALMANAC'  # ONEIL or ALMANAC
    cv_mode_ls = [1, 2, 3, 4, 5]

    lr_decay = 1 - 3e-4
    min_lr = 2e-6
    learning_rate = 1e-4
    weight_decay = 1e-2
    alpha = 1e-2

    for cv_mode, in itertools.product(cv_mode_ls):
        set_seed_all(SEED)

        drug_feature, cline_feature, synergy_data, sim_matrices, *extra = \
            load_data(dataset_name, use_disease)
        if use_disease:
            drug_sim_mat, cline_sim_mat, disease_sim_mat = sim_matrices
        else:
            drug_sim_mat, cline_sim_mat = sim_matrices
        drug_loader = Data.DataLoader(
            dataset=GraphDataset(graphs_dict=drug_feature),
            collate_fn=collate,
            batch_size=len(drug_feature),
            shuffle=False
        )
        cline_loader = Data.DataLoader(
            dataset=Data.TensorDataset(cline_feature),
            batch_size=len(cline_feature),
            shuffle=False
        )
        loaders = [drug_loader, cline_loader]
        if use_disease:
            disease_feature, interaction_pairs = extra
            disease_loader = Data.DataLoader(
                dataset=Data.TensorDataset(disease_feature),
                batch_size=len(disease_feature),
                shuffle=False
            )
            loaders.append(disease_loader)

            interaction_pairs = np.array(interaction_pairs)
            H_int_node = rearrange(
                interaction_pairs, 'n doublet -> (n doublet)')
            H_int_edge = repeat(
                np.arange(len(interaction_pairs)), 'n -> (n 2)')
            H_int = np.stack((H_int_node, H_int_edge), axis=0)
            H_int = torch.from_numpy(H_int).long().to(DEVICE)

        else:
            H_int = None

        data_cv_raw, tensor_test, label_test = data_split(
            synergy_data, SEED)

        if cv_mode == 1:
            data_cv = data_cv_raw
        elif cv_mode == 2:
            data_cv = np.unique(data_cv_raw[:, 2])
        elif cv_mode == 3:
            drugcomb = np.column_stack((data_cv_raw[:, 0], data_cv_raw[:, 1]))
            drugcomb = [tuple(sorted(pair)) for pair in drugcomb]
            data_cv = np.unique(drugcomb, axis=0)
        elif cv_mode in [4, 5]:
            data_cv = np.unique(np.concatenate(
                [data_cv_raw[:, 0], data_cv_raw[:, 1]]))
        else:
            raise NotImplementedError

        final_valid_metric = np.zeros(2)
        final_test_metric = np.zeros(2)
        final_valid_metrics, final_test_metrics = [], []
        kf = KFold(n_splits=num_split, shuffle=True, random_state=SEED)
        for idx, (train_index, valid_index) in enumerate(kf.split(data_cv), start=1):
            if cv_mode == 1:
                synergy_train, synergy_valid = data_cv[train_index], data_cv[valid_index]
            elif cv_mode == 2:
                train_name, test_name = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    cline = row[2]
                    if cline in train_name:
                        synergy_train.append(row)
                    elif cline in test_name:
                        synergy_valid.append(row)
                    else:
                        raise ValueError
            elif cv_mode == 3:
                pair_train, pair_validation = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    drugcomb = sorted((row[0], row[1]))
                    if any(all(x == y for x, y in zip(drugcomb, pair)) for pair in pair_train):
                        synergy_train.append(row)
                    else:
                        synergy_valid.append(row)
            elif cv_mode == 4:
                train_name, test_name = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    druga, drugb = row[0], row[1]
                    if (druga in train_name) and (drugb in train_name):
                        synergy_train.append(row)
                    elif (druga in test_name) and (drugb in test_name):
                        pass
                    else:
                        synergy_valid.append(row)
            elif cv_mode == 5:
                train_name, test_name = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    druga, drugb = row[0], row[1]
                    if (druga in train_name) and (drugb in train_name):
                        synergy_train.append(row)
                    elif (druga in test_name) and (drugb in test_name):
                        synergy_valid.append(row)
                    else:
                        pass
            else:
                raise NotImplementedError

            synergy_train = np.array(synergy_train)
            synergy_valid = np.array(synergy_valid)
            print(f"split {idx}")
            print("    number of train:", len(synergy_train))
            print("    number of valid:", len(synergy_valid))

            tensor_train = torch.from_numpy(synergy_train).to(DEVICE)
            tensor_valid = torch.from_numpy(synergy_valid).to(DEVICE)

            label_train = torch.from_numpy(
                np.array(synergy_train[:, 3], dtype='float32')).to(DEVICE)
            label_valid = torch.from_numpy(
                np.array(synergy_valid[:, 3], dtype='float32')).to(DEVICE)

            pos_node = synergy_train[synergy_train[:, 3] == 1, 0:3]
            num_synergy = len(pos_node)
            H_syn_node = rearrange(pos_node, 'n triplet -> (n triplet)')
            H_syn_edge = repeat(np.arange(num_synergy), 'n -> (n 3)')
            H_syn = np.stack((H_syn_node, H_syn_edge), axis=0)
            H_syn = torch.from_numpy(H_syn).long().to(DEVICE)

            set_seed_all(SEED)
            model = HERMERS(
                Initializer(
                    drug_dim=75, cline_dim=cline_feature.shape[-1], out_dim=refine_in_dim),
                Refiner(in_dim=refine_in_dim, out_dim=refine_out_dim),
                Consolidator(in_dim=refine_out_dim),
                use_disease=use_disease,
            ).to(DEVICE)
            model.initialize(num_synergy=num_synergy, H_syn=H_syn, H_int=H_int)

            ce_loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = TruncatedExponentialLR(
                optimizer, gamma=lr_decay, min_lr=min_lr)

            best_metric = [0, 0]
            best_epoch = 0
            for epoch in range(max_epoch):
                train_label_ls, train_pred_ls = [], []
                for _ in range(1):
                    train_loss, batch_label_ls, batch_pred_ls = train(
                        tensor_train, label_train, alpha, loaders, use_disease)
                    train_label_ls.extend(batch_label_ls)
                    train_pred_ls.extend(batch_pred_ls)
                valid_label_ls, valid_pred_ls = [], []
                for _ in range(1):
                    valid_loss, batch_label_ls, batch_pred_ls = test(
                        tensor_valid, label_valid, alpha, loaders, use_disease)
                    valid_label_ls.extend(batch_label_ls)
                    valid_pred_ls.extend(batch_pred_ls)
                scheduler.step()

                if epoch >= start_update_epoch:
                    train_metric = get_metrics(train_label_ls, train_pred_ls)
                    valid_metric = get_metrics(valid_label_ls, valid_pred_ls)

                    torch.save(model.state_dict(), f'{epoch}.pth')
                    valid_auc, best_auc = valid_metric[0], best_metric[0]
                    if valid_auc > best_auc:
                        best_metric = valid_metric
                        best_epoch = epoch
                    files = glob.glob('*.pth')
                    for f in files:
                        epoch_nb = int(f.split('.')[0])
                        if epoch_nb < best_epoch:
                            os.remove(f)

                    if (epoch + 1) % print_interval == 0:
                        print(f'Epoch: {epoch:04d},',
                              f'loss_train: {train_loss:.4f},',
                              f'AUC: {train_metric[0]:.4f},',
                              f'AUPR: {train_metric[1]:.4f},',
                              )
                        print(f'Epoch: {epoch:04d},',
                              f'loss_valid: {valid_loss:.4f},',
                              f'AUC: {valid_metric[0]:.4f},',
                              f'AUPR: {valid_metric[1]:.4f},',
                              )
                        print("-" * 71)

            files = glob.glob('*.pth')
            for f in files:
                epoch_nb = int(f.split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(f)
            print('The best results on valid set,',
                  f'Epoch: {best_epoch:04d},',
                  f'AUC: {best_metric[0]:.4f},',
                  f'AUPR: {best_metric[1]:.4f},',
                  )

            model.load_state_dict(torch.load(f'{best_epoch}.pth'))
            valid_label_ls, valid_pred_ls = [], []
            for _ in range(1):
                _, batch_label_ls, batch_pred_ls = test(
                    tensor_valid, label_valid, alpha, loaders, use_disease)
                valid_label_ls.extend(batch_label_ls)
                valid_pred_ls.extend(batch_pred_ls)
            valid_metric = get_metrics(valid_label_ls, valid_pred_ls)
            test_label_ls, test_pred_ls = [], []
            for _ in range(1):
                _, batch_label_ls, batch_pred_ls = test(
                    tensor_test, label_test, alpha, loaders, use_disease)
                test_label_ls.extend(batch_label_ls)
                test_pred_ls.extend(batch_pred_ls)
            test_metric = get_metrics(test_label_ls, test_pred_ls)

            final_valid_metric += valid_metric
            final_test_metric += test_metric

            final_valid_metrics.append(valid_metric)
            final_test_metrics.append(test_metric)

        pd.DataFrame(final_valid_metrics).to_csv(f'mode_{cv_mode}.csv')

        print("-" * 71)
        for i, metrics in enumerate(final_valid_metrics, 1):
            print(f'{i}-th valid results,',
                  f'AUC: {metrics[0]:.4f},',
                  f'AUPR: {metrics[1]:.4f},',
                  )
        print("-" * 71)

        final_valid_metric /= idx
        print('Final 5-cv valid results,',
              f'AUC: {final_valid_metric[0]:.4f},',
              f'AUPR: {final_valid_metric[1]:.4f},',
              )

        final_test_metric /= idx
        print('Final 5-cv test results,',
              f'AUC: {final_test_metric[0]:.4f},',
              f'AUPR: {final_test_metric[1]:.4f},',
              )

        end_time = time.time()

        elapsed_time = end_time - start_time

        elapsed_time = int(elapsed_time)
        print(
            f"Program execution time: {elapsed_time//60} min {elapsed_time%60} sec")
        print("=" * 71)
        print()
