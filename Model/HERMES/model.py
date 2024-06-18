import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from utils import reset, LambdaLayer

DISEASE_DIM = 768


class Initializer(nn.Module):
    def __init__(self, drug_dim, cline_dim, out_dim, use_disease=False, use_GMP=False):
        super(Initializer, self).__init__()

        self.out_dim = out_dim
        self.enc_hdn_layers = 2

        self.use_disease = use_disease
        if self.use_disease:
            self.set_disease_parameter()

        drug_heads = 4
        self.drug_same_layers = self.enc_hdn_layers
        self.drug_first = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(drug_dim, out_dim//drug_heads,
             heads=drug_heads, root_weight=False), 'x, edge_index -> x'),
            (nn.ReLU(), 'x -> x'),
            (nn.BatchNorm1d(self.out_dim), 'x -> x'),
        ])
        self.drug_conv_same = nn.ModuleList([gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(self.out_dim, self.out_dim//drug_heads,
             heads=drug_heads, root_weight=False), 'x, edge_index -> x'),
            (nn.ReLU(), 'x -> x'),
            (nn.BatchNorm1d(self.out_dim), 'x -> x'),
        ]) for _ in range(self.drug_same_layers)])
        self.use_GMP = use_GMP

        self.cline_same_layers = self.enc_hdn_layers
        self.cline_first = nn.Sequential(
            nn.Linear(cline_dim, self.out_dim),
            nn.Tanh(),
        )
        self.cline_same = nn.ModuleList([nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) for _ in range(self.cline_same_layers)])

        self.reset_para()

    def set_use_disease(self):
        if not self.use_disease:
            self.use_disease = True
            self.set_disease_parameter()

    def set_disease_parameter(self):
        if not hasattr(self, 'disease_same_layers'):
            self.disease_same_layers = self.enc_hdn_layers
            self.disease_first = nn.Sequential(
                nn.Linear(DISEASE_DIM, self.out_dim, bias=True),
                nn.Tanh(),
            )
            self.disease_same = nn.ModuleList([nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(0.1),
            ) for _ in range(self.disease_same_layers)])

    def forward(self, drug_x, drug_adj, ibatch, cline_x, *args):
        drug_x = self.drug_first(drug_x, drug_adj)
        for i in range(self.drug_same_layers):
            drug_x = drug_x + self.drug_conv_same[i](drug_x, drug_adj)
        if self.use_GMP:
            drug_x = gnn.global_max_pool(drug_x, ibatch)
        else:
            drug_x = gnn.global_mean_pool(drug_x, ibatch)

        cline_x = self.cline_first(cline_x)
        for i in range(self.cline_same_layers):
            cline_x = cline_x + self.cline_same[i](cline_x)

        if self.use_disease:
            disease_x = args[0]
            disease_x = self.disease_first(disease_x)
            for i in range(self.disease_same_layers):
                disease_x = disease_x + self.disease_same[i](disease_x)
            return drug_x, cline_x, disease_x
        else:
            return drug_x, cline_x

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Refiner(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Refiner, self).__init__()
        self.out_dim = out_dim

        self.scalar = 1.0
        hidden_dim = int(out_dim * self.scalar)
        if in_dim == hidden_dim:
            self.lin_up = nn.Identity()
        else:
            self.lin_up = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=True),
                nn.ReLU(),
            )

        self.hdn_layers = 3

        self.conv_same = nn.ModuleList([gnn.Sequential('x, hyperedge_index, hyperedge_weight', [
            (nn.BatchNorm1d(hidden_dim), 'x -> x'),
            (gnn.HypergraphConv(hidden_dim, hidden_dim),
             'x, hyperedge_index, hyperedge_weight -> x'),
            (nn.ReLU(), 'x -> x'),
        ]) for _ in range(self.hdn_layers)])

        init_bias = 0.0
        self.conv_w_out = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid(),
        ) for _ in range(self.hdn_layers)])

        for layer in self.conv_w_out:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.constant_(sublayer.bias, init_bias)

        if hidden_dim == out_dim:
            self.lin_down = nn.Identity()

            self.conv_down = gnn.Sequential('x, hyperedge_index, hyperedge_weight', [
                (nn.Identity(), 'x -> x'),
            ])
        else:
            self.lin_down = nn.Sequential(
                nn.Linear(hidden_dim, out_dim, bias=True),
            )

            self.conv_down = gnn.Sequential('x, hyperedge_index, hyperedge_weight', [
                (nn.BatchNorm1d(hidden_dim), 'x -> x'),
                (gnn.HypergraphConv(hidden_dim, out_dim),
                 'x, hyperedge_index, hyperedge_weight -> x'),
                (nn.ReLU(), 'x -> x'),
            ])

    def forward(self, X, H, hyperedge_weight):
        X = self.lin_up(X)
        identity = X

        for i in range(self.hdn_layers):
            X = X + self.conv_same[i](X, H, hyperedge_weight) * \
                self.conv_w_out[i](X)

        X = self.lin_down(X + identity) + \
            self.conv_down(X, H, hyperedge_weight)

        return X


class Consolidator(torch.nn.Module):
    def __init__(self, in_dim):
        super(Consolidator, self).__init__()

        self.act = nn.ReLU()
        in_dim = in_dim * 3

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            self.act,
            nn.Dropout(0.20),
            nn.Linear(in_dim // 2, in_dim // 4),
            self.act,
            nn.Dropout(0.40),
            nn.Linear(in_dim // 4, 1),
            LambdaLayer(lambda x: x.squeeze()),
        )

        self.reset_parameters()

    def forward(self, graph_embed, druga_id, drugb_id, cline_id):
        druga = graph_embed[druga_id, :]
        drugb = graph_embed[drugb_id, :]
        cline = graph_embed[cline_id, :]
        if self.training:
            preds = self.forward_once(druga, drugb, cline)
        else:
            preds = self.forward_once(druga, drugb, cline)
        return preds

    def forward_once(self, druga, drugb, cline):
        cand_h = torch.cat((druga, drugb, cline), -1)
        logits = self.mlp(cand_h)
        return torch.sigmoid(logits)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class HERMERS(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder, use_disease=False):
        super(HERMERS, self).__init__()
        self.initializer: Initializer = bio_encoder
        self.refiner: Refiner = graph_encoder
        self.consolidator: Consolidator = decoder

        self.use_disease = use_disease
        if self.use_disease:
            self.initializer.set_use_disease()

        hgnn_out_dim = self.refiner.out_dim
        self.drug_rec_weight = nn.Parameter(
            torch.rand(hgnn_out_dim, hgnn_out_dim))
        self.cline_rec_weight = nn.Parameter(
            torch.rand(hgnn_out_dim, hgnn_out_dim))
        if self.initializer.use_disease:
            self.disease_rec_weight = nn.Parameter(
                torch.rand(hgnn_out_dim, hgnn_out_dim))
        self.sim_proj = nn.Identity()
        self.reset_parameters()
        self.register_buffer('initted', torch.tensor(False))

    def reset_parameters(self):
        reset(self.initializer)
        reset(self.refiner)
        reset(self.consolidator)

    def initialize(self, H_syn, H_int=None, num_synergy=0):
        if self.use_disease:
            syn_weight, int_weight = 1.0, 0.05
            num_interaction = int(H_int[1].max()) + 1
            print(
                f"num of synergy = {num_synergy}; num of interaction = {num_interaction}.")
            hyperedge_weight = torch.cat((
                torch.full((num_synergy, ), syn_weight),
                torch.full((num_interaction, ), int_weight)
            ), dim=0).to(H_int.device)

            H_int[1] += num_synergy
            H = torch.cat((H_syn, H_int), dim=1)
        else:
            hyperedge_weight = torch.ones(num_synergy).to(H_syn.device)
            H = H_syn

        self.register_buffer('H', H)
        self.register_buffer('hyperedge_weight', hyperedge_weight)

    def forward(self, drug_x, drug_adj, ibatch, cline_x, druga_id, drugb_id, cline_id, *args):
        if not self.initted:
            self.num_drug = max(ibatch).item() + 1
            self.num_cline = len(cline_x)
            self.initted.fill_(True)

        noise_scale = 0.0
        if not self.training and noise_scale:
            drug_x += torch.rand_like(drug_x) * noise_scale
            cline_x += torch.rand_like(cline_x) * noise_scale
            if self.use_disease:
                disease_feature = args[0]
                disease_feature += torch.rand_like(
                    disease_feature) * noise_scale

        if self.use_disease:
            disease_feature = args[0]
            drug_embed, cline_embed, disease_embd = self.initializer(
                drug_x, drug_adj, ibatch, cline_x, disease_feature)

            merge_embed = torch.cat(
                (drug_embed, cline_embed, disease_embd), dim=0)
        else:
            drug_embed, cline_embed = self.initializer(
                drug_x, drug_adj, ibatch, cline_x)

            merge_embed = torch.cat(
                (drug_embed, cline_embed), dim=0)

        graph_embed = self.refiner(
            merge_embed, self.H, self.hyperedge_weight)

        graph_embed = self.sim_proj(graph_embed)
        drug_emb = graph_embed[:self.num_drug]
        rec_drug = torch.sigmoid(torch.einsum(
            'i d, d e, j e -> i j', drug_emb, self.drug_rec_weight, drug_emb))

        cline_emb = graph_embed[self.num_drug: self.num_drug+self.num_cline]
        rec_cline = torch.sigmoid(torch.einsum(
            'i d, d e, j e -> i j', cline_emb, self.cline_rec_weight, cline_emb))

        if self.initializer.use_disease:
            disease_emb = graph_embed[self.num_drug+self.num_cline:]
            rec_disease = torch.sigmoid(torch.einsum(
                'i d, d e, j e -> i j', disease_emb, self.disease_rec_weight, disease_emb))
            rec_s = rec_drug, rec_cline, rec_disease
        else:
            rec_s = rec_drug, rec_cline

        pred = self.consolidator(graph_embed, druga_id, drugb_id, cline_id)
        return pred, rec_s
