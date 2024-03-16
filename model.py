import torch
import torch.nn as nn
from function import ReverseLayerF


class CDCOR(nn.Module):
    def __init__(self, input_dim_user, input_dim_s_item, input_dim_t_item, latent_dim):
        """
        s_n_item: number of item in source domain
        t_n_item: number of item in target domain
        n_user: number of user
        input_dim_user: input dim of user
        input_dim_s_item: input dim of source item
        input_dim_t_item: input dim of target item
        latent_dim: latent dim
        """

        super(CDCOR, self).__init__()
        self.input_dim_s_item = input_dim_s_item
        self.input_dim_t_item = input_dim_t_item
        self.input_dim_user = input_dim_user
        self.latent_dim = latent_dim

        self.s_user2latent = nn.Embedding(self.input_dim_user, self.latent_dim)
        self.t_user2latent = nn.Embedding(self.input_dim_user, self.latent_dim)

        self.embed_s_user = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU()
        )
        self.embed_t_user = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU()
        )

        self.embed_s_item = nn.Embedding(self.input_dim_s_item, self.latent_dim)
        self.embed_t_item = nn.Embedding(self.input_dim_t_item, self.latent_dim)

        self.adj_matrix = nn.Parameter(torch.rand(2 * self.latent_dim, 2 * self.latent_dim),
                                       requires_grad=True)

        self.embed_c_user = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Sigmoid(),
            nn.Linear(self.latent_dim // 2, 2)
        )

        self.softmax = nn.Softmax(dim=1)

        self.s_trans = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.t_trans = nn.Linear(2 * self.latent_dim, self.latent_dim)

        self.s_predict = nn.Linear(self.latent_dim, 2)
        self.t_predict = nn.Linear(self.latent_dim, 2)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.adj_matrix, std=0.15)

    def threshold(self):
        with torch.no_grad():
            for i in range(0, 2 * self.latent_dim):
                for j in range(0, 2 * self.latent_dim):
                    if -1e-04 < self.adj_matrix[i, j] < 1e-04:
                        self.adj_matrix[i, j] = 0

    def forward(self, s_u, t_u, s_i, t_i):
        self.threshold()
        s_user_attr = self.s_user2latent(s_u)
        s_user_emb = self.embed_s_user(s_user_attr)
        s_item_emb = self.embed_s_item(s_i)
        s_user_c_emb = self.embed_c_user(s_user_attr)
        s_input = torch.cat((s_user_attr, s_user_c_emb), dim=1)
        s_causal = torch.mm(s_input, self.adj_matrix)
        s_pref = s_causal[:, -self.latent_dim:]

        s_user_f = torch.cat((s_user_emb, s_pref), dim=1)
        s_user = self.s_trans(s_user_f)

        s_prediction = self.s_predict(s_user * s_item_emb)
        s_reversed = ReverseLayerF.apply(s_user_c_emb, 1.)
        s_class = self.classifier(s_reversed)

        t_user_attr = self.t_user2latent(t_u)
        t_user_emb = self.embed_t_user(t_user_attr)
        t_item_emb = self.embed_t_item(t_i)
        t_user_c_emb = self.embed_c_user(t_user_attr)
        t_input = torch.cat((t_user_attr, t_user_c_emb), dim=1)
        t_causal = torch.mm(t_input, self.adj_matrix)
        t_pref = t_causal[:, -self.latent_dim:]

        t_user_f = torch.cat((t_user_emb, t_pref), dim=1)
        t_user = self.t_trans(t_user_f)

        t_prediction = self.t_predict(t_user * t_item_emb)
        t_reversed = ReverseLayerF.apply(t_user_c_emb, 1.)
        t_class = self.classifier(t_reversed)

        return s_prediction, t_prediction, s_class, t_class, s_input, t_input, \
            s_causal, t_causal, self.adj_matrix

    def preforward(self, s_u, t_u, s_i, t_i):
        self.threshold()
        s_user_attr = self.s_user2latent(s_u)
        s_user_emb = self.embed_s_user(s_user_attr)
        s_item_emb = self.embed_s_item(s_i)
        s_user_c_emb = self.embed_c_user(s_user_attr)
        s_input = torch.cat((s_user_attr, s_user_c_emb), dim=1)
        s_causal = torch.mm(s_input, self.adj_matrix)
        s_pref = s_causal[:, -self.latent_dim:]

        s_user_f = torch.cat((s_user_emb, s_pref), dim=1)
        s_user = self.s_trans(s_user_f)

        s_prediction = self.s_predict(s_user * s_item_emb)
        s_class = self.classifier(s_user_c_emb)

        t_user_attr = self.t_user2latent(t_u)
        t_user_emb = self.embed_t_user(t_user_attr)
        t_item_emb = self.embed_t_item(t_i)
        t_user_c_emb = self.embed_c_user(t_user_attr)
        t_input = torch.cat((t_user_attr, t_user_c_emb), dim=1)
        t_causal = torch.mm(t_input, self.adj_matrix)
        t_pref = t_causal[:, -self.latent_dim:]

        t_user_f = torch.cat((t_user_emb, t_pref), dim=1)
        t_user = self.t_trans(t_user_f)

        t_prediction = self.t_predict(t_user * t_item_emb)
        t_class = self.classifier(t_user_c_emb)

        return s_prediction, t_prediction, s_class, t_class, s_input, t_input, \
            s_causal, t_causal, self.adj_matrix

    def predict(self, t_u, t_i):
        t_user_attr = self.t_user2latent(t_u)
        t_user_emb = self.embed_t_user(t_user_attr)
        t_item_emb = self.embed_t_item(t_i)
        t_user_c_emb = torch.zeros_like(t_user_attr)
        t_input = torch.cat((t_user_attr, t_user_c_emb), dim=1)
        t_causal = torch.mm(t_input, self.adj_matrix)
        t_pref = t_causal[:, -self.latent_dim:]

        t_user_f = torch.cat((t_user_emb, t_pref), dim=1)
        t_user = self.t_trans(t_user_f)

        t_prediction = self.t_predict(t_user * t_item_emb)

        return t_prediction
