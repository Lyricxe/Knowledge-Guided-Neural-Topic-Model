import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# seeds
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


class Knowledge_Guided_NTM(nn.Module):
    def __init__(self, bow_dim=1000, n_topic=20, bert_f_dim=768, h_dim=256, seed_topic_distribution=None,
                 has_trans=True):
        super(Knowledge_Guided_NTM, self).__init__()
        self.has_trans = has_trans
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.self_thought_theta = nn.Parameter((torch.ones(self.n_topic) / self.n_topic).unsqueeze(0), requires_grad=False)

        self.regular_topic_dis_trans = nn.Parameter(torch.ones(self.n_topic, self.bow_dim), requires_grad=True)
        self.regular_topic_dis_comm = nn.Parameter(torch.ones(self.n_topic, self.bow_dim), requires_grad=True)

        nn.init.uniform_(self.regular_topic_dis_trans)
        nn.init.uniform_(self.regular_topic_dis_comm)
        self.seed_topic_dis = nn.Parameter(torch.ones(self.n_topic, self.bow_dim),
                                           requires_grad=True)
        self.seed_topic_dis.data = seed_topic_distribution
        self.trans_pi = nn.Parameter(torch.ones(self.n_topic), requires_grad=True)
        self.comm_pi = nn.Parameter(torch.ones(self.n_topic), requires_grad=True)

        self.bert_f_encoder = nn.Sequential(
            nn.Linear(bert_f_dim, 512, bias=False), nn.ReLU(),
            nn.Linear(512, h_dim, bias=False), nn.ReLU())
        self.bow_f_encoder = nn.Sequential(
            nn.Linear(bow_dim, 1024, bias=False), nn.ReLU(),
            nn.Linear(1024, h_dim, bias=False), nn.ReLU())

        self.con_bert_bow_f_encoder = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim), nn.ReLU())

        self.image_f_encoder = nn.Sequential(
            nn.Linear(512, h_dim), nn.ReLU())

        self.motion_f_encoder = nn.Sequential(
            nn.Linear(512, h_dim), nn.ReLU())

        self.aural_f_encoder = nn.Sequential(
            nn.Linear(768, h_dim), nn.ReLU())

        data_type = 5 if self.has_trans else 4
        self.con_fea_encoder = nn.Sequential(
            nn.Linear(data_type * h_dim, 2 * h_dim), nn.ReLU(),
            nn.Linear(2 * h_dim, h_dim), nn.ReLU())

        self.fc_mu = nn.Linear(h_dim, n_topic)
        self.fc_logvar = nn.Linear(h_dim, n_topic)

        self.fc_sub_q = nn.Sequential(nn.Linear(h_dim, 512), nn.ReLU())
        self.fc_sub_k = nn.Sequential(nn.Linear(2 * bow_dim + 1, 512), nn.ReLU())
        self.fc_regular_topic = nn.Sequential(nn.Linear(bow_dim, 512), nn.ReLU())
        self.fc_seed_topic = nn.Sequential(nn.Linear(bow_dim, 512), nn.ReLU())

        self.fc_pi = nn.Sequential(nn.Linear(2 * 512, 512), nn.ReLU(), nn.Linear(512, 1), nn.ReLU())
        self.fc_eta = nn.Sequential(nn.Linear(h_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.fc_classifier = nn.Linear(n_topic, 2)

    def set_pi(self, weight):
        self.trans_pi = nn.Parameter(self.trans_pi + weight, requires_grad=False)
        self.comm_pi = nn.Parameter(self.comm_pi + weight, requires_grad=False)

    def set_regular_topic_prior_dis(self, regular_topic_dis_trans, regular_topic_dis_comm):
        self.regular_topic_dis_trans.data = regular_topic_dis_trans
        self.regular_topic_dis_comm.data = regular_topic_dis_comm

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def reparameterize_gumbel(self, s):
        _s = 1 - s
        s_s = torch.cat((s.unsqueeze(2), _s.unsqueeze(2)), dim=2)
        sub_topic_idx = F.gumbel_softmax(s_s, tau=0.1, hard=True)[:, :, 0:1].squeeze()
        return sub_topic_idx

    def get_trans_topic_word_dist(self, device, normalize=True):
        self.eval()
        pi = torch.sigmoid(self.trans_pi).unsqueeze(1)
        global_topic_dis = pi * self.regular_topic_dis_trans + (1 - pi) * self.seed_topic_dis
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(device)
            word_dist = torch.mm(idxes, global_topic_dis)
            if normalize:
                word_dist = F.softmax(word_dist, dim=1)
            return word_dist.detach().cpu()

    def get_comm_topic_word_dist(self, device, normalize=True):
        self.eval()
        pi = self.generate_topic_indicator()
        global_topic_dis = pi * self.regular_topic_dis_comm + (1 - pi) * self.seed_topic_dis
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(device)
            word_dist = torch.mm(idxes, global_topic_dis)
            if normalize:
                word_dist = F.softmax(word_dist, dim=1)
            return word_dist.detach().cpu()

    def generate_sub_indicator(self, theta, trans_bert_fea):
        repeated_seed_topic = self.seed_topic_dis.unsqueeze(0).repeat(theta.shape[0], 1, 1)
        repeated_regular_topic = self.regular_topic_dis_trans.unsqueeze(0).repeat(theta.shape[0], 1, 1)
        repeated_topic = torch.cat((repeated_seed_topic, repeated_regular_topic), dim=2)
        thea_topic_fea = torch.cat((theta.unsqueeze(2), repeated_topic), dim=2)
        q = self.fc_sub_q(trans_bert_fea)
        k = self.fc_sub_k(thea_topic_fea)
        s = torch.cosine_similarity(q.unsqueeze(2), k.transpose(1, 2))
        s = torch.sigmoid(s)
        return s

    def generate_topic_indicator(self):
        regular_topic_dis = self.fc_regular_topic(self.regular_topic_dis_trans)
        seed_topic_dis = self.fc_seed_topic(self.seed_topic_dis)
        con_topic = torch.cat((regular_topic_dis, seed_topic_dis), dim=1)
        pi = self.fc_pi(con_topic)
        pi = torch.sigmoid(pi)
        self.pi = pi
        return pi

    def forward(self, trans_x, comm_x, trans_b_fea, comm_b_fea, image_fea, motion_fea, aural_fea):

        comm_bow_fea = self.bow_f_encoder(comm_x)
        comm_bert_fea = self.bert_f_encoder(comm_b_fea)
        comm_fea = self.con_bert_bow_f_encoder(torch.cat((comm_bow_fea, comm_bert_fea), dim=1))

        image_fea = self.image_f_encoder(image_fea)
        motion_fea = self.motion_f_encoder(motion_fea)
        aural_fea = self.aural_f_encoder(aural_fea)

        if self.has_trans:
            trans_bow_fea = self.bow_f_encoder(trans_x)
            trans_bert_fea = self.bert_f_encoder(trans_b_fea)
            trans_fea = self.con_bert_bow_f_encoder(torch.cat((trans_bow_fea, trans_bert_fea), dim=1))
            con_fea = torch.stack([trans_fea, comm_fea, image_fea, motion_fea, aural_fea], dim=1).view(image_fea.shape[0], -1)

        else:
            con_fea = torch.stack([comm_fea, image_fea, motion_fea, aural_fea], dim=1).view(image_fea.shape[0], -1)
        fea = self.con_fea_encoder(con_fea)
        mu, log_var = self.fc_mu(fea), self.fc_logvar(fea)
        _theta = self.reparameterize(mu, log_var)
        theta = F.softmax(_theta, dim=1)
        label_pred = self.fc_classifier(theta)
        trans_pi = torch.sigmoid(self.trans_pi).unsqueeze(1)
        comm_pi = torch.sigmoid(self.comm_pi).unsqueeze(1)
        eta = torch.sigmoid(self.fc_eta(comm_fea))

        topic_dis_trans = trans_pi * self.regular_topic_dis_trans + (1 - trans_pi) * self.seed_topic_dis
        topic_dis_comm = comm_pi * self.regular_topic_dis_comm + (1 - comm_pi) * self.seed_topic_dis

        if self.has_trans:
            s = self.generate_sub_indicator(theta, trans_bert_fea)
            sub_topic_idx = self.reparameterize_gumbel(s)

            theta_trans = theta * sub_topic_idx
            theta_trans = theta_trans / torch.sum(theta_trans, dim=1).unsqueeze(1)

            trans_recon = F.relu(torch.mm(theta_trans, topic_dis_trans))

            video_topic_recon = torch.mm(theta, topic_dis_comm)
            self_thought_recon = torch.mm(self.self_thought_theta, topic_dis_comm)
            comm_recon = F.relu(torch.mul(eta, video_topic_recon) + torch.mm(1-eta, self_thought_recon))
            return trans_recon, comm_recon, torch.sigmoid(theta), mu, log_var, s, trans_pi, comm_pi,  _theta, label_pred

        else:
            video_topic_recon = torch.mm(theta, topic_dis_comm)
            self_thought_recon = torch.mm(self.self_thought_theta, topic_dis_comm)
            comm_recon = F.relu(torch.mul(eta, video_topic_recon) + torch.mm(1-eta, self_thought_recon))
            return comm_recon, torch.sigmoid(theta), mu, log_var, comm_pi, _theta, label_pred