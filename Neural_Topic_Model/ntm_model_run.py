import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .NTM_model.knowledge_guided_ntm import Knowledge_Guided_NTM
from dataset_tm import Dataset_TM, Dataset_General
from sklearn import metrics
from gensim.models.coherencemodel import CoherenceModel
import random

np.set_printoptions(suppress=True)


seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, model, model_nt, docSet, train_d, test_d, batch_size=128, n_epoch=800, lr=2e-3, has_trans=True):
        self.model = model
        self.model_nt = model_nt
        self.has_trans = has_trans
        self.n_epoch = n_epoch
        self.lr = lr
        self.train_dataloader = train_d
        self.test_dataloader = test_d
        self.batch_size = batch_size
        self.dictionary = docSet.dictionary
        self.id2word = docSet.id2token
        # self.docs = docSet.trans_docs + docSet.comments_docs
        self.docs = docSet.comments_docs
        self.seed_topics_idx = [0, 1, 2]
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def train_for_base_model(self):
        print("train base model")
        self.model.train().to(device)
        self.model.self_thought_theta.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1)
        for epoch in range(self.n_epoch):
            train_loss = 0
            cls_loss_tol = 0
            for data in self.train_dataloader:
                optimizer.zero_grad()
                data = [sub_data.to(device) for sub_data in data]
                trans_bow, trans_feature, comm_bow, comm_feature, image_fea, motion_fea, aural_fea, label = data
                label = label.type(torch.LongTensor).to(device)

                trans_bow_recon, comments_bow_recon, theta, mu, log_var, s, trans_pi, comm_pi, _theta, label_pred = \
                    self.model(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)
                # calculating trans loss
                trans_logSoftmax = torch.log_softmax(trans_bow_recon, dim=1)
                trans_recon_loss = -1.0 * torch.sum(trans_bow * trans_logSoftmax)
                # calculating comm loss
                comm_logSoftmax = torch.log_softmax(comments_bow_recon, dim=1)
                comm_recon_loss = -1.0 * torch.sum(comm_bow * comm_logSoftmax)
                # KL divergence
                kl_div_theta = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                kl_div_s = torch.sum(s*torch.log(s/theta)+(1-s)*torch.log((1-s)/(1-theta)))
                pi_prior = torch.ones_like(trans_pi).squeeze(1)*0.999
                pi_prior[self.seed_topics_idx] = 1e-6
                pi_prior = pi_prior.unsqueeze(1)
                kl_div_trans_pi_all = trans_pi*torch.log(trans_pi/pi_prior)+(1-trans_pi)*torch.log((1-trans_pi)/(1-pi_prior))
                kl_div_comm_pi_all = comm_pi*torch.log(comm_pi/pi_prior)+(1-comm_pi)*torch.log((1-comm_pi)/(1-pi_prior))
                kl_div_trans_pi_seed = torch.sum(kl_div_trans_pi_all[:3])
                kl_div_comm_pi_seed = torch.sum(kl_div_comm_pi_all[:3])
                # classifier loss
                cls_loss = self.criterion(label_pred, label)

                loss = trans_recon_loss + comm_recon_loss + kl_div_theta + kl_div_s + (kl_div_trans_pi_seed+kl_div_comm_pi_seed) + cls_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
                cls_loss_tol += cls_loss
            avg_loss = train_loss/len(self.train_dataloader.sampler)
            avg_cls_loss = cls_loss_tol/len(self.train_dataloader.sampler)
            scheduler.step()
            acc, precis, recall, f1 = self.test_classification()
            print(f'epoch:{epoch}, train_loss:{avg_loss}, cls_loss:{avg_cls_loss}, acc:{acc}, precis:{precis}, recall:{recall}, f1:{f1}')
            # topic quality evaluation
            if (epoch) % 100 == 0:
                # self.show_trans_topic_words()
                topic_words = self.show_comm_topic_words(top_n=20)
                # self.save_model(epoch=epoch)
                # print(torch.sigmoid(self.model.pi))
                coherence_top10, coherence_top20 = self.topic_evaluation(topic_words=topic_words)
                print(f'coherence_top 10 20:{coherence_top10}, {coherence_top20}')

        self.save_model(epoch=self.n_epoch)

    def train_for_model_nt(self):
        self.model.eval().to(device)
        self.model_nt.train().to(device)
        optimizer = torch.optim.Adam(self.model_nt.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)
        for epoch in range(self.n_epoch):
            train_loss = 0
            cls_loss_tol = 0
            for data in self.train_dataloader:
                optimizer.zero_grad()
                data = [sub_data.to(device) for sub_data in data]
                trans_bow, trans_feature, comm_bow, comm_feature, image_fea, motion_fea, aural_fea, label = data
                label = label.type(torch.LongTensor).to(device)
                _theta = self.model(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)[8]
                comm_bow_recon, theta, mu, log_var, pi, _theta_nt, label_pred = self.model_nt(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)
                comm_logSoftmax = torch.log_softmax(comm_bow_recon, dim=1)
                comm_recon_loss = -1.0 * torch.sum(comm_bow * comm_logSoftmax)

                kl_div_theta = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                trans_count = torch.sum(trans_bow, dim=1)
                sample_weight = torch.where(trans_count > 0, 1, 0)

                _theta_loss = torch.cosine_similarity(_theta, _theta_nt)
                _theta_loss = -1*torch.sum(_theta_loss*sample_weight)
                # initialize
                pi_prior = torch.ones_like(pi).squeeze(1)*0.99
                pi_prior[self.seed_topics_idx] = 1e-6
                pi_prior = pi_prior.unsqueeze(1)
                kl_div_pi_all = pi*torch.log(pi/pi_prior)+(1-pi)*torch.log((1-pi)/(1-pi_prior))
                kl_div_pi_seed = torch.sum(kl_div_pi_all[:3])
                cls_loss = self.criterion(label_pred, label)
                loss = comm_recon_loss + kl_div_theta + kl_div_pi_seed + cls_loss + _theta_loss

                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
                cls_loss_tol += cls_loss
            avg_loss = train_loss/len(self.train_dataloader.sampler)
            avg_cls_loss = cls_loss_tol/len(self.train_dataloader.sampler)
            scheduler.step()
            acc, precis, recall, f1 = self.test_classification()
            print(f'epoch:{epoch}, train_loss:{avg_loss}, cls_loss:{avg_cls_loss}, acc:{acc}, precis: {precis}, recall:{recall}, f1:{f1}')

            # topic quality evaluation
            if epoch > 500 and (epoch) % 10 == 0:
                self.show_trans_topic_words()
                topic_words = self.show_comm_topic_words(top_n=20)
                coherence_top10, coherence_top20 = self.topic_evaluation(topic_words=topic_words)
                print(f'coherence_top 10 20:{coherence_top10}, {coherence_top20}')
                self.save_model(epoch=epoch)
                print(torch.sigmoid(self.model.pi))
        self.save_model(epoch=epoch)

    def test_classification(self):
        self.model.eval().to(device)
        self.model_nt.eval().to(device)

        labels_all = np.array([], dtype=float)
        pred_all = np.array([], dtype=float)

        with torch.no_grad():
            for data in self.test_dataloader:
                data = [sub_data.to(device) for sub_data in data]
                trans_bow, trans_feature, comm_bow, comm_feature, image_fea, motion_fea, aural_fea, label = data
                label = label.type(torch.LongTensor).numpy()
                if self.has_trans:
                    pred = self.model(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)[9]

                else:
                    pred = self.model_nt(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)[6]

                pred = torch.softmax(pred.data, dim=1)
                pred = torch.where(pred[:, 1] > 0.5, 1, 0).cpu().numpy()
                labels_all = np.append(labels_all, label)
                pred_all = np.append(pred_all, pred)

        label_list = []
        pred_list = []
        for idx, label in enumerate(labels_all):
            label_list.append(label)
            pred_list.append(pred_all[idx])
        acc = metrics.accuracy_score(label_list, pred_list)
        f1 = metrics.f1_score(label_list, pred_list)
        recall = metrics.recall_score(label_list, pred_list)
        precis = metrics.precision_score(label_list, pred_list)
        return acc, precis, recall, f1

    def topic_evaluation(self, topic_words):
        coherence_model_top10 = CoherenceModel(topics=topic_words, texts=self.docs, dictionary=self.dictionary, coherence='u_mass', topn=10)
        coherence_model_top20 = CoherenceModel(topics=topic_words, texts=self.docs, dictionary=self.dictionary, coherence='u_mass', topn=20)
        coherence_top10 = coherence_model_top10.get_coherence()
        coherence_top20 = coherence_model_top20.get_coherence()
        return coherence_top10, coherence_top20

    def show_trans_topic_words(self, top_n=100):
        if self.has_trans:
            model = self.model.to(device)
        else:
            model = self.model_nt.to(device)
        topic_word_distribution = model.get_trans_topic_word_dist(device=device)
        _, top_words_idx = torch.topk(topic_word_distribution, k=top_n, dim=1)
        top_words_idx = top_words_idx.cpu().tolist()
        topic_words = []

        for topic_id, words_idx in enumerate(top_words_idx):
            words = [self.id2word[idx] for idx in words_idx]
            print(f'topic:{topic_id}{words}')
            topic_words.append(words)
        print(f'trans_pi:{torch.sigmoid(1-model.trans_pi)}')
        return topic_words

    def show_comm_topic_words(self, top_n=100):
        if self.has_trans:
            model = self.model.to(device)
        else:
            model = self.model_nt.to(device)
        topic_word_distribution = model.get_comm_topic_word_dist(device=device)
        _, top_words_idx = torch.topk(topic_word_distribution, k=top_n, dim=1)
        top_words_idx = top_words_idx.cpu().tolist()
        topic_words = []
        for topic_id, words_idx in enumerate(top_words_idx):
            words = [self.id2word[idx] for idx in words_idx]
            topic_words.append(words)
        return topic_words

    def inference_doc_topic(self):
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for idx, data in enumerate(self.test_dataloader):
                data = [sub_data.to(device) for sub_data in data]
                trans_bow, trans_feature, comm_bow, comm_feature, image_fea, motion_fea, aural_fea, label = data
                label = label.type(torch.LongTensor).numpy()
                if self.has_trans:
                    theta = self.model(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)[2]
                else:
                    theta = self.model_nt(trans_bow, comm_bow, trans_feature, comm_feature, image_fea, motion_fea, aural_fea)[1]
                if idx == 0:
                    all_theta = theta
                else:
                    all_theta = torch.cat((all_theta, theta), dim=0)
        doc_topic_array = all_theta.cpu().numpy()
        return doc_topic_array

    def save_model(self, epoch):
        if self.has_trans:
            path = f'checkpoints/tiktok/ours_trans_depression_{epoch}.pkl'
            torch.save(self.model.to('cpu').state_dict(), path)
            self.model.to(device)
        else:
            path = f'checkpoints/tiktok/ours_nt_depression_{epoch}.pkl'
            torch.save(self.model_nt.to('cpu').state_dict(), path)
            self.model_nt.to(device)


def get_seed_topic_distribution(n_topic, bow_dim, word2id:dict, path):
    seed_topic_distribution = torch.zeros(n_topic, bow_dim)
    with open(f'{path}/guided_LDA_words.txt', 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            seed_words_idx = set([])
            guided_words = set([])
            seed_words = line.strip().split()
            for word in seed_words:
                if word in word2id.keys():
                    seed_words_idx.add(word2id[word])
                    guided_words.add(word)
            seed_topic_distribution[idx][list(seed_words_idx)] = 1/len(seed_words_idx)
    return seed_topic_distribution


def idx_sampler(full_list: [], exist_list: [], sample_num: int):
    sample_space = set(full_list).difference(exist_list)
    sampled_idx = np.random.choice(list(sample_space), sample_num, replace=False)
    return sampled_idx.tolist()


def load_prior_distribution(path):
    regular_topic_dis_trans = np.load(f'{path}/pretrained_topic/comm_topic_dis.npy')
    regular_topic_dis_comm = np.load(f'{path}/pretrained_topic/comm_topic_dis.npy')
    return torch.from_numpy(regular_topic_dis_trans).float(), torch.from_numpy(regular_topic_dis_comm).float()


def get_video_topic_distribution():
    path = '../data/Douyin_Data'
    docSet_nt = Dataset_TM(path, has_trans=True)
    voc_size = docSet_nt.vocabsize
    n_topic = 40
    seed_topics = get_seed_topic_distribution(n_topic, voc_size, docSet_nt.dictionary.token2id, path)
    voc_size = docSet_nt.vocabsize
    all_data = DataLoader(docSet_nt, batch_size=512, shuffle=False)
    n_topic = 40
    regular_topic_trans, regular_topic_comm = load_prior_distribution(path)

    seed_ntm_nt = Knowledge_Guided_NTM(bow_dim=voc_size, n_topic=n_topic, bert_f_dim=768,
                                                        seed_topic_distribution=seed_topics, has_trans=False)

    seed_ntm_nt.set_regular_topic_prior_dis(regular_topic_trans, regular_topic_comm)

    trainer_nt = Trainer(seed_ntm_nt, seed_ntm_nt, docSet_nt, all_data, all_data, batch_size=128, n_epoch=3, has_trans=False)

    trainer_nt.show_trans_topic_words(top_n=100)
    trainer_nt.test_classification()

    doc_topic_array = trainer_nt.inference_doc_topic()
    np.save(f'{path}/doc_topic_array', doc_topic_array)


def train_test_on_short_video_data():

    path = '../data/Tiktok_Data'
    test_rate = 0.3
    docSet = Dataset_TM(path, has_trans=True)
    voc_size = docSet.vocabsize
    n_topic = 40
    seed_topics = get_seed_topic_distribution(n_topic, voc_size, docSet.dictionary.token2id, path)
    regular_topic_trans, regular_topic_comm = load_prior_distribution(path)

    valid_idx = idx_sampler(range(0, len(docSet)), [], int(len(docSet)*test_rate))
    train_idx = idx_sampler(range(0, len(docSet)), valid_idx, len(docSet)-len(valid_idx))
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    # valid_data_loader = DataLoader(docSet, batch_size=128, sampler=valid_sampler)
    # train_data_loader = DataLoader(docSet, batch_size=128, sampler=train_sampler)

    docSet_nt = Dataset_TM(path, has_trans=False)
    voc_size_nt = docSet_nt.vocabsize

    video_ids = [docSet.v_ids[idx] for idx in train_idx]
    existing_idx = [docSet_nt.v_ids.index(v_id) for v_id in video_ids]

    test_idx = idx_sampler(range(0, len(docSet_nt)), existing_idx, int(len(docSet_nt)*test_rate))
    train_nt_idx = idx_sampler(range(0, len(docSet_nt)), test_idx, len(docSet_nt) - len(test_idx))

    test_sampler = SubsetRandomSampler(test_idx)
    train_nt_sampler = SubsetRandomSampler(train_nt_idx)
    test_data_loader = DataLoader(docSet_nt, batch_size=128, sampler=test_sampler)
    train_data_nt_loader = DataLoader(docSet_nt, batch_size=128, sampler=train_nt_sampler)


    seed_ntm = Knowledge_Guided_NTM(bow_dim=voc_size, n_topic=n_topic, bert_f_dim=768, seed_topic_distribution=seed_topics, has_trans=True)
    seed_ntm.set_regular_topic_prior_dis(regular_topic_trans, regular_topic_comm)
    seed_ntm_nt = Knowledge_Guided_NTM(bow_dim=voc_size_nt, n_topic=n_topic, bert_f_dim=768, seed_topic_distribution=seed_topics, has_trans=False)

    trainer_nt = Trainer(seed_ntm, seed_ntm_nt, docSet_nt, train_data_nt_loader, test_data_loader, batch_size=128, n_epoch=2000, has_trans=False)
    trainer_nt.train_for_model_nt()


def test_on_short_video_data():
    path = '../data/General_Data/TikTok_Data'
    docSet = Dataset_General(path)
    n_topic = 40
    voc_size = docSet.vocabsize
    seed_topics = get_seed_topic_distribution(n_topic, voc_size, docSet.dictionary, path)
    seed_ntm = Knowledge_Guided_NTM(bow_dim=voc_size, n_topic=n_topic, bert_f_dim=768,
                                                     seed_topic_distribution=seed_topics, has_trans=True)

    seed_ntm_nt = Knowledge_Guided_NTM(bow_dim=voc_size, n_topic=n_topic, bert_f_dim=768,
                                                        seed_topic_distribution=seed_topics, has_trans=False)

    test_data_loader = DataLoader(docSet, batch_size=512, shuffle=False)
    trainer = Trainer(seed_ntm, seed_ntm_nt, docSet, test_data_loader, test_data_loader, batch_size=128, n_epoch=813,
                      has_trans=True)

    acc, precis, recall, f1 = trainer.test_classification()
    print(f'acc: {acc}, precis:{precis},recall:{recall},f1:{f1}')


def test_on_case_data():

    path = '../data/Case_Data'
    docSet = Dataset_General(path)
    n_topic = 40
    voc_size = docSet.vocabsize

    seed_topics = get_seed_topic_distribution(n_topic, voc_size, docSet.dictionary, path)
    seed_ntm = Knowledge_Guided_NTM(bow_dim=voc_size, n_topic=n_topic, bert_f_dim=768,
                                                     seed_topic_distribution=seed_topics, has_trans=True)

    seed_ntm_nt = Knowledge_Guided_NTM(bow_dim=voc_size, n_topic=n_topic, bert_f_dim=768,
                                                        seed_topic_distribution=seed_topics, has_trans=False)
    test_data_loader = DataLoader(docSet, batch_size=1, shuffle=False)
    trainer = Trainer(seed_ntm, seed_ntm_nt, docSet, test_data_loader, test_data_loader, batch_size=1,
                         n_epoch=3000, has_trans=True)

    acc, precis, recall, f1 = trainer.test_classification()
    print(f'acc: {acc}, precis:{precis},recall:{recall},f1:{f1}')


if __name__ == '__main__':
    train_test_on_short_video_data()
    test_on_short_video_data()