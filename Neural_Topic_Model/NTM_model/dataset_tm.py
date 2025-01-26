from torch.utils.data import Dataset
from tokenization import HanLPTokenizer, JiebaTokenizer, SpacyTokenizer
from gensim.corpora import Dictionary
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Dataset_TM(Dataset):
    def __init__(self, path, has_trans=True):
        self.path = path
        self.v_ids, self.trans_texts, self.comments, self.labels = self.load_data(f'{self.path}/data.txt')

        self.trans_docs, self.comments_docs = self.text_tokenize()
        self.dictionary = Dictionary(self.trans_docs + self.comments_docs)
        self.dictionary.filter_extremes(no_below=3, no_above=0.8, keep_n=None)
        self.dictionary.compactify()
        self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
        self.id2token = self.dictionary.id2token
        self.trans_bow_docs, self.comments_bow_docs = self.get_bows()
        self.vocabsize = len(self.dictionary)
        self.save_vocab()

        self.labels = np.array([1 if float(label) > 0.10 else 0 for label in self.labels])
        self.trans_feature = None
        self.comment_feature = None
        self.image_feature = None
        self.aural_feature = None
        self.motion_feature = None
        self.load_multimodal_data()

        if has_trans:
            self.filter_data()
        self.num_docs = len(self.trans_bow_docs)

        print(f'Processed {self.num_docs} documents, vocab_size: {self.vocabsize}')
        print(f'positive_rate:{np.sum(self.labels)/self.num_docs}')

    def load_data(self, path):
        v_ids_list = []
        trans_list = []
        comment_list = []
        label_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                split_line = line.strip().split('\t')
                v_ids_list.append(split_line[0])
                trans_list.append(split_line[1])
                comment_list.append(split_line[2])
                label_list.append(split_line[3])
        return v_ids_list, trans_list, comment_list, label_list

    def text_tokenize(self):
        with open(f'{self.path}/stop_words.txt', 'r', encoding='utf-8') as f:
            stop_words = [word.strip() for word in f.readlines()]
        text_tokenizer = JiebaTokenizer(stopwords=stop_words)

        trans_docs = text_tokenizer.tokenize(self.trans_texts)
        comments_docs = text_tokenizer.tokenize(self.comments)

        with open(f'{self.path}/docs_temp/trans_docs.txt', 'w', encoding='utf-8') as f:
            for doc in trans_docs:
                doc_line = ' '.join(doc)
                f.write(f'{doc_line}\n')
        with open(f'{self.path}/docs_temp/comm_docs.txt', 'w', encoding='utf-8') as f:
            for doc in comments_docs:
                doc_line = ' '.join(doc)
                f.write(f'{doc_line}\n')
        return trans_docs, comments_docs

    def get_bows(self):
        trans_bow_docs = []
        comments_bow_docs = []
        for idx, doc in enumerate(self.trans_docs):
            bow = self.dictionary.doc2bow(doc)
            trans_bow_docs.append(bow)
        for idx, doc in enumerate(self.comments_docs):
            bow = self.dictionary.doc2bow(doc)
            comments_bow_docs.append(bow)
        return trans_bow_docs, comments_bow_docs

    def load_multimodal_data(self):
        trans_idx_list = [idx.strip() for idx in open(f'{self.path}/trans_id_list.txt', 'r', encoding='utf-8').readlines()]
        trans_feature = np.load(f'{self.path}/trans_feature.npy')
        feature_list = []
        for i, v_id in enumerate(self.v_ids):
            if v_id in trans_idx_list and len(self.trans_bow_docs[i]) > 0:
                idx = trans_idx_list.index(v_id)
                feature = trans_feature[idx]
            else:
                feature = np.zeros_like(trans_feature[0])
            feature_list.append(feature)
        self.trans_feature = np.array(feature_list)

        comment_idx_list = [idx.strip() for idx in open(f'{self.path}/comment_id_list.txt', 'r', encoding='utf-8').readlines()]
        comment_feature = np.load(f'{self.path}/comment_feature.npy')
        idx_list = [comment_idx_list.index(v_id) for v_id in self.v_ids]
        self.comment_feature = comment_feature[idx_list]

        image_idx_list = [idx.strip() for idx in open(f'{self.path}/image_id_list.txt', 'r', encoding='utf-8').readlines()]
        image_feature = np.load(f'{self.path}/image_feature.npy')
        idx_list = [image_idx_list.index(v_id) for v_id in self.v_ids]
        self.image_feature = image_feature[idx_list]

        aural_idx_list = [idx.strip() for idx in open(f'{self.path}/aural_id_list.txt', 'r', encoding='utf-8').readlines()]
        aural_feature = np.load(f'{self.path}/aural_feature.npy')
        idx_list = [aural_idx_list.index(v_id) for v_id in self.v_ids]
        self.aural_feature = aural_feature[idx_list]

        motion_idx_list = [idx.strip() for idx in open(f'{self.path}/motion_id_list.txt', 'r', encoding='utf-8').readlines()]
        self.motion_feature = np.zeros_like(self.image_feature)

    def filter_data(self):
        selected_idx = [i for i in range(len(self.v_ids)) if len(self.trans_bow_docs[i]) > 0]
        self.trans_bow_docs = [self.trans_bow_docs[i] for i in range(len(self.v_ids)) if i in selected_idx]
        self.comments_bow_docs = [self.comments_bow_docs[i] for i in range(len(self.v_ids)) if i in selected_idx]
        self.labels = [self.labels[i] for i in range(len(self.v_ids)) if i in selected_idx]
        self.trans_feature = self.trans_feature[selected_idx]
        self.comment_feature = self.comment_feature[selected_idx]
        self.image_feature = self.image_feature[selected_idx]
        self.aural_feature = self.aural_feature[selected_idx]
        self.motion_feature = self.motion_feature[selected_idx]
        self.v_ids = [self.v_ids[i] for i in range(len(self.v_ids)) if i in selected_idx]

    def save_vocab(self):
        with open(f'{self.path}/vocab.txt', 'w', encoding='utf-8') as f:
            for idx in range(len(self.dictionary.id2token)):
                f.write(f'{self.dictionary.id2token[idx]}\n')

    def __getitem__(self, idx):
        trans_bow = torch.zeros(len(self.dictionary))
        if len(self.trans_bow_docs[idx]) > 0:
            item_trans = list(zip(*self.trans_bow_docs[idx]))
            trans_bow[list(item_trans[0])] = torch.tensor(list(item_trans[1])).float()

        comments_bow = torch.zeros(len(self.dictionary))
        if len(self.comments_bow_docs[idx]) > 0:
            item_comments = list(zip(*self.comments_bow_docs[idx]))
            comments_bow[list(item_comments[0])] = torch.tensor(list(item_comments[1])).float()
        return trans_bow, self.trans_feature[idx], comments_bow, self.comment_feature[idx], \
               self.image_feature[idx], self.motion_feature[idx], self.aural_feature[idx], self.labels[idx]

    def __len__(self):
        return self.num_docs


class Dataset_General(Dataset):
    def __init__(self, path):
        self.path = path
        self.v_ids, self.trans_texts, self.comments, self.labels = self.load_data(f'{self.path}/data.txt')
        self.trans_docs, self.comments_docs = self.text_tokenize()
        self.dictionary = self.load_dictionary()
        self.id2token = {v: k for k, v in self.dictionary.items()}
        self.trans_bows, self.comments_bows = self.get_bows()
        self.vocabsize = len(self.dictionary)
        self.labels = np.array([1 if float(label) > 0.15 else 0 for label in self.labels])
        self.trans_feature = None
        self.comment_feature = None
        self.image_feature = None
        self.aural_feature = None
        self.motion_feature = None
        self.load_multimodal_data()
        self.num_docs = len(self.v_ids)
        print(f'Processed {self.num_docs} documents, vocab_size: {self.vocabsize}')
        print(f'positive_rate:{np.sum(self.labels)/self.num_docs}')

    def load_data(self, path):
        v_ids_list = []
        trans_list = []
        comment_list = []
        label_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                split_line = line.strip().split('\t')
                v_ids_list.append(split_line[0])
                trans_list.append(split_line[1])
                comment_list.append(split_line[2])
                label_list.append(split_line[3])
        return v_ids_list, trans_list, comment_list, label_list

    def text_tokenize(self):
        with open(f'{self.path}/stop_words.txt', 'r', encoding='utf-8') as f:
            stop_words = [word.strip() for word in f.readlines()]
        text_tokenizer = SpacyTokenizer(stopwords=stop_words)
        trans_docs = text_tokenizer.tokenize(self.trans_texts)
        comments_docs = text_tokenizer.tokenize(self.comments)
        with open(f'{self.path}/docs_temp/trans_docs.txt', 'w', encoding='utf-8') as f:
            for doc in trans_docs:
                doc_line = ' '.join(doc)
                f.write(f'{doc_line}\n')
        with open(f'{self.path}/docs_temp/comm_docs.txt', 'w', encoding='utf-8') as f:
            for doc in comments_docs:
                doc_line = ' '.join(doc)
                f.write(f'{doc_line}\n')
        return trans_docs, comments_docs

    def load_dictionary(self):
        vocab = [line.replace('\n', '') for line in open(f'{self.path}/vocab.txt', encoding='utf-8').readlines()]
        word2id = {word: idx for idx, word in enumerate(vocab)}
        return word2id

    def get_bows(self):
        trans_bows = []
        comments_bows = []
        for idx, doc in enumerate(self.trans_docs):
            bow_tensor = torch.zeros(len(self.dictionary))
            for word in doc:
                if word in self.dictionary.keys():
                    bow_tensor[self.dictionary[word]] += 1
            trans_bows.append(bow_tensor.tolist())
        for idx, doc in enumerate(self.comments_docs):
            bow_tensor = torch.zeros(len(self.dictionary))
            for word in doc:
                if word in self.dictionary.keys():
                    bow_tensor[self.dictionary[word]] += 1
            comments_bows.append(bow_tensor.tolist())
        return torch.tensor(trans_bows).float(), torch.tensor(comments_bows).float()

    def load_multimodal_data(self):
        trans_idx_list = [idx.strip() for idx in open(f'{self.path}/trans_id_list.txt', 'r', encoding='utf-8').readlines()]
        trans_feature = np.load(f'{self.path}/trans_feature.npy')
        feature_list = []
        for i, v_id in enumerate(self.v_ids):
            if v_id in trans_idx_list and len(self.trans_bows[i]) > 0:
                idx = trans_idx_list.index(v_id)
                feature = trans_feature[idx]
            else:
                feature = np.zeros_like(trans_feature[0])
            feature_list.append(feature)
        self.trans_feature = np.array(feature_list)

        comment_idx_list = [idx.strip() for idx in open(f'{self.path}/comment_id_list.txt', 'r', encoding='utf-8').readlines()]
        comment_feature = np.load(f'{self.path}/comment_feature.npy')
        idx_list = [comment_idx_list.index(v_id) for v_id in self.v_ids]
        self.comment_feature = comment_feature[idx_list]

        image_idx_list = [idx.strip() for idx in open(f'{self.path}/image_id_list.txt', 'r', encoding='utf-8').readlines()]
        image_feature = np.load(f'{self.path}/image_feature.npy')
        idx_list = [image_idx_list.index(v_id) for v_id in self.v_ids]
        self.image_feature = image_feature[idx_list]

        aural_idx_list = [idx.strip() for idx in open(f'{self.path}/aural_id_list.txt', 'r', encoding='utf-8').readlines()]
        aural_feature = np.load(f'{self.path}/aural_feature.npy')
        aural_feature_list = []
        for v_id in self.v_ids:
            if v_id in aural_idx_list:
                feature = aural_feature[aural_idx_list.index(v_id)]
            else:
                feature = np.zeros_like(aural_feature[0])
            aural_feature_list.append(feature)
        self.aural_feature = np.array(aural_feature_list)

        motion_idx_list = [idx.strip() for idx in open(f'{self.path}/motion_id_list.txt', 'r', encoding='utf-8').readlines()]
        motion_feature = np.load(f'{self.path}/motion_feature.npy')
        motion_feature_list = []
        for v_id in self.v_ids:
            if v_id in motion_idx_list:
                feature = motion_feature[motion_idx_list.index(v_id)]
            else:
                feature = np.zeros_like(motion_feature[0])
            motion_feature_list.append(feature)
        self.motion_feature = np.array(motion_feature_list)

    def __getitem__(self, idx):
        return self.trans_bows[idx], self.trans_feature[idx], self.comments_bows[idx], self.comment_feature[idx], \
               self.image_feature[idx], self.motion_feature[idx], self.aural_feature[idx], self.labels[idx]

    def __len__(self):
        return self.num_docs