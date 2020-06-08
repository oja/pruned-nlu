import collections

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

def build_glove(word2idx, idx2word, dim=100):
    word2vecs = {}

    with open(f'glove/glove.6B.{dim}d.txt') as glove_file:
        for i, line in enumerate(glove_file):
            splat = line.split()
            word = str(splat.pop(0))
            if word in word2idx:
                word2vecs[word] = np.array(splat).astype(float)

    vectors = []
    for word in [idx2word[i] for i in range(len(idx2word))]:
        if word in word2vecs:
            vectors.append(torch.from_numpy(word2vecs[word]).float())
        else:
            vectors.append(torch.from_numpy(np.random.normal(0, 0.5, size=(dim,))).float())

    return torch.stack(vectors)

class Corpus(Dataset):
    def __init__(self, dataset, split_name, seq_len: int):
        self.seq_len = seq_len
        self.queries = []
        self.intents = []
        self.slots = []

        self.word2idx = {}
        self.idx2word = {}
        self.intent2idx = {}
        self.slot2idx = {}

        self._register(PAD)
        self._register(BOS)
        self._register(EOS)

        for split in ['train', 'valid', 'test']:
            with open(f'datasets/{dataset}/{split}/label') as intent_file:
                for line in intent_file:
                    intent = line.rstrip()
                    if intent not in self.intent2idx:
                        self.intent2idx[intent] = len(self.intent2idx)
                
            with open(f'datasets/{dataset}/{split}/seq.in') as queries_file:
                for line in queries_file:
                    query = line.rstrip().split()
                    for word in query:
                        if word not in self.word2idx:
                            idx = len(self.word2idx)
                            self.word2idx[word] = idx
                            self.idx2word[idx] = word

            with open(f'datasets/{dataset}/{split}/seq.out') as slotses_file:
                for line in slotses_file:
                    slots = line.rstrip().split()
                    for slot in slots:
                        if slot not in self.slot2idx:
                            self.slot2idx[slot] = len(self.slot2idx)
        
        with open(f'datasets/{dataset}/{split_name}/label') as intent_file:
            for line in intent_file:
                intent = line.rstrip()
                self.intents.append(intent)
            
        with open(f'datasets/{dataset}/{split_name}/seq.in') as queries_file:
            for line in queries_file:
                query = line.rstrip().split()                
                self.queries.append(query)

        with open(f'datasets/{dataset}/{split_name}/seq.out') as slotses_file:
            for line in slotses_file:
                slots = line.rstrip().split()  
                self.slots.append(slots)

        self.idx2intent = {v: k for k, v in self.intent2idx.items()}
        self.idx2slot = {v : k for k, v in self.slot2idx.items()}


    def _register(self, word):
        if word in self.word2idx:
            return

        assert(len(self.idx2word) == len(self.word2idx))
        idx = len(self.idx2word)
        self.idx2word[idx] = word
        self.word2idx[word] = idx

    def pad_query(self, sequence):
        sequence = [self.word2idx[BOS]] + sequence + [self.word2idx[EOS]]
        sequence = sequence[:self.seq_len]
        sequence = np.pad(sequence, (0, self.seq_len - len(sequence)), mode='constant', constant_values=(self.word2idx[PAD],))
        return sequence

    def pad_slots(self, sequence):
        sequence = [-1] + sequence + [-1]
        sequence = sequence[:self.seq_len]
        sequence = np.pad(sequence, (0, self.seq_len - len(sequence)), mode='constant', constant_values=(-1,))
        return sequence

    def __getitem__(self, i):
        query = torch.from_numpy(self.pad_query([self.word2idx[word] for word in self.queries[i]]))
        intent = torch.tensor(self.intent2idx[self.intents[i]])
        slots = torch.from_numpy(self.pad_slots([self.slot2idx[slot] for slot in self.slots[i]]))
        true_length = torch.tensor(min(len(self.queries[i]), self.seq_len))
        return query, intent, slots, true_length, (self.queries[i], self.intents[i], self.slots[i]), (self.idx2word, self.idx2intent, self.idx2slot)

    def __len__(self):
        assert(len(self.queries) == len(self.intents))
        return len(self.queries)

def load(dataset, batch_size, seq_len):
    train_corpus, valid_corpus, test_corpus = Corpus(dataset, 'train', seq_len), Corpus(dataset, 'valid', seq_len), Corpus(dataset, 'test', seq_len)

    # sanity checks
    assert(len(train_corpus.word2idx) == len(valid_corpus.word2idx) == len(test_corpus.word2idx))
    assert(len(train_corpus.intent2idx) == len(valid_corpus.intent2idx) == len(test_corpus.intent2idx))
    assert(len(train_corpus.slot2idx) == len(valid_corpus.slot2idx) == len(test_corpus.slot2idx))

    num_words, num_intents, num_slots = len(train_corpus.word2idx), len(train_corpus.intent2idx), len(train_corpus.slot2idx)
    wordvecs = build_glove(train_corpus.word2idx, train_corpus.idx2word)
    return (DataLoader(train_corpus, batch_size, shuffle=True), 
            DataLoader(valid_corpus, batch_size, shuffle=False),
            DataLoader(test_corpus, batch_size), 
            num_words, num_intents, num_slots, wordvecs)
