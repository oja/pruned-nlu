import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNIntent(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, filter_sizes, kernel_size, dropout, wordvecs=None):
        super().__init__()

        if wordvecs is not None:
            self.embedding = nn.Embedding.from_pretrained(wordvecs)
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            
        self.convs = nn.ModuleList(
            [nn.Conv1d(filter_sizes[i - 1] if i > 0 else embedding_dim, filter_sizes[i], kernel_size) for i in range(len(filter_sizes))]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filter_sizes[-1], output_dim)

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.unpruned_count = sum(filter_sizes)

    def forward(self, query): # query shape: [batch, seq len]
        x = self.embedding(query).permute(0, 2, 1) # [batch, embedding dim, seq len]
        for conv in self.convs:
            x = conv(x)
            x = torch.rrelu(x)
        x = x.permute(0, 2, 1)
        x, _ = torch.max(x, dim=1)
        return self.fc(self.dropout(x))

    def prune(self, count, norm=2):
        if not (sum(self.filter_sizes) - count > 0): # ensure we will have > 0 filters over
            exit(0)
        
        rankings = [] # list of (conv #, filter #, norm)
        for i, conv in enumerate(self.convs):
            for k, filter in enumerate(conv.weight):
                rankings.append((i, k, torch.norm(filter.view(-1), p=norm, dim=0).item()))
        rankings.sort(key = lambda x: x[2])

        for ranking in rankings[:count]:
            conv_num, filter_num, _ = ranking

            # remove filter
            new_weight = torch.cat((self.convs[conv_num].weight[:filter_num],
                                    self.convs[conv_num].weight[filter_num + 1:]))
            new_bias = torch.cat((self.convs[conv_num].bias[:filter_num],
                                  self.convs[conv_num].bias[filter_num + 1:]))

            self.convs[conv_num] = nn.Conv1d(self.filter_sizes[conv_num - 1] if conv_num > 0 else self.embedding_dim,
                                             self.filter_sizes[conv_num] - 1,
                                             self.kernel_size)
            self.convs[conv_num].weight = nn.Parameter(new_weight)
            self.convs[conv_num].bias = nn.Parameter(new_bias)

            # update channel in succeeding layer
            if conv_num == len(self.filter_sizes) - 1: # prune linear
                new_weight = torch.cat((self.fc.weight[:,:filter_num], self.fc.weight[:,filter_num + 1:]), dim=1)
                new_bias = self.fc.bias

                self.fc = nn.Linear(self.fc.in_features - 1, self.fc.out_features)
                self.fc.weight = nn.Parameter(new_weight)
                self.fc.bias = nn.Parameter(new_bias)
            else: # prune conv
                new_weight = torch.cat((self.convs[conv_num + 1].weight[:,:filter_num], self.convs[conv_num + 1].weight[:,filter_num + 1:]), dim=1)
                new_bias = self.convs[conv_num + 1].bias

                self.convs[conv_num + 1] = nn.Conv1d(self.filter_sizes[conv_num] - 1,
                                                     self.filter_sizes[conv_num + 1],
                                                     self.kernel_size)

                self.convs[conv_num + 1].weight = nn.Parameter(new_weight)
                self.convs[conv_num + 1].bias = nn.Parameter(new_bias)

            self.filter_sizes = tuple([filter_size - 1 if i == conv_num else filter_size for i, filter_size in enumerate(self.filter_sizes)])


class CNNSlot(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, filter_sizes, kernel_size, dropout, wordvecs=None):
        super().__init__()

        if wordvecs is not None:
            self.embedding = nn.Embedding.from_pretrained(wordvecs)
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.convs = nn.ModuleList(
            [nn.Conv1d(filter_sizes[i - 1] if i > 0 else embedding_dim, filter_sizes[i], kernel_size) for i in range(len(filter_sizes))]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filter_sizes[-1], output_dim)

        self.padding = int((kernel_size - 1) / 2)

        self.embedding_dim = embedding_dim
        self.unpruned_count = sum(filter_sizes)
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size

    def forward(self, query):  # query shape: [batch, seq len]
        x = self.embedding(query) # embedded shape: [batch, seq len, embedding dim]
        x = x.permute(0, 2, 1) # x shape: [batch, embedding dim, seq len]
        for conv in self.convs:
            x = F.pad(x, (self.padding, self.padding)) # x shape: [batch, filter count, seq len]
            x = conv(x)
            x = torch.rrelu(x)
        x = x.permute(0, 2, 1) # x shape: [batch, seq len, filter count]
        x = self.fc(self.dropout(x))
        return x

    def prune(self, count, norm=2):
        if not (sum(self.filter_sizes) - count > 0): # ensure we will have > 0 filters over
            exit(0)
        
        rankings = [] # list of (conv #, filter #, norm)
        for i, conv in enumerate(self.convs):
            for k, filter in enumerate(conv.weight):
                rankings.append((i, k, torch.norm(filter.view(-1), p=norm, dim=0).item()))
        rankings.sort(key = lambda x: x[2])

        for ranking in rankings[:count]:
            conv_num, filter_num, _ = ranking

            # remove filter
            new_weight = torch.cat((self.convs[conv_num].weight[:filter_num],
                                    self.convs[conv_num].weight[filter_num + 1:]))
            new_bias = torch.cat((self.convs[conv_num].bias[:filter_num],
                                  self.convs[conv_num].bias[filter_num + 1:]))

            self.convs[conv_num] = nn.Conv1d(self.filter_sizes[conv_num - 1] if conv_num > 0 else self.embedding_dim,
                                             self.filter_sizes[conv_num] - 1,
                                             self.kernel_size)
            self.convs[conv_num].weight = nn.Parameter(new_weight)
            self.convs[conv_num].bias = nn.Parameter(new_bias)

            # update channel in succeeding layer
            if conv_num == len(self.filter_sizes) - 1: # prune linear
                new_weight = torch.cat((self.fc.weight[:,:filter_num], self.fc.weight[:,filter_num + 1:]), dim=1)
                new_bias = self.fc.bias

                self.fc = nn.Linear(self.fc.in_features - 1, self.fc.out_features)
                self.fc.weight = nn.Parameter(new_weight)
                self.fc.bias = nn.Parameter(new_bias)
            else: # prune conv
                new_weight = torch.cat((self.convs[conv_num + 1].weight[:,:filter_num], self.convs[conv_num + 1].weight[:,filter_num + 1:]), dim=1)
                new_bias = self.convs[conv_num + 1].bias

                self.convs[conv_num + 1] = nn.Conv1d(self.filter_sizes[conv_num] - 1,
                                                     self.filter_sizes[conv_num + 1],
                                                     self.kernel_size)

                self.convs[conv_num + 1].weight = nn.Parameter(new_weight)
                self.convs[conv_num + 1].bias = nn.Parameter(new_bias)

            self.filter_sizes = tuple([filter_size - 1 if i == conv_num else filter_size for i, filter_size in enumerate(self.filter_sizes)])

    
class CNNJoint(nn.Module):
    def __init__(self, input_dim, embedding_dim, intent_dim, slot_dim, filter_sizes, kernel_size, dropout, wordvecs=None):
        super().__init__()

        if wordvecs is not None:
            self.embedding = nn.Embedding.from_pretrained(wordvecs)
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.convs = nn.ModuleList(
            [nn.Conv1d(filter_sizes[i - 1] if i > 0 else embedding_dim, filter_sizes[i], kernel_size) for i in range(len(filter_sizes))]
        )

        self.intent_dropout = nn.Dropout(dropout)
        self.intent_fc = nn.Linear(filter_sizes[-1], intent_dim)
        self.slot_dropout = nn.Dropout(dropout)
        self.slot_fc = nn.Linear(filter_sizes[-1], slot_dim)

        self.padding = int((kernel_size - 1) / 2)
        self.unpruned_count = sum(filter_sizes)
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size

    def forward(self, query):
        x = self.embedding(query).permute(0, 2, 1)
        for conv in self.convs:
            x = F.pad(x, (self.padding, self.padding))
            x = conv(x)
            x = torch.rrelu(x)
        x = x.permute(0, 2, 1)

        intent_pred = self.intent_fc(self.intent_dropout(torch.max(x, dim=1)[0]))
        slot_pred = self.slot_fc(self.slot_dropout(x))

        return intent_pred, slot_pred.permute(0, 2, 1)

    def prune(self, count, norm=2):
        if not (sum(self.filter_sizes) - count > 0): # ensure we will have > 0 filters over
            exit(0)
        
        rankings = [] # list of (conv #, filter #, norm)
        for i, conv in enumerate(self.convs):
            for k, filter in enumerate(conv.weight):
                rankings.append((i, k, torch.norm(filter.view(-1), p=norm, dim=0).item()))
        rankings.sort(key = lambda x: x[2])

        for ranking in rankings[:count]:
            conv_num, filter_num, _ = ranking

            # remove filter
            new_weight = torch.cat((self.convs[conv_num].weight[:filter_num],
                                    self.convs[conv_num].weight[filter_num + 1:]))
            new_bias = torch.cat((self.convs[conv_num].bias[:filter_num],
                                  self.convs[conv_num].bias[filter_num + 1:]))

            self.convs[conv_num] = nn.Conv1d(self.filter_sizes[conv_num - 1] if conv_num > 0 else self.embedding_dim,
                                             self.filter_sizes[conv_num] - 1,
                                             self.kernel_size)
            self.convs[conv_num].weight = nn.Parameter(new_weight)
            self.convs[conv_num].bias = nn.Parameter(new_bias)

            # update channel in succeeding layer
            if conv_num == len(self.filter_sizes) - 1: # prune linear
                new_intent_weight = torch.cat((self.intent_fc.weight[:,:filter_num], self.intent_fc.weight[:,filter_num + 1:]), dim=1)
                new_intent_bias = self.intent_fc.bias

                self.intent_fc = nn.Linear(self.intent_fc.in_features - 1, self.intent_fc.out_features)
                self.intent_fc.weight = nn.Parameter(new_intent_weight)
                self.intent_fc.bias = nn.Parameter(new_intent_bias)

                new_slot_weight = torch.cat((self.slot_fc.weight[:,:filter_num], self.slot_fc.weight[:,filter_num + 1:]), dim=1)
                new_slot_bias = self.slot_fc.bias

                self.slot_fc = nn.Linear(self.slot_fc.in_features - 1, self.slot_fc.out_features)
                self.slot_fc.weight = nn.Parameter(new_slot_weight)
                self.slot_fc.bias = nn.Parameter(new_slot_bias)
            else: # prune conv
                new_weight = torch.cat((self.convs[conv_num + 1].weight[:,:filter_num], self.convs[conv_num + 1].weight[:,filter_num + 1:]), dim=1)
                new_bias = self.convs[conv_num + 1].bias

                self.convs[conv_num + 1] = nn.Conv1d(self.filter_sizes[conv_num] - 1,
                                                     self.filter_sizes[conv_num + 1],
                                                     self.kernel_size)

                self.convs[conv_num + 1].weight = nn.Parameter(new_weight)
                self.convs[conv_num + 1].bias = nn.Parameter(new_bias)

            self.filter_sizes = tuple([filter_size - 1 if i == conv_num else filter_size for i, filter_size in enumerate(self.filter_sizes)])
