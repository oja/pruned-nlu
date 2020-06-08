import time
import copy
import argparse

import torch
import torch.nn as nn

import models
import dataset
import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    
    if 'atis' in args.name:
        args.dataset = 'atis'
    elif 'snips' in args.name:
        args.dataset = 'snips'

    if 'intent' in args.name:
        args.model = 'intent'
    elif 'slot' in args.name:
        args.model = 'slot'
    elif 'joint' in args.name:
        args.model = 'joint'
    
    args.dropout = 0

    cuda = torch.cuda.is_available()
    train, valid, test, num_words, num_intent, num_slot, wordvecs = dataset.load(args.dataset, batch_size=8, seq_len=50)
    model = util.load_model(args.model, num_words, num_intent, num_slot, args.dropout, wordvecs)
    model.eval()
    
    model.load_state_dict(torch.load(args.name))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if cuda:
        model = model.cuda()

    best_valid_loss = float('inf')
    last_epoch_to_improve = 0
    best_model = model

    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count)

    if args.model == 'intent':
        _, test_acc = util.valid_intent(model, test, criterion, cuda)
        print(f"Test Acc: {test_acc:.5f}")
    elif args.model == 'slot':
        _, test_f1 = util.valid_slot(model, test, criterion, cuda)
        print(f"Test F1: {test_f1:.5f}")
    elif args.model == 'joint':
        _, (_, intent_test_acc), (_, slot_test_f1) = util.valid_joint(model, test, criterion, cuda, 0)
        print(f"Test Intent Acc: {intent_test_acc:.5f}, Slot F1: {slot_test_f1:.5f}")
        
