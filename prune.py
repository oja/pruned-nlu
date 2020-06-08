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
    parser.add_argument('--zeroshot', action='store_true')

    parser.add_argument('--epochs', default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--filename')
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

    print(f"seed {util.rep(args.seed)}")

    cuda = torch.cuda.is_available()
    train, valid, test, num_words, num_intent, num_slot, wordvecs = dataset.load(args.dataset, batch_size=8, seq_len=50)

    model = util.load_model(args.model, num_words, num_intent, num_slot, args.dropout, wordvecs)

    model.load_state_dict(torch.load(args.name))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if cuda:
        model = model.cuda()

    if len(args.name.split('/')) > 1:
        nameprefix = args.name.split('/')[-1]
    else:
        nameprefix = args.name
        

    filename = args.filename if args.filename else f"results/{nameprefix}_{'zeroshot' if args.zeroshot else 'retrain'}_l{args.l}_alpha{args.alpha}.csv"
    if args.model == 'intent':
        open(filename, 'w').close() # clear the file
        f = open(filename, "a")
        while sum(model.filter_sizes) > 0:
            _, test_acc = util.valid_intent(model, test, criterion, cuda)
            print(f"{sum(model.filter_sizes)}, {test_acc:.5f}", file=f, flush=True)
            
            if sum(model.filter_sizes) > 10:
                model.prune(5, args.l)
            else:
                model.prune(1, args.l)

            if not args.zeroshot:
                optimizer = torch.optim.Adam(model.parameters())
                best_epoch = 0
                best_valid_loss, _ = util.valid_intent(model, valid, criterion, cuda)
                best_model = copy.deepcopy(model)
                epoch = 1

                while epoch <= best_epoch + args.patience:
                    train_loss, train_acc = util.train_intent(model, train, criterion, optimizer, cuda)
                    valid_loss, valid_acc = util.valid_intent(model, valid, criterion, cuda)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        best_model = copy.deepcopy(model)

                    epoch += 1

                model = best_model
    elif args.model == 'slot':
        open(filename, 'w').close() # clear the file
        f = open(filename, "a")
        while sum(model.filter_sizes) > 0:
            _, test_f1 = util.valid_slot(model, test, criterion, cuda)
            print(f"{sum(model.filter_sizes)}, {test_f1:.5f}", file=f, flush=True)
            
            if sum(model.filter_sizes) > 10:
                model.prune(5, args.l)
            else:
                model.prune(1, args.l)

            if not args.zeroshot:
                optimizer = torch.optim.Adam(model.parameters())
                best_epoch = 0
                best_valid_loss, _ = util.valid_slot(model, valid, criterion, cuda)
                best_model = copy.deepcopy(model)
                epoch = 1

                while epoch <= best_epoch + args.patience:
                    train_loss, train_f1 = util.train_slot(model, train, criterion, optimizer, cuda)
                    valid_loss, valid_f1 = util.valid_slot(model, valid, criterion, cuda)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        best_model = copy.deepcopy(model)

                    epoch += 1

                model = best_model
    elif args.model == 'joint':
        open(filename, 'w').close() # clear the file
        f = open(filename, "a")
        while sum(model.filter_sizes) > 0:
            _, (_, test_acc), (_, test_f1) = util.valid_joint(model, test, criterion, cuda, args.alpha)
            print(f"{sum(model.filter_sizes)}, {test_acc:.5f}, {test_f1:.5f}", file=f, flush=True)
            
            if sum(model.filter_sizes) > 10:
                model.prune(5, args.l)
            else:
                model.prune(1, args.l)

            if not args.zeroshot:
                optimizer = torch.optim.Adam(model.parameters())
                best_epoch = 0
                best_valid_loss, (_, _), (_, _) = util.valid_joint(model, valid, criterion, cuda, args.alpha)
                best_model = copy.deepcopy(model)
                epoch = 1

                while epoch <= best_epoch + args.patience:
                    train_loss, (_, _), (_, _) = util.train_joint(model, train, criterion, optimizer, cuda, args.alpha)
                    valid_loss, (_, _), (_, _) = util.valid_joint(model, valid, criterion, cuda, args.alpha)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        best_model = copy.deepcopy(model)

                    epoch += 1

                model = best_model
                
