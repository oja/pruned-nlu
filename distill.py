import time
import copy
import argparse

import torch
import torch.nn as nn

import dataset
import util
import models

from itertools import chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--filename', required=True)

    parser.add_argument('--epochs', default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
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
    
    open(args.filename, 'w').close() # clear the file
    f = open(args.filename, "a")

    for filter_count in chain(range(300, 10, -5), range(10, 0, -1)):
        if args.model == 'intent':
            model = models.CNNIntent(num_words, 100, num_intent, (filter_count,), 5, args.dropout, wordvecs)
        elif args.model == 'slot':
            model = models.CNNSlot(num_words, 100, num_slot, (filter_count,), 5, args.dropout, wordvecs)
        elif args.model == 'joint':
            model = models.CNNJoint(num_words, 100, num_intent, num_slot, (filter_count,), 5, args.dropout, wordvecs)

        teacher = util.load_model(args.model, num_words, num_intent, num_slot, args.dropout, wordvecs)
        teacher.load_state_dict(torch.load(args.name))

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        distill_criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(model.parameters())

        if cuda:
            model = model.cuda()
            teacher = teacher.cuda()

        best_valid_loss = float('inf')
        last_epoch_to_improve = 0
        best_model = model
        model_filename = f"models/{args.dataset}_{args.model}"
        
        if args.model == 'intent':
            for epoch in range(args.epochs):
                start_time = time.time()
                train_loss, train_acc = util.distill_intent(teacher, model, 1.0, train, distill_criterion, optimizer, cuda)
                valid_loss, valid_acc = util.valid_intent(model, valid, criterion, cuda)
                end_time = time.time()

                elapsed_time = end_time - start_time

                print(f"Epoch {epoch + 1:03} took {elapsed_time:.3f} seconds")
                print(f"\tTrain Loss: {train_loss:.5f}, Acc: {train_acc:.5f}")
                print(f"\tValid Loss: {valid_loss:.5f}, Acc: {valid_acc:.5f}")

                if valid_loss < best_valid_loss:
                    last_epoch_to_improve = epoch
                    best_valid_loss = valid_loss
                    best_model = copy.deepcopy(model)
                    print("\tNew best valid loss!")

                if last_epoch_to_improve + args.patience < epoch:
                    break

            _, test_acc = util.valid_intent(best_model, test, criterion, cuda)
            print(f"Test Acc: {test_acc:.5f}")
            print(f"{sum(best_model.filter_sizes)}, {test_acc:.5f}", file=f, flush=True)
        elif args.model == 'slot':
            for epoch in range(args.epochs):
                start_time = time.time()
                train_loss, train_f1 = util.distill_slot(teacher, model, 1.0, train, distill_criterion, optimizer, cuda)
                valid_loss, valid_f1 = util.valid_slot(model, valid, criterion, cuda)
                end_time = time.time()

                elapsed_time = end_time - start_time

                print(f"Epoch {epoch + 1:03} took {elapsed_time:.3f} seconds")
                print(f"\tTrain Loss: {train_loss:.5f}, F1: {train_f1:.5f}")
                print(f"\tValid Loss: {valid_loss:.5f}, F1: {valid_f1:.5f}")

                if valid_loss < best_valid_loss:
                    last_epoch_to_improve = epoch
                    best_valid_loss = valid_loss
                    best_model = copy.deepcopy(model)
                    print("\tNew best valid loss!")

                if last_epoch_to_improve + args.patience < epoch:
                    break

            _, test_f1 = util.valid_slot(best_model, test, criterion, cuda)
            print(f"Test F1: {test_f1:.5f}")
            print(f"{sum(best_model.filter_sizes)}, {test_f1:.5f}", file=f, flush=True)
        elif args.model == 'joint':
            for epoch in range(args.epochs):
                start_time = time.time()
                train_loss, (intent_train_loss, intent_train_acc), (slot_train_loss, slot_train_f1) = util.distill_joint(teacher, model, 1.0, train, distill_criterion, optimizer, cuda, args.alpha)
                valid_loss, (intent_valid_loss, intent_valid_acc), (slot_valid_loss, slot_valid_f1) = util.valid_joint(model, valid, criterion, cuda, args.alpha)
                end_time = time.time()

                elapsed_time = end_time - start_time

                print(f"Epoch {epoch + 1:03} took {elapsed_time:.3f} seconds")
                print(f"\tTrain Loss: {train_loss:.5f}, (Intent Loss: {intent_train_loss:.5f}, Acc: {intent_train_acc:.5f}), (Slot Loss: {slot_train_loss:.5f}, F1: {slot_train_f1:.5f})")
                print(f"\tValid Loss: {valid_loss:.5f}, (Intent Loss: {intent_valid_loss:.5f}, Acc: {intent_valid_acc:.5f}), (Slot Loss: {slot_valid_loss:.5f}, F1: {slot_valid_f1:.5f})")

                if valid_loss < best_valid_loss:
                    last_epoch_to_improve = epoch
                    best_valid_loss = valid_loss
                    best_model = copy.deepcopy(model)
                    print("\tNew best valid loss!")

                if last_epoch_to_improve + args.patience < epoch:
                    break
            _, (_, intent_test_acc), (_, slot_test_f1) = util.valid_joint(best_model, test, criterion, cuda, args.alpha)
            print(f"Test Intent Acc: {intent_test_acc:.5f}, Slot F1: {slot_test_f1:.5f}")
            print(f"{sum(best_model.filter_sizes)}, {intent_test_acc:.5f}, {slot_test_f1:.5f}", file=f, flush=True)

