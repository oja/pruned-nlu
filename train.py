import time
import copy
import argparse

import torch

import dataset
import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['atis', 'snips'])
    parser.add_argument('--model', choices=['intent', 'slot', 'joint'])
    parser.add_argument('--name', default=None)

    parser.add_argument('--epochs', default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()

    print(f"seed {util.rep(args.seed)}")

    cuda = torch.cuda.is_available()
    train, valid, test, num_words, num_intent, num_slot, wordvecs = dataset.load(args.dataset, batch_size=8, seq_len=50)

    model = util.load_model(args.model, num_words, num_intent, num_slot, args.dropout, wordvecs)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters())

    if cuda:
        model = model.cuda()

    best_valid_loss = float('inf')
    last_epoch_to_improve = 0
    best_model = model
    model_filename = f"models/{args.dataset}_{args.model}" if not args.name else args.name

    if args.model == 'intent':
        for epoch in range(0, args.epochs):
            start_time = time.time()
            train_loss, train_acc = util.train_intent(model, train, criterion, optimizer, cuda)
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
                torch.save(model.state_dict(), model_filename)
                print("\tNew best valid loss!")

            if last_epoch_to_improve + args.patience < epoch:
                break

        _, test_acc = util.valid_intent(best_model, test, criterion, cuda)
        print(f"Test Acc: {test_acc:.5f}")
    elif args.model == 'slot':
        for epoch in range(0, args.epochs):
            start_time = time.time()
            train_loss, train_f1 = util.train_slot(model, train, criterion, optimizer, cuda)
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
                torch.save(model.state_dict(), model_filename)
                print("\tNew best valid loss!")

            if last_epoch_to_improve + args.patience < epoch:
                break

        _, test_f1 = util.valid_slot(best_model, test, criterion, cuda)
        print(f"Test F1: {test_f1:.5f}")
    elif args.model == 'joint':
        for epoch in range(0, args.epochs):
            start_time = time.time()
            train_loss, (intent_train_loss, intent_train_acc), (slot_train_loss, slot_train_f1) = util.train_joint(model, train, criterion, optimizer, cuda, args.alpha)
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
                torch.save(model.state_dict(), model_filename)
                print("\tNew best valid loss!")

            if last_epoch_to_improve + args.patience < epoch:
                break
        _, (_, intent_test_acc), (_, slot_test_f1) = util.valid_joint(best_model, test, criterion, cuda, args.alpha)
        print(f"Test Intent Acc: {intent_test_acc:.5f}, Slot F1: {slot_test_f1:.5f}")
        
