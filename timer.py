import time
import argparse

import torch
import sklearn
import numpy as np

import dataset
import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['intent', 'slot', 'joint'])
    parser.add_argument('--dataset', choices=['atis', 'snips'])
    parser.add_argument('--filters', type=int)
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--gpu', action="store_true") # pass for CUDA, otherwise will run on CPU
    
    args = parser.parse_args()
    args.dropout = 0
    
    train, valid, test, num_words, num_intent, num_slot, wordvecs = dataset.load(args.dataset, batch_size=8, seq_len=50)

    # change the model below
    
    
    

    for sparsity in [0, 20, 40, 60, 80, 90, 95, 99]:
        filters = 300 - 300 * (sparsity / 100)
        model = util.load_model(args.model, num_words, num_intent, num_slot, args.dropout, wordvecs, 100, int(filters))
        if args.gpu:
            model = model.cuda()
        print(f"sparsity {sparsity}, params {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    times = []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    for i in range(args.runs):
        start_time = time.time()
        if args.model == 'intent':
            util.valid_intent(model, test, criterion, args.gpu)
        elif args.model == 'slot':
            util.valid_slot(model, test, criterion, args.gpu)
        elif args.model == 'joint':
            util.valid_joint(model, test, criterion, args.gpu, 0.2)
        end_time = time.time()

        elapsed_time = end_time - start_time

        times.append(elapsed_time)

    print(f"mean: {sum(times) / float(len(times)):.5f} sec, std deviation: {np.std(times):.5f}")


