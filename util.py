import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score

import models

kernel_size = 5

def load_model(model_name, num_words, num_intent, num_slot, dropout, wordvecs=None, embedding_dim=100, filter_count=300):
    if model_name == 'intent':
        model = models.CNNIntent(num_words, embedding_dim, num_intent, (filter_count,), kernel_size, dropout, wordvecs)
    elif model_name == 'slot':
        model = models.CNNSlot(num_words, embedding_dim, num_slot, (filter_count,), kernel_size, dropout, wordvecs)
    elif model_name == 'joint':
        model = models.CNNJoint(num_words, embedding_dim, num_intent, num_slot, (filter_count,), kernel_size, dropout, wordvecs)
    return model

def rep(seed=None):
    if not seed:
        seed = random.randint(0, 10000)

    torch.manual_seed(seed)
    np.random.seed(seed)
        
    # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def train_intent(model, iter, criterion, optimizer, cuda):
    model.train()
    epoch_loss = 0
    true_intents = []
    pred_intents = []

    for i, batch in enumerate(iter):
        optimizer.zero_grad()
        query = batch[0]
        true_intent = batch[1]

        if cuda:
            query = query.cuda()
            true_intent = true_intent.cuda()
        
        pred_intent = model(query)

        true_intents += true_intent.tolist()
        pred_intents += pred_intent.max(1)[1].tolist()

        loss = criterion(pred_intent, true_intent)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iter), accuracy_score(true_intents, pred_intents)


def distill_intent(teacher, student, temperature, iter, criterion, optimizer, cuda):
    teacher.eval()
    student.train()

    true_intents = []
    pred_intents = []
    
    epoch_loss = 0
    for i, batch in enumerate(iter):
        optimizer.zero_grad()
        query = batch[0]
        true_intent = batch[1]

        if cuda:
            query = query.cuda()
            true_intent = true_intent.cuda()

        with torch.no_grad():
            teacher_pred_intent = teacher(query)

        student_pred_intent = student(query)

        true_intents += true_intent.tolist()
        pred_intents += student_pred_intent.max(1)[1].tolist()

        loss = criterion(F.log_softmax(student_pred_intent / temperature, dim=-1), F.softmax(teacher_pred_intent / temperature, dim=-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iter), accuracy_score(true_intents, pred_intents)

def valid_intent(model, iter, criterion, cuda):
    model.eval()
    epoch_loss = 0
    true_intents = []
    pred_intents = []

    for i, batch in enumerate(iter):
        query = batch[0]
        true_intent = batch[1]

        if cuda:
            query = query.cuda()
            true_intent = true_intent.cuda()
        
        pred_intent = model(query)

        true_intents += true_intent.tolist()
        pred_intents += pred_intent.max(1)[1].tolist()

        loss = criterion(pred_intent, true_intent)
        epoch_loss += loss.item()
    
    return epoch_loss / len(iter), accuracy_score(true_intents, pred_intents)
        

def train_slot(model, iter, criterion, optimizer, cuda):
    model.train()
    epoch_loss = 0
    true_history = []
    pred_history = []

    for i, batch in enumerate(iter):
        optimizer.zero_grad()
        query = batch[0]
        true_slots = batch[2]
        true_length = batch[3]

        if cuda:
            query = query.cuda()
            true_slots = true_slots.cuda()
        
        pred_slots = model(query).permute(0, 2, 1) # batch * slots * seq len

        true_history += [str(item) for batch_num, sublist in enumerate(true_slots.tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        pred_history += [str(item) for batch_num, sublist in enumerate(pred_slots.max(1)[1].tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]

        loss = criterion(pred_slots, true_slots)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iter), f1_score(true_history, pred_history)

def distill_slot(teacher, student, temperature, iter, criterion, optimizer, cuda):
    teacher.eval()
    student.train()

    true_history = []
    pred_history = []

    epoch_loss = 0
    for i, batch in enumerate(iter):
        optimizer.zero_grad()
        query = batch[0]
        true_slots = batch[2]
        true_length = batch[3]

        if cuda:
            query = query.cuda()
            true_slots = true_slots.cuda()
            true_length = true_length.cuda()

        with torch.no_grad():
            teacher_pred_slot = teacher(query).permute(0, 2, 1) # batch * slot * seq len

        student_pred_slot = student(query).permute(0, 2, 1)

        true_history += [str(item) for batch_num, sublist in enumerate(true_slots.tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        pred_history += [str(item) for batch_num, sublist in enumerate(student_pred_slot.max(1)[1].tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        
        loss = criterion(F.log_softmax(student_pred_slot / temperature, dim=1), F.softmax(teacher_pred_slot / temperature, dim=1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iter), f1_score(true_history, pred_history)

def valid_slot(model, iter, criterion, cuda):
    model.eval()
    epoch_loss = 0
    true_history = []
    pred_history = []

    for i, batch in enumerate(iter):
        query = batch[0]
        true_slots = batch[2]
        true_length = batch[3]

        if cuda:
            query = query.cuda()
            true_slots = true_slots.cuda()
        
        pred_slots = model(query).permute(0, 2, 1) # batch * slots * seq len
        
        true_history += [str(item) for batch_num, sublist in enumerate(true_slots.tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        pred_history += [str(item) for batch_num, sublist in enumerate(pred_slots.max(1)[1].tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]

        loss = criterion(pred_slots, true_slots)
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iter), f1_score(true_history, pred_history)


def train_joint(model, iter, criterion, optimizer, cuda, alpha):
    model.train()
    epoch_loss = 0

    epoch_intent_loss = 0
    true_intents = []
    pred_intents = []

    epoch_slot_loss = 0
    true_history = []
    pred_history = []

    for i, batch in enumerate(iter):
        optimizer.zero_grad()
        query = batch[0]
        true_intent = batch[1]
        true_slots = batch[2]
        true_length = batch[3]

        if cuda:
            query = query.cuda()
            true_intent = true_intent.cuda()
            true_slots = true_slots.cuda()
            true_length = true_length.cuda()

        pred_intent, pred_slots = model(query)
        
        true_intents += true_intent.tolist()
        pred_intents += pred_intent.max(1)[1].tolist()
        intent_loss = criterion(pred_intent, true_intent)
        epoch_intent_loss += intent_loss

        #pred_slots.permute(0, 2, 1)
        true_history += [str(item) for batch_num, sublist in enumerate(true_slots.tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        pred_history += [str(item) for batch_num, sublist in enumerate(pred_slots.max(1)[1].tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        slot_loss = criterion(pred_slots, true_slots)
        epoch_slot_loss += slot_loss

        loss = alpha * intent_loss + (1 - alpha) * slot_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    return (epoch_loss / len(iter),
            (epoch_intent_loss / len(iter), accuracy_score(true_intents, pred_intents)),
            (epoch_slot_loss / len(iter), f1_score(true_history, pred_history)))

def distill_joint(teacher, student, temperature, iter, criterion, optimizer, cuda, alpha):
    teacher.eval()
    student.train()

    epoch_loss = 0

    epoch_intent_loss = 0
    true_intents = []
    pred_intents = []

    epoch_slot_loss = 0
    true_history = []
    pred_history = []

    for i, batch in enumerate(iter):
        optimizer.zero_grad()
        query = batch[0]
        true_intent = batch[1]
        true_slots = batch[2]
        true_length = batch[3]

        if cuda:
            query = query.cuda()
            true_intent = true_intent.cuda()
            true_slots = true_slots.cuda()
            true_length = true_length.cuda()

        with torch.no_grad():
            teacher_pred_intent, teacher_pred_slot = teacher(query)

        student_pred_intent, student_pred_slot = student(query)

        true_intents += true_intent.tolist()
        pred_intents += student_pred_intent.max(1)[1].tolist()
        intent_loss = criterion(F.log_softmax(student_pred_intent / temperature, dim=-1), F.softmax(teacher_pred_intent / temperature, dim=-1))
        epoch_intent_loss += intent_loss

        true_history += [str(item) for batch_num, sublist in enumerate(true_slots.tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        pred_history += [str(item) for batch_num, sublist in enumerate(student_pred_slot.max(1)[1].tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        slot_loss = criterion(F.log_softmax(student_pred_slot / temperature, dim=1), F.softmax(teacher_pred_slot / temperature, dim=1))
        epoch_slot_loss += slot_loss
        
        loss = alpha * intent_loss + (1 - alpha) * slot_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return (epoch_loss / len(iter),
            (epoch_intent_loss / len(iter), accuracy_score(true_intents, pred_intents)),
            (epoch_slot_loss / len(iter), f1_score(true_history, pred_history)))

def valid_joint(model, iter, criterion, cuda, alpha):
    model.eval()
    epoch_loss = 0

    epoch_intent_loss = 0
    true_intents = []
    pred_intents = []

    epoch_slot_loss = 0
    true_history = []
    pred_history = []

    for i, batch in enumerate(iter):
        query = batch[0]
        true_intent = batch[1]
        true_slots = batch[2]
        true_length = batch[3]

        if cuda:
            query = query.cuda()
            true_intent = true_intent.cuda()
            true_slots = true_slots.cuda()
            true_length = true_length.cuda()

        pred_intent, pred_slots = model(query)
        
        true_intents += true_intent.tolist()
        pred_intents += pred_intent.max(1)[1].tolist()
        intent_loss = criterion(pred_intent, true_intent)
        epoch_intent_loss += intent_loss

        #pred_slots.permute(0, 2, 1)
        true_history += [str(item) for batch_num, sublist in enumerate(true_slots.tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        pred_history += [str(item) for batch_num, sublist in enumerate(pred_slots.max(1)[1].tolist()) for item in sublist[1:true_length[batch_num].item() + 1]]
        slot_loss = criterion(pred_slots, true_slots)
        epoch_slot_loss += slot_loss

        loss = alpha * intent_loss + (1 - alpha) * slot_loss
        epoch_loss += loss.item()

    return (epoch_loss / len(iter),
            (epoch_intent_loss / len(iter), accuracy_score(true_intents, pred_intents)),
            (epoch_slot_loss / len(iter), f1_score(true_history, pred_history)))


