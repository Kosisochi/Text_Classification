#TRYING TO DEBUG DAVIDSON USING FOUNTA CODE
#this contains the new train-val-test-split and a fixed filter of 3*1 in the second layer of the 2CNN

#Project Interpreter: /local/scratch/miniconda3/envs/untitled1/bin/python3.7

import numpy as np
np.random.seed(12)      # seeding
import pandas as pd
import torch
torch.manual_seed(0)   #You can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
import sys
import math
from torch.nn.utils.rnn import pack_padded_sequence
import time
import random
from numpy import binary_repr
# from transformers import AdamW
# from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, matthews_corrcoef,f1_score, precision_score, recall_score, roc_auc_score

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# BERT_layer = []
# for b in range (1,13):
#     BERT_layer.append(b)

BERT_layer = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}

CNN1_dropout =  {0:0.0, 1:0.05, 2:0.1, 3:0.15, 4:0.2, 5:0.25, 6:0.3, 7:0.35, 8:0.4, 9:0.45, 10:0.5}
CNN1_kernel_size_a = {0:2, 1:3, 2:4, 3:5}
BiLSTM1_dropout = {0:0.0, 1:0.05, 2:0.1, 3:0.15, 4:0.2, 5:0.25, 6:0.3, 7:0.35, 8:0.4, 9:0.45, 10:0.5}

BiLSTM1_output_neuron = []
for a in range(1,769):
    BiLSTM1_output_neuron.append(a)

Select_model_architecture = []
for b in range (0,4):
    Select_model_architecture.append(b)

#Final_Activation= {0: nn.Softmax(dim=1) , 1:nn.Sigmoid()}

Final_Activation = []
for x in range (0,2):
    Final_Activation.append(x)
#
individuals= 20
population = []
for i in range(individuals):
    # B = binary_repr(random.choice(BERT_layer), width=4)
    # B = [int(x) for x in B]
    # Ba = np.array(B)

    A_random_key = random.choice(list(BERT_layer.keys()))
    Aa = binary_repr(A_random_key, width=4)
    Aa = [int(x) for x in Aa]
    Ba = np.array(Aa)

    D_random_key = random.choice(list(CNN1_dropout.keys()))
    D = binary_repr(D_random_key, width=4)
    D = [int(x) for x in D]
    Da = np.array(D)

    Db_random_key = random.choice(list(CNN1_kernel_size_a.keys()))
    Db = binary_repr(Db_random_key, width=2)
    Db = [int(x) for x in Db]
    Db = np.array(Db)

    E_random_key =  random.choice(list(BiLSTM1_dropout.keys()))
    E = binary_repr(E_random_key, width=4)
    E = [int(x) for x in E]
    Ea = np.array(E)

    Eb = binary_repr(random.choice(BiLSTM1_output_neuron), width=10)
    Eb = [int(x) for x in Eb]
    Eb = np.array(Eb)

    #####################    SELECT MODEL ARCHITECTURE  ##########################
    B1 = binary_repr(random.choice(Select_model_architecture), width=2)
    B1 = [int(x) for x in B1]
    Ba1 = np.array(B1)

    # F_random_key = random.choice(list(Final_Activation.keys()))
    # F = binary_repr(F_random_key, width=1)
    # F = [int(x) for x in F]
    # F1 = np.array(F)

    F_random_key = binary_repr(random.choice(Final_Activation), width=1)
    # F = binary_repr(F_random_key, width=1)
    F = [int(x) for x in F_random_key]
    F1 = np.array(F)

    rep_array = np.concatenate((Ba, Da, Db, Ea, Eb, Ba1, F1), axis=None)
    population.append(rep_array)

pop = np.array(population)
print('shape of the population',pop.shape)
# pop_size = pop.shape


def pad_sents(sents, pad_token):  #Pad list of sentences according to the longest sentence in the batch.
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded


def sents_to_tensor(tokenizer, sents, device):
    tokens_list = [tokenizer.tokenize(str(sent)) for sent in sents]
    sents_lengths = [len(tokens) for tokens in tokens_list]
    # tokens_sents_zip = zip(tokens_list, sents_lengths)
    # tokens_sents_zip = sorted(tokens_sents_zip, key=lambda x: x[1], reverse=True)
    # tokens_list, sents_lengths = zip(*tokens_sents_zip)
    tokens_list_padded = pad_sents(tokens_list, '[PAD]')
    sents_lengths = torch.tensor(sents_lengths, device=device)

    masks = []
    for tokens in tokens_list_padded:
        mask = [0 if token=='[PAD]' else 1 for token in tokens]
        masks.append(mask)
    masks_tensor = torch.tensor(masks, dtype=torch.long, device=device)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long, device=device)

    return sents_tensor, masks_tensor, sents_lengths #sents_tensor is the id after tokenization, mask_tensor contans 1 and 0s, sent_lengths contains length before padding

class CustomBertLSTMModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, lstm_hidden_size=None):

        super(CustomBertLSTMModel, self).__init__()

        #self.bert_config = bert_config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config =self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = hidden_encoded_layer.permute(1, 0, 2)   #permute rotates the tensor. if tensor.shape = 3,4,5  tensor.permute(1,0,2), then tensor,shape= 4,3,5  (batch_size, sequence_length, hidden_size)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(hidden_encoded_layer, sents_lengths, enforce_sorted=False)) #enforce_sorted=False  #pack_padded_sequence(data and batch_sizes
        #enc_hiddens.data.shape: (batch_sum_seq_len X hidden_dim)
        #what is the size of last_hidden and why do we need to concat the [0] and [1]
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

class CustomBert2LSTMModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, lstm_hidden_size=None):

        super(CustomBert2LSTMModel, self).__init__()

        #self.bert_config = bert_config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config =self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, num_layers=2, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = hidden_encoded_layer.permute(1, 0, 2)   #permute rotates the tensor. if tensor.shape = 3,4,5  tensor.permute(1,0,2), then tensor,shape= 4,3,5  (batch_size, sequence_length, hidden_size)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(hidden_encoded_layer, sents_lengths, enforce_sorted=False)) #enforce_sorted=False  #pack_padded_sequence(data and batch_sizes
        #enc_hiddens.data.shape: (batch_sum_seq_len X hidden_dim)
        #what is the size of last_hidden and why do we need to concat the [0] and [1]
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

class CustomBertConvModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, CNN1_KS, out_channel=1):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param out_channel: int, NOTE: out_channel per layer of BERT
        """

        super(CustomBertConvModel, self).__init__()

        #self.bert_config = bert_config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.CNN1_KS = CNN1_KS
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.out_channels = self.bert.config.num_hidden_layers*self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.conv = nn.Conv2d(in_channels=self.bert.config.num_hidden_layers,
                              out_channels=self.out_channels,
                              kernel_size=(self.CNN1_KS, self.bert.config.hidden_size),
                              groups=self.bert.config.num_hidden_layers)
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self,BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = torch.unsqueeze(hidden_encoded_layer, dim=1)
        hidden_encoded_layer = hidden_encoded_layer.repeat(1, 12, 1, 1)
        conv_out = self.conv(hidden_encoded_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax

class CustomBert2ConvModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, CNN1_KS, out_channel=16):
        super(CustomBert2ConvModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.CNN1_KS = CNN1_KS
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.out_channels = self.bert.config.num_hidden_layers * self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.conv = nn.Conv2d(in_channels=self.bert.config.num_hidden_layers, out_channels=self.out_channels,
                              kernel_size=(self.CNN1_KS, 768),
                              groups=self.bert.config.num_hidden_layers)  # self.bert.config.num_hidden_layers
        self.conv1 = nn.Conv2d(in_channels=self.out_channels, out_channels=192, kernel_size=(3,3), padding=1,
                               groups=self.bert.config.num_hidden_layers)  # self.bert.config.num_hidden_layers
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self,BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        # print(hidden_encoded_layer.dtype)
        # hidden_encoded_layer = hidden_encoded_layer[0]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = torch.unsqueeze(hidden_encoded_layer, dim=1)
        hidden_encoded_layer = hidden_encoded_layer.repeat(1, 12, 1, 1)
        conv_out = self.conv(hidden_encoded_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = self.conv1(conv_out)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        # conv_out = self.conv1(conv_out)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax


def batch_iter(data, batch_size, shuffle=False, bert=None):
    batch_num = math.ceil(data.shape[0] / batch_size)
    index_array = list(range(data.shape[0]))

    if shuffle:
        data = data.sample(frac=1)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]

        examples = data.iloc[indices] #.sort_values(by='ProcessedText_BERT'+bert+'_length', ascending=False)
        sents = list(examples.train_BERT_tweet)

        targets = list(examples.label.values)
        yield sents, targets  # list[list[str]] if not bert else list[str], list[int]



def validation(model, df_val, loss_func, device):

    was_training = model.training
    model.eval()

    #df_val = df_val.sort_values(by='ProcessedText_BERT'+bert_size+'_length', ascending=False)

    train_BERT_tweet = list(df_val.train_BERT_tweet)
    train_label = list(df_val.label)
    val_batch_size = 32

    n_batch = int(np.ceil(df_val.shape[0]/val_batch_size))

    total_loss = 0.

    with torch.no_grad():
        for i in range(n_batch):
            sents =  train_BERT_tweet[i*val_batch_size: (i+1)*val_batch_size]
            targets = torch.tensor(train_label[i*val_batch_size: (i+1)*val_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)
            pre_softmax = model(sents)
            batch_loss = loss_func(pre_softmax, targets)
            total_loss += batch_loss.item()*batch_size

    if was_training:
        model.train()

    return total_loss/df_val.shape[0]



def train(Model_arch_select,BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL):
    label_name = ['Hate', 'Offensive', 'Neutral']
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    start_time = time.time()
    print('Importing data...', file=sys.stderr)
    df_train = pd.read_csv('new_train_davidsonBERT.csv') #, index_col=0)
    #df_val = pd.read_csv('founta_val.csv')   #, index_col=0)
    train_label = dict(df_train.label.value_counts())
    #print('train label', train_label)   #train label {1: 8008, 2: 1801, 0: 600}
    label_max = float(max(train_label.values()))
    #print('label max', label_max)  #label max 8008.0
    train_label_weight = torch.tensor([label_max/train_label[i] for i in range(len(train_label))], device=device)

    # print('train label weight',train_label_weight)  #train label weight tensor([13.3467,  1.0000,  4.4464])
    # print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    # print('-' * 80, file=sys.stderr)

    start_time = time.time()
    print('Set up model...', file=sys.stderr)

    if Model_arch_select == 0:
        model = CustomBertLSTMModel(device=device, dropout_rate=BiLSTM1_D, n_class=len(label_name),lstm_hidden_size=BiLSTM1_ON)
        print('LSTM')

    elif Model_arch_select == 1:
        model= CustomBert2LSTMModel(device=device, dropout_rate=BiLSTM1_D, n_class=len(label_name),lstm_hidden_size=BiLSTM1_ON)
        print('2LSTM')
    elif Model_arch_select == 2:
        model = CustomBertConvModel(device=device, dropout_rate=CNN1_D, n_class=3, CNN1_KS=CNN1_KS, out_channel=1)
        print('CNN')
    else:
        model = CustomBert2ConvModel(device=device, dropout_rate=CNN1_D, n_class=3, CNN1_KS=CNN1_KS)
        print('2CNN')

    optimizer = AdamW(model.parameters(), lr=1e-3, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)  #changed the last 2 arguments to old ones
    #warmup_steps = 100, t_total = 1000
    #num_warmup_steps=100, num_training_steps=1000
    model = model.to(device)
    print('Use device: %s' % device, file=sys.stderr)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    model.train()

    cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight.float(), reduction='mean')
    #cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight, reduction='elementwise_mean')
    torch.save(cn_loss, 'loss_funcD2')  # for later testing

    train_batch_size = 32
    valid_niter = 500
    log_every = 10
    #model_save_path = 'LSTM3_bert_uncased_model.bin'

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Begin Maximum Likelihood training...')
    #model = model.float()  #=============================================================================== i added this whole line
    for epoch in range(15):

        for sents, targets in batch_iter(df_train, batch_size=train_batch_size, shuffle=True):  # for each epoch
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(sents)
            pre_softmax = model(BERT_layer_value, sents).float()   #y_p red = model(x_batch)===============================================================================
            loss = cn_loss(pre_softmax, torch.tensor(targets,dtype=torch.long , device=device))  #loss = loss_fn(y_pred, y_batch) dtype=np.float32 dtype=torch.long
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses_val = loss.item() * batch_size
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, '
                      'cum. examples %d, speed %.2f examples/sec, '
                      'time elapsed %.2f sec' % (epoch, train_iter,report_loss / report_examples,cum_examples,report_examples / (time.time() - train_time),time.time() - begin_time), file=sys.stderr)
                train_time = time.time()
                report_loss = report_examples = 0.

    model.eval()
    df_test = pd.read_csv('new_val_davidsonBERT.csv')
    test_batch_size = 32
    n_batch = int(np.ceil(df_test.shape[0]/test_batch_size))
    cn_loss = torch.load('loss_funcD2', map_location=lambda storage, loc: storage).to(device)
    train_BERT_tweet = list(df_test.train_BERT_tweet)
    train_label = list(df_test.label)
    test_loss = 0.
    prediction = []
    prob = []
    if Final_AL == 0:
        softmax = nn.Softmax(dim=1)
    else:
        softmax= nn.Sigmoid()

    with torch.no_grad():
        for i in range(n_batch):
            sents = train_BERT_tweet[i*test_batch_size: (i+1)*test_batch_size]
            targets = torch.tensor(train_label[i * test_batch_size: (i + 1) * test_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)

            pre_softmax = model(BERT_layer_value, sents) #BERT_layer.get(decimal_value1[0])
            batch_loss = cn_loss(pre_softmax, targets)
            test_loss += batch_loss.item()*batch_size
            prob_batch = softmax(pre_softmax)
            prob.append(prob_batch)

            prediction.extend([t.item() for t in list(torch.argmax(prob_batch, dim=1))])

    accuracy = accuracy_score(df_test.label.values, prediction)
    matthews = matthews_corrcoef(df_test.label.values, prediction)
    f1_macro = f1_score(df_test.label.values, prediction, average='macro')
    f1_micro = f1_score(df_test.label.values, prediction, average='micro')
    precision_macro = precision_score(df_test.label.values, prediction, average='macro')
    recall_macro = recall_score(df_test.label.values, prediction, average='macro')
    print('accuracy: %.2f' % accuracy)
    print('matthews coef: %.2f' % matthews)
    print('f1_macro: %.2f' % f1_macro)
    print('f1_micro: %.2f' % f1_micro)
    print('precision macro: %.2f' % precision_macro)
    print('recall macro: %.2f' % recall_macro)

    return  f1_macro


def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def calc_fitness(pop2):
    f1_scores = []
    for i in range(pop2.shape[0]):
        print('Solution %d' % i)
        # pick the first array in pop
        solution = pop2[i]
        solution1 = np.reshape(solution, (1, pop2.shape[1]))

        # s1 = solution1[:, 0:4]  # (1,2)  BERT__Layer  #4 bits
        # decimal_value1 = [bool2int(x[::-1]) for x in s1]
        # BERT_layer_value = decimal_value1[0]

        s1 = solution1[:, 0:4]  # (1,2)  BERT_Encoder_Layer  #4 bits
        decimal_value1 = [bool2int(x[::-1]) for x in s1]
        BERT_layer_value = BERT_layer.get(decimal_value1[0], 12)


        s13 = solution1[:, 4:7]  # (1,2)  CNN1_dropout #4 bits
        decimal_value13 = [bool2int(x[::-1]) for x in s13]
        CNN1_D = CNN1_dropout.get(decimal_value13[0], 0.1)

        s14 = solution1[:, 7:9]  # (1,2)  CNN1_kernel_size_a #3 bits
        decimal_value14 = [bool2int(x[::-1]) for x in s14]
        CNN1_KS = CNN1_kernel_size_a.get(decimal_value14[0], 3)

        s23 = solution1[:, 9:13]  # (1,2)  BiLSTM1_dropout  #4 bits
        decimal_value23 = [bool2int(x[::-1]) for x in s23]
        BiLSTM1_D = BiLSTM1_dropout.get(decimal_value23[0], 0.1)

        s24 = solution1[:, 13:23]  # (1,2) BiLSTM1_output_neuron #10 bits
        decimal_value24 = [bool2int(x[::-1]) for x in s24]
        BiLSTM1_ON = int(decimal_value24[0])

        s2 = solution1[:, 23:25]  # (1,2)  Select_Architecture_Design  #2 bits
        decimal_value2 = [bool2int(x[::-1]) for x in s2]
        Model_arch_select = decimal_value2[0]

        s3 = solution1[:, 25:26]  #1 bit
        decimal_value27 = [bool2int(x[::-1]) for x in s3]
        Final_AL = decimal_value27[0]

        f1_macro= train(Model_arch_select, BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL)
        # print(Train_Model)
        # f1_macro  = test(Model_arch_select, BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL)
        f1_scores.append(f1_macro)

    return f1_scores

def returns_f1_scores(f1_scores):
    return f1_scores

def elitism(f1_scores, pop2):
    max_fitness_idx = np.where(f1_scores == np.max(f1_scores))
    max_fitness_idx1 = max_fitness_idx[0][0]

    elite_mem = pop2[max_fitness_idx1, :]

    return elite_mem
    #append this solution to the new population


def tournament_selection(pop2,f1_scores):
    import random
    best_indv= np.empty((1,pop2.shape[1]))
    best_indv_fitness=0
    for i in range(2):
        A = random.randint(0, pop2.shape[0]-1)
        current_indv= pop2[A]
        current_indv_fitness = f1_scores[A]
        if best_indv_fitness == 0 or current_indv_fitness > best_indv_fitness:
            best_indv = current_indv
            best_indv_fitness = current_indv_fitness
    return best_indv


def create_parent(pop2,f1_scores):
    selected_parents = np.empty((10,pop2.shape[1])) #offspring_size
    for k in range(10): #offspring_size
        parentA = tournament_selection(pop2, f1_scores)
        parentA_list = parentA.tolist()
        selected_parents[k,0:pop2.shape[1]] = parentA_list
    print(selected_parents.shape)
    return selected_parents

def crossover_new(selected_parents, offspring_size):  # offspring size ahould be 2
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        selected_parents1_idx = k % selected_parents.shape[0]
        # Index of the second parent to mate.
        selected_parents2_idx = (k + 1) % selected_parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = selected_parents[selected_parents1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = selected_parents[selected_parents2_idx, crossover_point:]
    return offspring  # this will return 2 new individuals


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    from random import random
    for k in range(0, offspring_crossover.shape[0]):
        for j in range(0, offspring_crossover.shape[1]):
            A = random()
            if A > 0.7:
                if offspring_crossover[k, j] == 0:
                    offspring_crossover[k, j] = 1
                else:
                    offspring_crossover[k, j] = 0
            else:
                continue
    return offspring_crossover

from random import random
num_generations = 20
for generation in range(num_generations):
    t3 = time.time()
    print("Generation : ", generation)
    t0 = time.time()
    f1_scores = calc_fitness(pop)
    print("Time it takes to calculate fitness of one population:", round(time.time() - t0, 3), 's')
    print("F1_scores", f1_scores)
    print("Best result for each generation : ", np.max(f1_scores))

    elite_mem = elitism(f1_scores, pop)
    elite_mem_list = elite_mem.tolist()
    j = 0
    for i in range(0, 10):
        j = j + 2
        if i <= 8:
            # call the tournament selection twice to give the parents that will be part of the mating
            parent1 = tournament_selection(pop, f1_scores)
            parent2 = tournament_selection(pop, f1_scores)

            selected_parents = np.empty((2, pop.shape[1]))
            selected_parents[0] = parent1
            selected_parents[1] = parent2

            # generate a random number as the crossover_probability
            crossover_probability = random()
            if crossover_probability <= 0.9:
                # do crossover
                two_new_indv = crossover_new(selected_parents, (2, 27))
                # also check mutation probability
                mutation_probability = random()
                if mutation_probability <= 0.3:
                    # do mutation
                    mutation_output = mutation(two_new_indv)
                    # send to the new population
                    pop[j - 2:j, :] = two_new_indv  #######################################
                else:
                    # send parent directly to new population
                    pop[j - 2:j, :] = selected_parents  #######################################
            else:
                # go straight to mutation
                # also check mutation probability
                mutation_probability = random()
                if mutation_probability <= 0.3:
                    # do mutation
                    mutation_output = mutation(selected_parents)
                    pop[j - 2:j, :] = mutation_output  #######################################
                else:
                    # send parent directly to new population
                    pop[j - 2:j, :] = selected_parents  #######################################
    else:
        # call the tournament selection twice to give the parents that will be part of the mating
        parent1 = tournament_selection(pop, f1_scores)
        parent2 = tournament_selection(pop, f1_scores)

        selected_parents = np.empty((2, pop.shape[1]))
        selected_parents[0] = parent1
        selected_parents[1] = parent2

        # generate a random number as the crossover_probability
        crossover_probability = random()
        if crossover_probability <= 0.9:
            # do crossover
            two_new_indv = crossover_new(selected_parents, (2, 27))
            # also check mutation probability
            mutation_probability = random()
            if mutation_probability <= 0.3:
                # do mutation
                mutation_output = mutation(two_new_indv)
                # send to the new population
                pop[j - 2, :] = two_new_indv[0]  #######################################
            else:
                # send parent directly to new population
                pop[j - 2, :] = selected_parents[0]  #######################################
        else:
            # go straight to mutation
            # also check mutation probability
            mutation_probability = random()
            if mutation_probability <= 0.3:
                # do mutation
                mutation_output = mutation(selected_parents)
                pop[j - 2, :] = mutation_output[0]  #######################################
            else:
                # send parent directly to new population
                pop[j - 2, :] = selected_parents[0]  #######################################
        pop[j - 1] = elite_mem_list
    print(pop)


# f1_scores=returns_f1_scores(f1_scores)
best_match_idx = np.where(f1_scores== np.max(f1_scores))
print("BMI",best_match_idx)
here= best_match_idx[0][0]
print("Best solution in the final generation: ", pop[here, :])#TRYING TO DEBUG DAVIDSON USING FOUNTA CODE
#this contains the new train-val-test-split and a fixed filter of 3*1 in the second layer of the 2CNN

#Project Interpreter: /local/scratch/miniconda3/envs/untitled1/bin/python3.7


######### THIS IS CODE USED FOR THE WORK IN THE PAPER #################
#eRROR IN gpu During the 3rd run . Results saved in Davidson_ThirdRun

#
#   File "Davidson2.py", line 1379, in <module>
#     f1_scores = calc_fitness(pop2)
#   File "Davidson2.py", line 1284, in calc_fitness
#     f1_macro= train(Model_arch_select, BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL)
#   File "Davidson2.py", line 1173, in train
# #     pre_softmax = model(BERT_layer_value, sents).float()   #y_p red = model(x_batch)===============================================================================
#   File "/usr/pkg/lib/python3.8/site-packages/torch/nn/modules/module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "Davidson2.py", line 1042, in forward
#     conv_out = self.conv1(conv_out)
#   File "/usr/pkg/lib/python3.8/site-packages/torch/nn/modules/module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/usr/pkg/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 349, in forward
#     return self._conv_forward(input, self.weight)
#   File "/usr/pkg/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 345, in _conv_forward
#     return F.conv2d(input, weight, self.bias, self.stride,
# RuntimeError: Calculated padded input size per channel: (2 x 1). Kernel size: (3 x 1). Kernel size can't be greater than actual input size
#
#




#f1_scores = calc_fitness(pop2)

# File"Davidson1.py", line668, in calc_fitness
# f1_macro = train(Model_arch_select, BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL)

# File "Davidson1.py", line558, in train
# loss = cn_loss(pre_softmax, torch.tensor(targets, dtype=torch.long, device=device))  # loss = loss_fn(y_pred, y_batch)
#
# File "/usr/pkg/lib/python3.8/site-packages/torch/nn/modules/module.py", line 550, in __call__
# result = self.forward(*input, **kwargs)
# File "/usr/pkg/lib/python3.8/site-packages/torch/nn/modules/loss.py", line931, in forward
# return F.cross_entropy(input, target, weight=self.weight,
# File "/usr/pkg/lib/python3.8/site-packages/torch/nn/functional.py", line 2317, in cross_entropy
# return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
# File"/usr/pkg/lib/python3.8/site-packages/torch/nn/functional.py", line2115, in nll_loss
# ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

# RuntimeError: Expected object of scalar type Float but got scalar type Double for argument  # 3 'weight' in call to _thnn_nll_loss_forward

import numpy as np
np.random.seed(12)      # seeding
import pandas as pd
import torch
torch.manual_seed(0)   #You can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
import sys
import math
from torch.nn.utils.rnn import pack_padded_sequence
import time
import random
from numpy import binary_repr
# from transformers import AdamW
# from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, matthews_corrcoef,f1_score, precision_score, recall_score, roc_auc_score

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# BERT_layer = []
# for b in range (1,13):
#     BERT_layer.append(b)

BERT_layer = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}

CNN1_dropout =  {0:0.0, 1:0.05, 2:0.1, 3:0.15, 4:0.2, 5:0.25, 6:0.3, 7:0.35, 8:0.4, 9:0.45, 10:0.5}
CNN1_kernel_size_a = {0:2, 1:3, 2:4, 3:5}
BiLSTM1_dropout = {0:0.0, 1:0.05, 2:0.1, 3:0.15, 4:0.2, 5:0.25, 6:0.3, 7:0.35, 8:0.4, 9:0.45, 10:0.5}

BiLSTM1_output_neuron = []
for a in range(1,769):
    BiLSTM1_output_neuron.append(a)

Select_model_architecture = []
for b in range (0,4):
    Select_model_architecture.append(b)

#Final_Activation= {0: nn.Softmax(dim=1) , 1:nn.Sigmoid()}

Final_Activation = []
for x in range (0,2):
    Final_Activation.append(x)
#
individuals= 20
population = []
for i in range(individuals):
    # B = binary_repr(random.choice(BERT_layer), width=4)
    # B = [int(x) for x in B]
    # Ba = np.array(B)

    A_random_key = random.choice(list(BERT_layer.keys()))
    Aa = binary_repr(A_random_key, width=4)
    Aa = [int(x) for x in Aa]
    Ba = np.array(Aa)

    D_random_key = random.choice(list(CNN1_dropout.keys()))
    D = binary_repr(D_random_key, width=4)
    D = [int(x) for x in D]
    Da = np.array(D)

    Db_random_key = random.choice(list(CNN1_kernel_size_a.keys()))
    Db = binary_repr(Db_random_key, width=2)
    Db = [int(x) for x in Db]
    Db = np.array(Db)

    E_random_key =  random.choice(list(BiLSTM1_dropout.keys()))
    E = binary_repr(E_random_key, width=4)
    E = [int(x) for x in E]
    Ea = np.array(E)

    Eb = binary_repr(random.choice(BiLSTM1_output_neuron), width=10)
    Eb = [int(x) for x in Eb]
    Eb = np.array(Eb)

    #####################    SELECT MODEL ARCHITECTURE  ##########################
    B1 = binary_repr(random.choice(Select_model_architecture), width=2)
    B1 = [int(x) for x in B1]
    Ba1 = np.array(B1)

    # F_random_key = random.choice(list(Final_Activation.keys()))
    # F = binary_repr(F_random_key, width=1)
    # F = [int(x) for x in F]
    # F1 = np.array(F)

    F_random_key = binary_repr(random.choice(Final_Activation), width=1)
    # F = binary_repr(F_random_key, width=1)
    F = [int(x) for x in F_random_key]
    F1 = np.array(F)

    rep_array = np.concatenate((Ba, Da, Db, Ea, Eb, Ba1, F1), axis=None)
    population.append(rep_array)

pop = np.array(population)
print('shape of the population',pop.shape)
# pop_size = pop.shape


def pad_sents(sents, pad_token):  #Pad list of sentences according to the longest sentence in the batch.
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded


def sents_to_tensor(tokenizer, sents, device):
    tokens_list = [tokenizer.tokenize(str(sent)) for sent in sents]
    sents_lengths = [len(tokens) for tokens in tokens_list]
    # tokens_sents_zip = zip(tokens_list, sents_lengths)
    # tokens_sents_zip = sorted(tokens_sents_zip, key=lambda x: x[1], reverse=True)
    # tokens_list, sents_lengths = zip(*tokens_sents_zip)
    tokens_list_padded = pad_sents(tokens_list, '[PAD]')
    sents_lengths = torch.tensor(sents_lengths, device=device)

    masks = []
    for tokens in tokens_list_padded:
        mask = [0 if token=='[PAD]' else 1 for token in tokens]
        masks.append(mask)
    masks_tensor = torch.tensor(masks, dtype=torch.long, device=device)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long, device=device)

    return sents_tensor, masks_tensor, sents_lengths #sents_tensor is the id after tokenization, mask_tensor contans 1 and 0s, sent_lengths contains length before padding

class CustomBertLSTMModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, lstm_hidden_size=None):

        super(CustomBertLSTMModel, self).__init__()

        #self.bert_config = bert_config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config =self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = hidden_encoded_layer.permute(1, 0, 2)   #permute rotates the tensor. if tensor.shape = 3,4,5  tensor.permute(1,0,2), then tensor,shape= 4,3,5  (batch_size, sequence_length, hidden_size)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(hidden_encoded_layer, sents_lengths, enforce_sorted=False)) #enforce_sorted=False  #pack_padded_sequence(data and batch_sizes
        #enc_hiddens.data.shape: (batch_sum_seq_len X hidden_dim)
        #what is the size of last_hidden and why do we need to concat the [0] and [1]
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

class CustomBert2LSTMModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, lstm_hidden_size=None):

        super(CustomBert2LSTMModel, self).__init__()

        #self.bert_config = bert_config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config =self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, num_layers=2, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = hidden_encoded_layer.permute(1, 0, 2)   #permute rotates the tensor. if tensor.shape = 3,4,5  tensor.permute(1,0,2), then tensor,shape= 4,3,5  (batch_size, sequence_length, hidden_size)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(hidden_encoded_layer, sents_lengths, enforce_sorted=False)) #enforce_sorted=False  #pack_padded_sequence(data and batch_sizes
        #enc_hiddens.data.shape: (batch_sum_seq_len X hidden_dim)
        #what is the size of last_hidden and why do we need to concat the [0] and [1]
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

class CustomBertConvModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, CNN1_KS, out_channel=1):
        """

        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param out_channel: int, NOTE: out_channel per layer of BERT
        """

        super(CustomBertConvModel, self).__init__()

        #self.bert_config = bert_config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.CNN1_KS = CNN1_KS
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.out_channels = self.bert.config.num_hidden_layers*self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config =self.bert_config)
        self.conv = nn.Conv2d(in_channels=self.bert.config.num_hidden_layers,
                              out_channels=self.out_channels,
                              kernel_size=(self.CNN1_KS, self.bert.config.hidden_size),
                              groups=self.bert.config.num_hidden_layers)
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self,BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = torch.unsqueeze(hidden_encoded_layer, dim=1)
        hidden_encoded_layer = hidden_encoded_layer.repeat(1, 12, 1, 1)
        conv_out = self.conv(hidden_encoded_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax

class CustomBert2ConvModel(nn.Module):

    def __init__(self, device, dropout_rate, n_class, CNN1_KS, out_channel=16):
        super(CustomBert2ConvModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.CNN1_KS = CNN1_KS
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.out_channels = self.bert.config.num_hidden_layers * self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.conv = nn.Conv2d(in_channels=self.bert.config.num_hidden_layers, out_channels=self.out_channels,
                              kernel_size=(self.CNN1_KS, 768),
                              groups=self.bert.config.num_hidden_layers)  # self.bert.config.num_hidden_layers
        self.conv1 = nn.Conv2d(in_channels=self.out_channels, out_channels=192, kernel_size=(3,3), padding=1,
                               groups=self.bert.config.num_hidden_layers)  # self.bert.config.num_hidden_layers
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self,BERT_layer_value, sents):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        hidden_encoded_layer = encoded_layers[2]
        # print(hidden_encoded_layer.dtype)
        # hidden_encoded_layer = hidden_encoded_layer[0]
        hidden_encoded_layer = hidden_encoded_layer[BERT_layer_value]
        hidden_encoded_layer = torch.unsqueeze(hidden_encoded_layer, dim=1)
        hidden_encoded_layer = hidden_encoded_layer.repeat(1, 12, 1, 1)
        conv_out = self.conv(hidden_encoded_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = self.conv1(conv_out)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        # conv_out = self.conv1(conv_out)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax


def batch_iter(data, batch_size, shuffle=False, bert=None):
    batch_num = math.ceil(data.shape[0] / batch_size)
    index_array = list(range(data.shape[0]))

    if shuffle:
        data = data.sample(frac=1)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]

        examples = data.iloc[indices] #.sort_values(by='ProcessedText_BERT'+bert+'_length', ascending=False)
        sents = list(examples.train_BERT_tweet)

        targets = list(examples.label.values)
        yield sents, targets  # list[list[str]] if not bert else list[str], list[int]



def validation(model, df_val, loss_func, device):

    was_training = model.training
    model.eval()

    #df_val = df_val.sort_values(by='ProcessedText_BERT'+bert_size+'_length', ascending=False)

    train_BERT_tweet = list(df_val.train_BERT_tweet)
    train_label = list(df_val.label)
    val_batch_size = 32

    n_batch = int(np.ceil(df_val.shape[0]/val_batch_size))

    total_loss = 0.

    with torch.no_grad():
        for i in range(n_batch):
            sents =  train_BERT_tweet[i*val_batch_size: (i+1)*val_batch_size]
            targets = torch.tensor(train_label[i*val_batch_size: (i+1)*val_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)
            pre_softmax = model(sents)
            batch_loss = loss_func(pre_softmax, targets)
            total_loss += batch_loss.item()*batch_size

    if was_training:
        model.train()

    return total_loss/df_val.shape[0]



def train(Model_arch_select,BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL):
    label_name = ['Hate', 'Offensive', 'Neutral']
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    start_time = time.time()
    print('Importing data...', file=sys.stderr)
    df_train = pd.read_csv('new_train_davidsonBERT.csv') #, index_col=0)
    #df_val = pd.read_csv('founta_val.csv')   #, index_col=0)
    train_label = dict(df_train.label.value_counts())
    #print('train label', train_label)   #train label {1: 8008, 2: 1801, 0: 600}
    label_max = float(max(train_label.values()))
    #print('label max', label_max)  #label max 8008.0
    train_label_weight = torch.tensor([label_max/train_label[i] for i in range(len(train_label))], device=device)

    # print('train label weight',train_label_weight)  #train label weight tensor([13.3467,  1.0000,  4.4464])
    # print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    # print('-' * 80, file=sys.stderr)

    start_time = time.time()
    print('Set up model...', file=sys.stderr)

    if Model_arch_select == 0:
        model = CustomBertLSTMModel(device=device, dropout_rate=BiLSTM1_D, n_class=len(label_name),lstm_hidden_size=BiLSTM1_ON)
        print('LSTM')

    elif Model_arch_select == 1:
        model= CustomBert2LSTMModel(device=device, dropout_rate=BiLSTM1_D, n_class=len(label_name),lstm_hidden_size=BiLSTM1_ON)
        print('2LSTM')
    elif Model_arch_select == 2:
        model = CustomBertConvModel(device=device, dropout_rate=CNN1_D, n_class=3, CNN1_KS=CNN1_KS, out_channel=1)
        print('CNN')
    else:
        model = CustomBert2ConvModel(device=device, dropout_rate=CNN1_D, n_class=3, CNN1_KS=CNN1_KS)
        print('2CNN')

    optimizer = AdamW(model.parameters(), lr=1e-3, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)  #changed the last 2 arguments to old ones
    #warmup_steps = 100, t_total = 1000
    #num_warmup_steps=100, num_training_steps=1000
    model = model.to(device)
    print('Use device: %s' % device, file=sys.stderr)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    model.train()

    cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight.float(), reduction='mean')
    #cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight, reduction='elementwise_mean')
    torch.save(cn_loss, 'loss_funcD2')  # for later testing

    train_batch_size = 32
    valid_niter = 500
    log_every = 10
    #model_save_path = 'LSTM3_bert_uncased_model.bin'

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Begin Maximum Likelihood training...')
    #model = model.float()  #=============================================================================== i added this whole line
    for epoch in range(15):

        for sents, targets in batch_iter(df_train, batch_size=train_batch_size, shuffle=True):  # for each epoch
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(sents)
            pre_softmax = model(BERT_layer_value, sents).float()   #y_p red = model(x_batch)===============================================================================
            loss = cn_loss(pre_softmax, torch.tensor(targets,dtype=torch.long , device=device))  #loss = loss_fn(y_pred, y_batch) dtype=np.float32 dtype=torch.long
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses_val = loss.item() * batch_size
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, '
                      'cum. examples %d, speed %.2f examples/sec, '
                      'time elapsed %.2f sec' % (epoch, train_iter,report_loss / report_examples,cum_examples,report_examples / (time.time() - train_time),time.time() - begin_time), file=sys.stderr)
                train_time = time.time()
                report_loss = report_examples = 0.

    model.eval()
    df_test = pd.read_csv('new_val_davidsonBERT.csv')
    test_batch_size = 32
    n_batch = int(np.ceil(df_test.shape[0]/test_batch_size))
    cn_loss = torch.load('loss_funcD2', map_location=lambda storage, loc: storage).to(device)
    train_BERT_tweet = list(df_test.train_BERT_tweet)
    train_label = list(df_test.label)
    test_loss = 0.
    prediction = []
    prob = []
    if Final_AL == 0:
        softmax = nn.Softmax(dim=1)
    else:
        softmax= nn.Sigmoid()

    with torch.no_grad():
        for i in range(n_batch):
            sents = train_BERT_tweet[i*test_batch_size: (i+1)*test_batch_size]
            targets = torch.tensor(train_label[i * test_batch_size: (i + 1) * test_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)

            pre_softmax = model(BERT_layer_value, sents) #BERT_layer.get(decimal_value1[0])
            batch_loss = cn_loss(pre_softmax, targets)
            test_loss += batch_loss.item()*batch_size
            prob_batch = softmax(pre_softmax)
            prob.append(prob_batch)

            prediction.extend([t.item() for t in list(torch.argmax(prob_batch, dim=1))])

    accuracy = accuracy_score(df_test.label.values, prediction)
    matthews = matthews_corrcoef(df_test.label.values, prediction)
    f1_macro = f1_score(df_test.label.values, prediction, average='macro')
    f1_micro = f1_score(df_test.label.values, prediction, average='micro')
    precision_macro = precision_score(df_test.label.values, prediction, average='macro')
    recall_macro = recall_score(df_test.label.values, prediction, average='macro')
    print('accuracy: %.2f' % accuracy)
    print('matthews coef: %.2f' % matthews)
    print('f1_macro: %.2f' % f1_macro)
    print('f1_micro: %.2f' % f1_micro)
    print('precision macro: %.2f' % precision_macro)
    print('recall macro: %.2f' % recall_macro)

    return  f1_macro


def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def calc_fitness(pop2):
    f1_scores = []
    for i in range(pop2.shape[0]):
        print('Solution %d' % i)
        # pick the first array in pop
        solution = pop2[i]
        solution1 = np.reshape(solution, (1, pop2.shape[1]))

        # s1 = solution1[:, 0:4]  # (1,2)  BERT__Layer  #4 bits
        # decimal_value1 = [bool2int(x[::-1]) for x in s1]
        # BERT_layer_value = decimal_value1[0]

        s1 = solution1[:, 0:4]  # (1,2)  BERT_Encoder_Layer  #4 bits
        decimal_value1 = [bool2int(x[::-1]) for x in s1]
        BERT_layer_value = BERT_layer.get(decimal_value1[0], 12)


        s13 = solution1[:, 4:7]  # (1,2)  CNN1_dropout #4 bits
        decimal_value13 = [bool2int(x[::-1]) for x in s13]
        CNN1_D = CNN1_dropout.get(decimal_value13[0], 0.1)

        s14 = solution1[:, 7:9]  # (1,2)  CNN1_kernel_size_a #3 bits
        decimal_value14 = [bool2int(x[::-1]) for x in s14]
        CNN1_KS = CNN1_kernel_size_a.get(decimal_value14[0], 3)

        s23 = solution1[:, 9:13]  # (1,2)  BiLSTM1_dropout  #4 bits
        decimal_value23 = [bool2int(x[::-1]) for x in s23]
        BiLSTM1_D = BiLSTM1_dropout.get(decimal_value23[0], 0.1)

        s24 = solution1[:, 13:23]  # (1,2) BiLSTM1_output_neuron #10 bits
        decimal_value24 = [bool2int(x[::-1]) for x in s24]
        BiLSTM1_ON = int(decimal_value24[0])

        s2 = solution1[:, 23:25]  # (1,2)  Select_Architecture_Design  #2 bits
        decimal_value2 = [bool2int(x[::-1]) for x in s2]
        Model_arch_select = decimal_value2[0]

        s3 = solution1[:, 25:26]  #1 bit
        decimal_value27 = [bool2int(x[::-1]) for x in s3]
        Final_AL = decimal_value27[0]

        f1_macro= train(Model_arch_select, BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL)
        # print(Train_Model)
        # f1_macro  = test(Model_arch_select, BERT_layer_value, BiLSTM1_ON, BiLSTM1_D, CNN1_D, CNN1_KS, Final_AL)
        f1_scores.append(f1_macro)

    return f1_scores

def returns_f1_scores(f1_scores):
    return f1_scores

def elitism(f1_scores, pop2):
    max_fitness_idx = np.where(f1_scores == np.max(f1_scores))
    max_fitness_idx1 = max_fitness_idx[0][0]

    elite_mem = pop2[max_fitness_idx1, :]

    return elite_mem
    #append this solution to the new population


def tournament_selection(pop2,f1_scores):
    import random
    best_indv= np.empty((1,pop2.shape[1]))
    best_indv_fitness=0
    for i in range(2):
        A = random.randint(0, pop2.shape[0]-1)
        current_indv= pop2[A]
        current_indv_fitness = f1_scores[A]
        if best_indv_fitness == 0 or current_indv_fitness > best_indv_fitness:
            best_indv = current_indv
            best_indv_fitness = current_indv_fitness
    return best_indv


def create_parent(pop2,f1_scores):
    selected_parents = np.empty((10,pop2.shape[1])) #offspring_size
    for k in range(10): #offspring_size
        parentA = tournament_selection(pop2, f1_scores)
        parentA_list = parentA.tolist()
        selected_parents[k,0:pop2.shape[1]] = parentA_list
    print(selected_parents.shape)
    return selected_parents

def crossover_new(selected_parents, offspring_size):  # offspring size ahould be 2
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        selected_parents1_idx = k % selected_parents.shape[0]
        # Index of the second parent to mate.
        selected_parents2_idx = (k + 1) % selected_parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = selected_parents[selected_parents1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = selected_parents[selected_parents2_idx, crossover_point:]
    return offspring  # this will return 2 new individuals


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    from random import random
    for k in range(0, offspring_crossover.shape[0]):
        for j in range(0, offspring_crossover.shape[1]):
            A = random()
            if A > 0.7:
                if offspring_crossover[k, j] == 0:
                    offspring_crossover[k, j] = 1
                else:
                    offspring_crossover[k, j] = 0
            else:
                continue
    return offspring_crossover

from random import random
num_generations = 20
for generation in range(num_generations):
    t3 = time.time()
    print("Generation : ", generation)
    t0 = time.time()
    f1_scores = calc_fitness(pop)
    print("Time it takes to calculate fitness of one population:", round(time.time() - t0, 3), 's')
    print("F1_scores", f1_scores)
    print("Best result for each generation : ", np.max(f1_scores))

    elite_mem = elitism(f1_scores, pop)
    elite_mem_list = elite_mem.tolist()
    j = 0
    for i in range(0, 10):
        j = j + 2
        if i <= 8:
            # call the tournament selection twice to give the parents that will be part of the mating
            parent1 = tournament_selection(pop, f1_scores)
            parent2 = tournament_selection(pop, f1_scores)

            selected_parents = np.empty((2, pop.shape[1]))
            selected_parents[0] = parent1
            selected_parents[1] = parent2

            # generate a random number as the crossover_probability
            crossover_probability = random()
            if crossover_probability <= 0.9:
                # do crossover
                two_new_indv = crossover_new(selected_parents, (2, 27))
                # also check mutation probability
                mutation_probability = random()
                if mutation_probability <= 0.3:
                    # do mutation
                    mutation_output = mutation(two_new_indv)
                    # send to the new population
                    pop[j - 2:j, :] = two_new_indv  #######################################
                else:
                    # send parent directly to new population
                    pop[j - 2:j, :] = selected_parents  #######################################
            else:
                # go straight to mutation
                # also check mutation probability
                mutation_probability = random()
                if mutation_probability <= 0.3:
                    # do mutation
                    mutation_output = mutation(selected_parents)
                    pop[j - 2:j, :] = mutation_output  #######################################
                else:
                    # send parent directly to new population
                    pop[j - 2:j, :] = selected_parents  #######################################
    else:
        # call the tournament selection twice to give the parents that will be part of the mating
        parent1 = tournament_selection(pop, f1_scores)
        parent2 = tournament_selection(pop, f1_scores)

        selected_parents = np.empty((2, pop.shape[1]))
        selected_parents[0] = parent1
        selected_parents[1] = parent2

        # generate a random number as the crossover_probability
        crossover_probability = random()
        if crossover_probability <= 0.9:
            # do crossover
            two_new_indv = crossover_new(selected_parents, (2, 27))
            # also check mutation probability
            mutation_probability = random()
            if mutation_probability <= 0.3:
                # do mutation
                mutation_output = mutation(two_new_indv)
                # send to the new population
                pop[j - 2, :] = two_new_indv[0]  #######################################
            else:
                # send parent directly to new population
                pop[j - 2, :] = selected_parents[0]  #######################################
        else:
            # go straight to mutation
            # also check mutation probability
            mutation_probability = random()
            if mutation_probability <= 0.3:
                # do mutation
                mutation_output = mutation(selected_parents)
                pop[j - 2, :] = mutation_output[0]  #######################################
            else:
                # send parent directly to new population
                pop[j - 2, :] = selected_parents[0]  #######################################
        pop[j - 1] = elite_mem_list
    print(pop)


# f1_scores=returns_f1_scores(f1_scores)
best_match_idx = np.where(f1_scores== np.max(f1_scores))
print("BMI",best_match_idx)
here= best_match_idx[0][0]
print("Best solution in the final generation: ", pop[here, :])