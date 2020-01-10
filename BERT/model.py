# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import torch
import numpy as np
from flyai.model.base import Base
from pytorch_transformers import BertConfig
from pytorch_transformers import BertTokenizer

import args
from net import Net
from utils import Util
from main import Instructor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.args = args
        self.args.num_labels = len(self.args.label_list)
        self.instructor = Instructor(arguments=self.args)
        self.init()

    def init(self):
        bert_config = BertConfig(self.args.output_config_file)
        if os.path.exists(self.args.output_model_file):
            if self.args.model_name == 'BertCNNPlus':
                bert_config.filter_num = self.args.filter_num
                bert_config.filter_sizes = [int(val) for val in self.args.filter_sizes.split()]
            elif self.args.model_name == 'BertRCNN':
                bert_config.rnn_hidden_size = self.args.rnn_hidden_size
                bert_config.num_layers = self.args.num_layers
                bert_config.bidirectional = self.args.bidirectional
                bert_config.dropout = self.args.dropout
            else:
                pass

            self.model = Net(config=bert_config)
            self.model.load_state_dict(torch.load(self.args.output_model_file))
            self.model.to(DEVICE)

        self.tokenizer = BertTokenizer(self.args.bert_vocab_file).from_pretrained(self.args.bert_model_dir,
                                                                                  do_lower_case=self.args.do_lower_case)

    def predict(self, **data):
        x_data = self.data.predict_data(**data)
        predict_dataloader, predict_examples_len = Util.load_data(news=x_data,
                                                                  category=None,
                                                                  data_type='predict',
                                                                  label_list=self.args.label_list,
                                                                  max_length=self.args.max_seq_length,
                                                                  tokenizer=self.tokenizer,
                                                                  batch_size=1)

        for step, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            _, input_ids, input_mask, segment_ids = batch

            input_ids = input_ids.to(DEVICE)
            input_mask = input_mask.to(DEVICE)
            segment_ids = segment_ids.to(DEVICE)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

        return self.data.to_categorys(torch.argmax(logits.view(-1, len(self.args.label_list))).cpu().numpy().tolist())

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(news=data['news'])

            labels.append(predicts)

        if self.args.use_pseudo_labeling:
            self.pseudo_labeling(datas=datas, labels=labels)

        return labels

    def pseudo_labeling(self, datas, labels):
        news = np.asarray([i['news'] for i in datas])
        category = np.asarray(labels)
        train_news, train_category, dev_news, dev_category = self.instructor.generate()

        train_news = np.concatenate([train_news, news])
        train_category = np.concatenate([train_category, category])

        np.random.shuffle(train_news)
        np.random.shuffle(train_category)

        self.instructor.train(Net=self.model, train_category=train_category, dev_category=dev_category,
                              train_news=train_news,
                              dev_news=dev_news, tokenizer=self.tokenizer)
        self.init()

        self.args.use_pseudo_labeling = False
        self.predict_all(datas=[{'news': i} for i in news])
