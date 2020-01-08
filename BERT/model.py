# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import torch
import args
from flyai.model.base import Base
from pytorch_transformers import BertConfig
from pytorch_transformers import BertTokenizer

from net import Net
from utils import Util

__import__('net', fromlist=["Net"])

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.args = args.get_args()
        bert_config = BertConfig(self.args.output_config_file)

        if os.path.exists(self.args.output_model_file):
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

        return torch.argmax(logits.view(-1, len(self.args.label_list))).cpu().numpy().tolist()

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(news=data['news'])

            labels.append(predicts)

        return labels
