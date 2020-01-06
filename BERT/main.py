# coding=utf-8
import random
import numpy as np
import argparse
import sys
import os
import shutil
import logging
import torch
import torch.nn as nn

from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

import args as arguments
from Utils.utils import get_device
from Utils.load_datatsets import load_data
from train_evalute import train, evaluate_save

from flyai.utils import remote_helper
from flyai.dataset import Dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

remote_helper.get_remote_date("https://www.flyai.com/m/chinese_base.zip")
shutil.copyfile(os.path.join(os.getcwd(), 'vocab.txt'),
                os.path.join(os.getcwd(), arguments.pretrained_bert_name, 'vocab.txt'))


class Instructor(object):
    """
    特点：使用flyai字典的get all data  | 自己进行划分next batch
    """

    def __init__(self, arguments):
        # 项目的超参
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
        self.args = parser.parse_args()
        self.arguments = arguments
        self.dataset = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)

    def run(self, config, model_times, label_list):
        if not os.path.exists(config.output_dir + model_times):
            os.makedirs(config.output_dir + model_times)

        if not os.path.exists(config.cache_dir + model_times):
            os.makedirs(config.cache_dir + model_times)

        # Bert 模型输出文件
        output_model_file = os.path.join(config.output_dir, model_times, WEIGHTS_NAME)
        output_config_file = os.path.join(config.output_dir, model_times, CONFIG_NAME)

        # 设备准备
        gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
        device, n_gpu = get_device(gpu_ids[0])
        if n_gpu > 1:
            n_gpu = len(gpu_ids)

        config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

        # 设定随机种子
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(config.seed)

        # 数据准备
        tokenizer = BertTokenizer(config.bert_vocab_file).from_pretrained(config.bert_model_dir,
                                                                          do_lower_case=config.do_lower_case)  # 分词器选择

        num_labels = len(label_list)

        # Train and dev
        if config.do_train:

            train_dataloader, train_examples_len = load_data(
                config.data_dir, tokenizer, config.max_seq_length, config.train_batch_size, "train", label_list)
            dev_dataloader, _ = load_data(
                config.data_dir, tokenizer, config.max_seq_length, config.dev_batch_size, "dev", label_list)

            num_train_optimization_steps = int(
                train_examples_len / config.train_batch_size / config.gradient_accumulation_steps) * config.num_train_epochs

            # 模型准备
            print("model name is {}".format(config.model_name))
            if config.model_name == "BertOrigin":
                from BertOrigin.BertOrigin import BertOrigin
                model = BertOrigin.from_pretrained(pretrained_model_name_or_path=config.bert_model_dir,
                                                   num_labels=config.num_labels,
                                                   cache_dir=config.cache_dir)
            elif config.model_name == "BertCNN":
                from BertCNN.BertCNN import BertCNN
                filter_sizes = [int(val) for val in config.filter_sizes.split()]
                model = BertCNN.from_pretrained(pretrained_model_name_or_path=config.bert_model_dir,
                                                num_labels=config.num_labels,
                                                cache_dir=config.cache_dir,
                                                n_filters=config.filter_num,
                                                filter_sizes=filter_sizes)
            elif config.model_name == "BertATT":
                from BertATT.BertATT import BertATT
                model = BertATT.from_pretrained(pretrained_model_name_or_path=config.bert_model_dir,
                                                num_labels=config.num_labels,
                                                cache_dir=config.cache_dir)

            elif config.model_name == "BertRCNN":
                from BertRCNN.BertRCNN import BertRCNN
                model = BertRCNN.from_pretrained(pretrained_model_name_or_path=config.bert_model_dir,
                                                 num_labels=config.num_labels,
                                                 cache_dir=config.cache_dir,
                                                 rnn_hidden_size=config.hidden_size,
                                                 num_layers=config.num_layers,
                                                 bidirectional=config.bidirectional,
                                                 dropout=config.dropout)

            elif config.model_name == "BertCNNPlus":
                from BertCNNPlus.BertCNNPlus import BertCNNPlus
                filter_sizes = [int(val) for val in config.filter_sizes.split()]
                model = BertCNNPlus.from_pretrained(pretrained_model_name_or_path=config.bert_model_dir,
                                                    num_labels=num_labels,
                                                    cache_dir=config.cache_dir,
                                                    n_filters=config.filter_num,
                                                    filter_sizes=filter_sizes)

            model.to(device)

            if n_gpu > 1:
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)

            """ 优化器准备 """
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate,
                              correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_proportion,
                                             t_total=num_train_optimization_steps)  # PyTorch scheduler

            """ 损失函数准备 """
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            train(config.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer, scheduler,
                  criterion, config.gradient_accumulation_steps, device, label_list, output_model_file,
                  output_config_file,
                  config.log_dir, config.print_step, config.early_stop)

        """ Test """

        # test 数据
        test_dataloader, _ = load_data(
            config.data_dir, tokenizer, config.max_seq_length, config.test_batch_size, "test", label_list)

        # 加载模型
        bert_config = BertConfig(output_config_file)

        if config.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin(config=bert_config)
        elif config.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in config.filter_sizes.split()]
            model = BertCNN(config=bert_config, n_filters=config.filter_num, filter_sizes=filter_sizes)
        elif config.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT(config=bert_config)
        elif config.model_name == "BertRCNN":
            from BertRCNN.BertRCNN import BertRCNN
            model = BertRCNN(config=bert_config, rnn_hidden_size=config.hidden_size, num_layers=config.num_layers,
                             bidirectional=config.bidirectional, dropout=config.dropout)
        elif config.model_name == "BertCNNPlus":
            from BertCNNPlus.BertCNNPlus import BertCNNPlus
            filter_sizes = [int(val) for val in config.filter_sizes.split()]
            model = BertCNNPlus(config=bert_config, n_filters=config.filter_num, filter_sizes=filter_sizes)

        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        # 损失函数准备
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # test the model
        test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds = evaluate_save(
            model, test_dataloader, criterion, device, label_list)
        print("-------------- Test -------------")
        print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc * 100: .3f} % | AUC:{test_auc}')

        for label in label_list:
            print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
        print_list = ['macro avg', 'weighted avg']

        for label in print_list:
            print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
                label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
