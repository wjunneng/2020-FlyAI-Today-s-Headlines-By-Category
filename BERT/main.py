# coding=utf-8
import sys
import os

os.chdir(sys.path[0])
import random
import numpy as np
import argparse
import shutil
import logging
import torch
import torch.nn as nn
from time import strftime, localtime
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from Utils.utils import get_device
from Utils.load_datatsets import load_data
from Utils.train_evalute import train, evaluate_save

from flyai.utils import remote_helper
from flyai.dataset import Dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# remote_helper.get_remote_date("https://www.flyai.com/m/chinese_base.zip")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        shutil.copyfile(os.path.join(os.getcwd(), 'vocab.txt'),
                        os.path.join(os.getcwd(), arguments.pretrained_bert_name, 'vocab.txt'))

    def run(self, model_times, label_list):
        if not os.path.exists(self.arguments.output_dir + model_times):
            os.makedirs(self.arguments.output_dir + model_times)

        if not os.path.exists(self.arguments.cache_dir + model_times):
            os.makedirs(self.arguments.cache_dir + model_times)

        # Bert 模型输出文件
        output_model_file = os.path.join(self.arguments.output_dir, model_times, WEIGHTS_NAME)
        output_config_file = os.path.join(self.arguments.output_dir, model_times, CONFIG_NAME)

        # 设备准备
        gpu_ids = [int(device_id) for device_id in self.arguments.gpu_ids.split()]
        device, n_gpu = get_device(gpu_ids[0])
        if n_gpu > 1:
            n_gpu = len(gpu_ids)

        self.arguments.train_batch_size = self.arguments.train_batch_size // self.arguments.gradient_accumulation_steps

        # 设定随机种子
        random.seed(self.arguments.seed)
        np.random.seed(self.arguments.seed)
        torch.manual_seed(self.arguments.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.arguments.seed)

        # 数据准备
        tokenizer = BertTokenizer(self.arguments.bert_vocab_file).from_pretrained(self.arguments.bert_model_dir,
                                                                                  do_lower_case=self.arguments.do_lower_case)  # 分词器选择

        num_labels = len(label_list)

        # Train and dev
        if self.arguments.do_train:

            train_dataloader, train_examples_len = load_data(
                self.arguments.data_dir, tokenizer, self.arguments.max_seq_length, self.arguments.train_batch_size,
                "train", label_list)
            dev_dataloader, _ = load_data(
                self.arguments.data_dir, tokenizer, self.arguments.max_seq_length, self.arguments.dev_batch_size, "dev",
                label_list)

            num_train_optimization_steps = int(
                train_examples_len / self.arguments.train_batch_size / self.arguments.gradient_accumulation_steps) * self.arguments.num_train_epochs

            # 模型准备
            print("model name is {}".format(self.arguments.model_name))
            if self.arguments.model_name == "BertOrigin":
                from BertOrigin.BertOrigin import BertOrigin
                model = BertOrigin.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                                   num_labels=self.arguments.num_labels,
                                                   cache_dir=self.arguments.cache_dir)
            elif self.arguments.model_name == "BertCNN":
                from BertCNN.BertCNN import BertCNN
                filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
                model = BertCNN.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                                num_labels=self.arguments.num_labels,
                                                cache_dir=self.arguments.cache_dir,
                                                n_filters=self.arguments.filter_num,
                                                filter_sizes=filter_sizes)
            elif self.arguments.model_name == "BertATT":
                from BertATT.BertATT import BertATT
                model = BertATT.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                                num_labels=self.arguments.num_labels,
                                                cache_dir=self.arguments.cache_dir)

            elif self.arguments.model_name == "BertRCNN":
                from BertRCNN.BertRCNN import BertRCNN
                model = BertRCNN.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                                 num_labels=self.arguments.num_labels,
                                                 cache_dir=self.arguments.cache_dir,
                                                 rnn_hidden_size=self.arguments.hidden_size,
                                                 num_layers=self.arguments.num_layers,
                                                 bidirectional=self.arguments.bidirectional,
                                                 dropout=self.arguments.dropout)

            elif self.arguments.model_name == "BertCNNPlus":
                from BertCNNPlus.BertCNNPlus import BertCNNPlus
                filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
                model = BertCNNPlus.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                                    num_labels=num_labels,
                                                    cache_dir=self.arguments.cache_dir,
                                                    n_filters=self.arguments.filter_num,
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

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.arguments.learning_rate,
                              correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.arguments.warmup_proportion,
                                             t_total=num_train_optimization_steps)  # PyTorch scheduler

            """ 损失函数准备 """
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            train(self.arguments.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer, scheduler,
                  criterion, self.arguments.gradient_accumulation_steps, device, label_list, output_model_file,
                  output_config_file,
                  self.arguments.log_dir, self.arguments.print_step, self.arguments.early_stop)

        """ Test """

        # test 数据
        test_dataloader, _ = load_data(
            self.arguments.data_dir, tokenizer, self.arguments.max_seq_length, self.arguments.test_batch_size, "test",
            label_list)

        # 加载模型
        bert_config = BertConfig(output_config_file)

        if self.arguments.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin(config=bert_config)
        elif self.arguments.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
            model = BertCNN(config=bert_config, n_filters=self.arguments.filter_num, filter_sizes=filter_sizes)
        elif self.arguments.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT(config=bert_config)
        elif self.arguments.model_name == "BertRCNN":
            from BertRCNN.BertRCNN import BertRCNN
            model = BertRCNN(config=bert_config, rnn_hidden_size=self.arguments.hidden_size,
                             num_layers=self.arguments.num_layers,
                             bidirectional=self.arguments.bidirectional, dropout=self.arguments.dropout)
        elif self.arguments.model_name == "BertCNNPlus":
            from BertCNNPlus.BertCNNPlus import BertCNNPlus
            filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
            model = BertCNNPlus(config=bert_config, n_filters=self.arguments.filter_num, filter_sizes=filter_sizes)

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


if __name__ == '__main__':
    model_name = 'BertOrigin'

    if model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == 'BertCNNPlus':
        from BertCNNPlus import args

    elif model_name == 'BertHAN':
        from BertHAN import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertOrigin":
        from BertOrigin import args

    data_dir = os.path.join(os.getcwd(), "data/input")
    output_dir = os.path.join(os.getcwd(), "data/output")
    cache_dir = os.path.join(os.getcwd(), "data/cache")
    log_dir = os.path.join(os.getcwd(), "data/log")
    bert_vocab_file = os.path.join(os.getcwd(), 'data/input/model/vocab.txt')
    bert_model_dir = os.path.join(os.getcwd(), 'data/input/model/pytorch_model.bin')
    config = args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir)

    args.pretrained_bert_name = 'data/input/model'
    args.seed = 42
    args.dataset = 'toutiao'
    args.log_path = 'data/log'
    args.label_list = ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house', 'news_car',
                       'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world', 'news_agriculture',
                       'news_game', 'stock', 'news_story']

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_path = os.path.join(os.getcwd(), args.log_path)
    if os.path.exists(log_path) is False:
        os.mkdir(log_path)
    log_file = '{}-{}-{}.log'.format(model_name, args.dataset, strftime('%y%m%d-%H%M', localtime()))
    logger.addHandler(logging.FileHandler(os.path.join(log_path, log_file)))

    instructor = Instructor(args)
    instructor.run(model_times=args.save_name, label_list=args.label_list)
