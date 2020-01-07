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
from tqdm import tqdm
import torch.nn as nn
from time import strftime, localtime
from pytorch_transformers import BertTokenizer
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from BERT.Utils.utils import evaluate
from BERT.Utils.load_datatsets import load_data

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
        parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
        parser.add_argument("-b", "--BATCH", default=10, type=int, help="batch size")
        self.args = parser.parse_args()
        self.arguments = arguments
        self.dataset = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)
        shutil.copyfile(os.path.join(os.getcwd(), 'vocab.txt'),
                        os.path.join(os.getcwd(), arguments.pretrained_bert_name, 'vocab.txt'))

    def run(self):
        if not os.path.exists(self.arguments.output_dir):
            os.mkdir(self.arguments.output_dir)

        # Bert 模型输出文件
        output_model_file = os.path.join(self.arguments.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.arguments.output_dir, CONFIG_NAME)

        self.arguments.train_batch_size = self.arguments.train_batch_size // self.arguments.gradient_accumulation_steps

        # 设定随机种子
        random.seed(self.arguments.seed)
        np.random.seed(self.arguments.seed)
        torch.manual_seed(self.arguments.seed)
        torch.cuda.manual_seed_all(self.arguments.seed)

        # 数据准备  分词器选择
        tokenizer = BertTokenizer(self.arguments.bert_vocab_file).from_pretrained(self.arguments.bert_model_dir,
                                                                                  do_lower_case=self.arguments.do_lower_case)

        news, category, _, _ = self.dataset.get_all_data()
        news = np.asarray([i['news'] for i in news])
        category = np.asarray([i['category'] for i in category])

        index = [i for i in range(len(news))]
        np.random.shuffle(np.asarray(index))
        train_news, dev_news = news[index[0:int(len(index) * 0.9)]], news[index[int(len(index) * 0.9):]]
        train_category, dev_category = category[index[0:int(len(index) * 0.9)]], category[
            index[int(len(index) * 0.9):]]

        train_dataloader, train_examples_len = load_data(news=train_news, category=train_category,
                                                         data_type='train',
                                                         label_list=self.arguments.label_list,
                                                         max_length=self.arguments.max_length,
                                                         tokenizer=tokenizer,
                                                         batch_size=self.args.BATCH)
        dev_dataloader, dev_examples_len = load_data(news=dev_news, category=dev_category, data_type='dev',
                                                     label_list=self.arguments.label_list,
                                                     max_length=self.arguments.max_length,
                                                     tokenizer=tokenizer,
                                                     batch_size=self.args.BATCH)

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
                                            n_filters=self.arguments.filter_num,
                                            filter_sizes=filter_sizes,
                                            cache_dir=self.arguments.cache_dir)
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
                                                num_labels=self.arguments.num_labels,
                                                cache_dir=self.arguments.cache_dir,
                                                n_filters=self.arguments.filter_num,
                                                filter_sizes=filter_sizes)

        model.to(DEVICE)

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.arguments.learning_rate,
                          correct_bias=False)
        # PyTorch scheduler
        scheduler = WarmupLinearSchedule(optimizer=optimizer, warmup_steps=self.arguments.warmup_proportion,
                                         t_total=num_train_optimization_steps)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(DEVICE)

        best_auc, best_acc, global_step, early_stop_times = 0, 0, 0, 0
        for epoch in range(int(self.args.EPOCHS)):
            if early_stop_times >= self.arguments.early_stop * (train_examples_len // self.args.BATCH):
                break

            print(f'---------------- Epoch: {epoch + 1:02} ----------')
            epoch_loss, train_steps = 0, 0

            all_preds = np.array([], dtype=int)
            all_labels = np.array([], dtype=int)

            scheduler.step()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()

                batch = tuple(t.to(DEVICE) for t in batch)
                _, input_ids, input_mask, segment_ids, label_ids = batch

                logits = model(input_ids, segment_ids, input_mask, labels=None)
                loss = criterion(logits.view(-1, self.arguments.num_labels), label_ids.view(-1))
                train_steps += 1
                loss.backward()

                # 用于画图和分析的数据
                epoch_loss += loss.item()
                preds = logits.detach().cpu().numpy()
                outputs = np.argmax(preds, axis=1)
                all_preds = np.append(all_preds, outputs)
                label_ids = label_ids.to('cpu').numpy()
                all_labels = np.append(all_labels, label_ids)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % self.arguments.print_step == 0 and global_step != 0:
                    dev_loss, dev_acc, dev_report, dev_auc = evaluate(model, dev_dataloader, criterion, DEVICE,
                                                                      self.arguments.label_list)

                    logger.info(
                        'dev_loss:{}, dev_acc:{}, dev_report:{}, dev_auc:{}'.format(dev_loss, dev_acc, dev_report,
                                                                                    dev_auc))
                    # 以 acc 取优
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        # 以 auc 取优
                        if dev_auc > best_auc:
                            best_auc = dev_auc

                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())
                        early_stop_times = 0
                    else:
                        early_stop_times += 1


if __name__ == '__main__':
    model_name = 'BertCNN'
    args = None
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
    bert_model_dir = os.path.join(os.getcwd(), 'data/input/model')
    args = args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir)

    args.pretrained_bert_name = 'data/input/model'
    args.seed = 42
    args.max_length = 64
    args.print_step = 5
    args.early_stop = 5
    args.gradient_accumulation_steps = 1
    args.dataset = 'toutiao'
    args.log_path = 'data/log'
    args.label_list = ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house', 'news_car',
                       'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world', 'news_agriculture',
                       'news_game', 'stock', 'news_story']
    args.num_labels = len(args.label_list)

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
    instructor.run()

# 加载模型
# bert_config = BertConfig(output_config_file)
#
# if self.arguments.model_name == "BertOrigin":
#     from BertOrigin.BertOrigin import BertOrigin
#     model = BertOrigin(config=bert_config)
# elif self.arguments.model_name == "BertCNN":
#     from BertCNN.BertCNN import BertCNN
#     filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
#     model = BertCNN(config=bert_config, n_filters=self.arguments.filter_num, filter_sizes=filter_sizes)
# elif self.arguments.model_name == "BertATT":
#     from BertATT.BertATT import BertATT
#     model = BertATT(config=bert_config)
# elif self.arguments.model_name == "BertRCNN":
#     from BertRCNN.BertRCNN import BertRCNN
#     model = BertRCNN(config=bert_config, rnn_hidden_size=self.arguments.hidden_size,
#                      num_layers=self.arguments.num_layers,
#                      bidirectional=self.arguments.bidirectional, dropout=self.arguments.dropout)
# elif self.arguments.model_name == "BertCNNPlus":
#     from BertCNNPlus.BertCNNPlus import BertCNNPlus
#     filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
#     model = BertCNNPlus(config=bert_config, n_filters=self.arguments.filter_num, filter_sizes=filter_sizes)
# 损失函数准备
# model.load_state_dict(torch.load(output_model_file))
# model.to(device)
# criterion = nn.CrossEntropyLoss()
# criterion = criterion.to(device)
# # test the model
# test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds = evaluate_save(
#     model, test_dataloader, criterion, device, self.arguments.label_list)
# print("-------------- Test -------------")
# print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc * 100: .3f} % | AUC:{test_auc}')
#
# for label in self.arguments.label_list:
#     print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
#         label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
# print_list = ['macro avg', 'weighted avg']
#
# for label in print_list:
#     print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
#         label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
