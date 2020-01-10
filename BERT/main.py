# coding=utf-8
import sys
import os

os.chdir(sys.path[0])
import argparse
import random
import shutil
import logging
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from time import strftime, localtime
from pytorch_transformers import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from flyai.utils import remote_helper
from flyai.dataset import Dataset

from utils import Util
from net import Net
import args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

remote_helper.get_remote_date("https://www.flyai.com/m/chinese_base.zip")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Instructor(object):
    """
    特点：使用flyai字典的get all data  | 自己进行划分next batch
    """

    def __init__(self, arguments):
        self.arguments = arguments
        self.dataset = Dataset(epochs=self.arguments.EPOCHS, batch=self.arguments.BATCH, val_batch=self.arguments.BATCH)
        shutil.copyfile(os.path.join(os.getcwd(), 'vocab.txt'),
                        os.path.join(arguments.bert_model_dir, 'vocab.txt'))

    def run(self):
        if not os.path.exists(self.arguments.output_dir):
            os.mkdir(self.arguments.output_dir)

        self.arguments.BATCH = self.arguments.BATCH // self.arguments.gradient_accumulation_steps

        # 设定随机种子
        random.seed(self.arguments.seed)
        np.random.seed(self.arguments.seed)
        torch.manual_seed(self.arguments.seed)
        torch.cuda.manual_seed_all(self.arguments.seed)

        # 数据准备  分词器选择
        tokenizer = BertTokenizer(self.arguments.bert_vocab_file).from_pretrained(self.arguments.bert_model_dir,
                                                                                  do_lower_case=self.arguments.do_lower_case)
        # 获取数据 news/keywords
        news, category, _, _ = self.dataset.get_all_data()
        news = np.asarray([i['news'] for i in news])
        category = np.asarray([i['category'] for i in category])

        index = [i for i in range(len(news))]
        np.random.shuffle(np.asarray(index))
        train_news, dev_news = news[index[0:int(len(index) * 0.9)]], news[index[int(len(index) * 0.9):]]
        train_category, dev_category = category[index[0:int(len(index) * 0.9)]], category[
            index[int(len(index) * 0.9):]]

        logger.info('>>train.shape: {} | dev.shape: {}'.format(train_category.shape, dev_category.shape))
        train_dataloader, train_examples_len = Util.load_data(news=train_news, category=train_category,
                                                              data_type='train',
                                                              label_list=self.arguments.label_list,
                                                              max_length=self.arguments.max_seq_length,
                                                              tokenizer=tokenizer,
                                                              batch_size=self.arguments.BATCH)
        dev_dataloader, dev_examples_len = Util.load_data(news=dev_news, category=dev_category, data_type='dev',
                                                          label_list=self.arguments.label_list,
                                                          max_length=self.arguments.max_seq_length,
                                                          tokenizer=tokenizer,
                                                          batch_size=self.arguments.BATCH)

        num_train_optimization_steps = int(
            train_examples_len / self.arguments.BATCH / self.arguments.gradient_accumulation_steps) * self.arguments.EPOCHS

        # 模型准备
        logger.info("model name is {}".format(self.arguments.model_name))
        if self.arguments.model_name == "BertOrigin":
            model = Net.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                        num_labels=self.arguments.num_labels,
                                        cache_dir=self.arguments.cache_dir)

        elif self.arguments.model_name == 'BertHAN':
            model = Net.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                        num_labels=self.arguments.num_labels,
                                        cache_dir=self.arguments.cache_dir)

        elif self.arguments.model_name == "BertCNN":
            filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
            model = Net.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                        num_labels=self.arguments.num_labels,
                                        n_filters=self.arguments.filter_num,
                                        filter_sizes=filter_sizes,
                                        cache_dir=self.arguments.cache_dir)

        elif self.arguments.model_name == "BertATT":
            model = Net.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                        num_labels=self.arguments.num_labels,
                                        cache_dir=self.arguments.cache_dir)

        elif self.arguments.model_name == "BertRCNN":
            model = Net.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
                                        num_labels=self.arguments.num_labels,
                                        cache_dir=self.arguments.cache_dir,
                                        rnn_hidden_size=self.arguments.rnn_hidden_size,
                                        num_layers=self.arguments.num_layers,
                                        bidirectional=self.arguments.bidirectional,
                                        dropout=self.arguments.dropout)

        elif self.arguments.model_name == "BertCNNPlus":
            filter_sizes = [int(val) for val in self.arguments.filter_sizes.split()]
            model = Net.from_pretrained(pretrained_model_name_or_path=self.arguments.bert_model_dir,
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
        for epoch in range(int(self.arguments.EPOCHS)):
            if early_stop_times >= self.arguments.early_stop * (train_examples_len // self.arguments.BATCH):
                break

            logger.info(f'---------------- Epoch: {epoch + 1:02} ----------')
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

                # 修正
                if self.arguments.gradient_accumulation_steps > 1:
                    loss = loss / self.arguments.gradient_accumulation_steps
                logger.info('\n>>train_loss: {}'.format(loss))
                train_steps += 1
                loss.backward()

                # 用于画图和分析的数据
                epoch_loss += loss.item()
                preds = logits.detach().cpu().numpy()
                outputs = np.argmax(preds, axis=1)
                all_preds = np.append(all_preds, outputs)
                label_ids = label_ids.to('cpu').numpy()
                all_labels = np.append(all_labels, label_ids)

                if (step + 1) % self.arguments.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.arguments.print_step == 0 and global_step != 0:
                        dev_loss, dev_acc, dev_report, dev_auc = Util.evaluate(model, dev_dataloader, criterion, DEVICE,
                                                                               self.arguments.label_list)
                        logger.info('\n>>>dev report: \n{}'.format(dev_report))
                        # 以 acc 取优
                        if dev_acc > best_acc:
                            best_acc = dev_acc
                            # 以 auc 取优
                            if dev_auc > best_auc:
                                best_auc = dev_auc

                            # 保存模型
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), self.arguments.output_model_file)
                            with open(self.arguments.output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())

                            early_stop_times = 0
                        else:
                            early_stop_times += 1

        if os.path.exists(self.arguments.output_config_file) is False:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), self.arguments.output_model_file)
            with open(self.arguments.output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT Baseline')
    # 训练轮数
    parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
    # 批量大小
    parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")

    config = parser.parse_args()

    args.num_labels = len(args.label_list)
    args.EPOCHS = config.EPOCHS
    args.BATCH = config.BATCH

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if os.path.exists(args.log_dir) is False:
        os.mkdir(args.log_dir)
    log_file = '{}-{}-{}.log'.format(args.model_name, args.data_type, strftime('%y%m%d-%H%M', localtime()))
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, log_file)))

    instructor = Instructor(args)
    instructor.run()
