# coding=utf-8

import os
import sys

os.chdir(sys.path[0])
import argparse
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

data_dir = os.path.join(os.getcwd(), "data/input")
output_dir = os.path.join(os.getcwd(), "data/output")
cache_dir = os.path.join(os.getcwd(), "data/cache")
log_dir = os.path.join(os.getcwd(), "data/log")
bert_vocab_file = os.path.join(os.getcwd(), 'data/input/model/vocab.txt')
bert_model_dir = os.path.join(os.getcwd(), 'data/input/model')
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)


def get_args():
    parser = argparse.ArgumentParser(description='BERT Baseline')

    parser.add_argument("--model_name",
                        default="BertOrigin",
                        type=str,
                        help="the name of model")

    parser.add_argument("--save_name",
                        default="BertOrigin",
                        type=str,
                        help="the name file of model")

    # 文件路径：数据目录， 缓存目录
    parser.add_argument("--data_dir",
                        default=data_dir,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=output_dir,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--output_model_file",
                        default=output_model_file,
                        type=str,
                        help="The output model save file")

    parser.add_argument("--output_config_file",
                        default=output_config_file,
                        type=str,
                        help="The output model config save file")

    parser.add_argument("--cache_dir",
                        default=cache_dir,
                        type=str,
                        help="缓存目录，主要用于模型缓存")

    parser.add_argument("--log_dir",
                        default=log_dir,
                        type=str,
                        help="日志目录，主要用于 tensorboard 分析")

    parser.add_argument("--bert_vocab_file",
                        default=bert_vocab_file,
                        type=str)
    parser.add_argument("--bert_model_dir",
                        default=bert_model_dir,
                        type=str)

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="随机种子 for initialization")

    parser.add_argument('--data_type',
                        type=str,
                        default='toutiao',
                        help="数据类型")

    parser.add_argument("--label_list",
                        type=list,
                        default=['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house',
                                 'news_car', 'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world',
                                 'news_agriculture', 'news_game', 'stock', 'news_story'],
                        help='标签列表'
                        )

    parser.add_argument("--gpu_ids",
                        type=str,
                        default="0",
                        help="gpu 的设备id")

    # 文本预处理参数
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_seq_length",
                        default=24,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # 训练参数
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # optimizer 参数
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam 的 学习率")

    # 梯度累积
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--print_step',
                        type=int,
                        default=5,
                        help="多少步进行模型保存以及日志信息写入")

    parser.add_argument("--early_stop",
                        type=int,
                        default=10,
                        help="提前终止，多少次dev loss 连续增大，就不再训练")
    config = parser.parse_args()

    return config
