# coding=utf-8

import os
import sys

os.chdir(sys.path[0])

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

# 1.BertOrigin
# model_name = 'BertOrigin'
# save_name = 'BertOrigin'

# 2.BertATT [优先选择, 效果好]
model_name = 'BertATT'
save_name = 'BertATT'

# 3.BertCNNPlus
# model_name = 'BertCNNPlus'
# save_name = 'BertCNNPlus'

# 4.BertRCNN
# model_name = 'BertRCNN'
# save_name = 'BertRCNN'

EPOCHS = 1
BATCH = 512
data_dir = os.path.join(os.getcwd(), "data/input")
output_dir = os.path.join(os.getcwd(), "data/output")
cache_dir = os.path.join(os.getcwd(), "data/cache")
log_dir = os.path.join(os.getcwd(), "data/log")
bert_vocab_file = os.path.join(os.getcwd(), 'data/input/model/vocab.txt')
bert_model_dir = os.path.join(os.getcwd(), 'data/input/model')
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

# 是否利用 伪标签 来提升模型的泛化能力
use_pseudo_labeling = True

seed = 42
data_type = 'toutiao'
label_list = ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house',
              'news_car', 'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world',
              'news_agriculture', 'news_game', 'stock', 'news_story']
gpu_ids = '0'
do_lower_case = True
max_seq_length = 28
warmup_proportion = 0.1
learning_rate = 5e-5
gradient_accumulation_steps = 1
print_step = 10
early_stop = 5

if model_name == 'BertCNNPlus':
    # CNN 参数
    filter_num = 250
    filter_sizes = "1 2 3 4 5 6 7 8 9 10 11"

elif model_name == 'BertRCNN':
    # textRnn 参数
    rnn_hidden_size = 300
    num_layers = 2
    bidirectional = True
    dropout = 0.2
