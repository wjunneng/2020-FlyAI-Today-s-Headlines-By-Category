# -*- coding: utf-8 -*-
import os

from .classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


def load_func(news, category):
    examples = []
    for i, new in enumerate(news):
        examples.append(InputExample(guid=i, text_a=new, text_b=None, label=category[i]))

    return examples


def load_data(news, category, data_type, label_list, max_length, tokenizer, batch_size):
    if data_type == "train":
        examples = load_func(news, category)
    elif data_type == "dev":
        examples = load_func(news, category)
    else:
        raise RuntimeError("should be train or dev")

    features = convert_examples_to_features(examples, label_list, max_length, tokenizer)

    dataloader = convert_features_to_tensors(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len
