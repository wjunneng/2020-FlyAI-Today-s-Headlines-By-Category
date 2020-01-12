# coding=utf-8
from sklearn import metrics
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from args import PAD


class InputExample(object):
    """单句子分类的 Example 类"""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, idx, input_ids, input_mask, segment_ids, label_id):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class Util(object):
    def __init__(self):
        pass

    @staticmethod
    def get_device(gpu_id):
        device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        if torch.cuda.is_available():
            print("device is cuda, # cuda is: ", n_gpu)
        else:
            print("device is cpu, not recommend")
        return device, n_gpu

    @staticmethod
    def set_seed(args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @staticmethod
    def classifiction_metric(preds, labels, label_list):
        """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

        acc = metrics.accuracy_score(labels, preds)

        labels_list_indices = [i for i in range(len(label_list))]

        report = metrics.classification_report(labels, preds,
                                               labels=labels_list_indices,
                                               target_names=label_list,
                                               digits=5,
                                               output_dict=False)

        if len(label_list) > 2:
            auc = 0.5
        else:
            auc = metrics.roc_auc_score(labels, preds)
        return acc, report, auc

    @staticmethod
    def load_func(news, category):
        examples = []
        for i, new in enumerate(news):
            if category is None:
                examples.append(InputExample(guid=i, text_a=new, text_b=None, label=None))
            else:
                examples.append(InputExample(guid=i, text_a=new, text_b=None, label=category[i]))

        return examples

    @staticmethod
    def load_data(news, category, data_type, label_list, max_length, tokenizer, batch_size):
        if data_type == "train":
            examples = Util.load_func(news, category)
        elif data_type == "dev":
            examples = Util.load_func(news, category)
        elif data_type == 'predict':
            examples = Util.load_func(news, category)
        else:
            raise RuntimeError("should be train or dev")

        features = ClassifierUtil.convert_examples_to_features(examples, label_list, max_length, tokenizer)

        dataloader = ClassifierUtil.convert_features_to_tensors(features, batch_size, data_type)

        examples_len = len(examples)

        return dataloader, examples_len

    @staticmethod
    def evaluate(model, dataloader, criterion, device, label_list, args):
        model.eval()
        if args.use_label_smoothing:
            criterion.eval()

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)
        epoch_loss = 0

        for _, input_ids, input_mask, segment_ids, label_ids in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss = criterion(inputs=logits, labels=label_ids, normalization=1.0, reduce=True)

            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)

            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            epoch_loss += loss.mean().item()

        acc, report, auc = Util.classifiction_metric(all_preds, all_labels, label_list)

        return epoch_loss / len(dataloader), acc, report, auc

    @staticmethod
    def evaluate_save(model, dataloader, criterion, device, label_list, args):
        model.eval()
        if args.use_label_smoothing:
            criterion.eval()

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)
        all_idxs = np.array([], dtype=int)

        epoch_loss = 0

        for idxs, input_ids, input_mask, segment_ids, label_ids in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1), normalization=1.0, reduce=False)

            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)

            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            idxs = idxs.detach().cpu().numpy()
            all_idxs = np.append(all_idxs, idxs)

            epoch_loss += loss.mean().item()

        acc, report, auc = Util.classifiction_metric(all_preds, all_labels, label_list)

        return epoch_loss / len(dataloader), acc, report, auc, all_idxs, all_labels, all_preds


class ClassifierUtil(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s.
        Args:
            examples: InputExample, 表示样本集
            label_list: 标签列表
            max_seq_length: 句子最大长度
            tokenizer： 分词器
        Returns:
            features: InputFeatures, 表示样本转化后信息
        """

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)  # 分词

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)  # 分词
                # “-3” 是因为句子中有[CLS], [SEP], [SEP] 三个标识，可参见论文
                # [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                ClassifierUtil._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # "- 2" 是因为句子中有[CLS], [SEP] 两个标识，可参见论文
                # [CLS] the dog is hairy . [SEP]
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # [CLS] 可以视作是保存句子全局向量信息
            # [SEP] 用于区分句子，使得模型能够更好的把握句子信息

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)  # 句子标识，0表示是第一个句子，1表示是第二个句子，参见论文

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将词转化为对应词表中的id

            # input_mask: 1 表示真正的 tokens， 0 表示是 padding tokens
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = None
            try:
                if example.label is not None:
                    label_id = label_map[example.label]
            except:
                print('label map error: the label does not exist in label_list:', example.label)
                exit()
            idx = int(example.guid)

            features.append(
                InputFeatures(idx=idx,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
        return features

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """ 截断句子a和句子b，使得二者之和不超过 max_length """

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def convert_features_to_tensors(features, batch_size, data_type):
        """ 将 features 转化为 tensor，并塞入迭代器
        Args:
            features: InputFeatures, 样本 features 信息
            batch_size: batch 大小
        Returns:
            dataloader: 以 batch_size 为基础的迭代器
        """
        all_idx_ids = torch.tensor([f.idx for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        try:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            data = TensorDataset(all_idx_ids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        except:
            data = TensorDataset(all_idx_ids, all_input_ids, all_input_mask, all_segment_ids)

        sampler = RandomSampler(data)
        if data_type == "test":
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        else:
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)

        return dataloader


class Criterion(nn.Module):
    """ Class for managing loss computation.

    """

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Compute the loss. Subclass must override this method.

        Args:
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        Returns:
            A non-reduced FloatTensor with shape (batch, )
        """
        raise NotImplementedError

    def forward(self, inputs, labels, normalization=1.0, reduce=True, **kwargs):
        """
        Compute loss given inputs and labels.

        Args:
            inputs: Input tensor of the criterion.
            labels: Label tensor of the criterion.
            reduce: Boolean value indicate whether the criterion should reduce the loss along the batch. If false,
                the criterion return a FloatTensor with shape (batch, ), otherwise a scalar.
            normalization: Normalization factor of the loss. Should be a float scalar or a FloatTensor with shape
                (batch, )
        """
        loss = self._compute_loss(inputs, labels, **kwargs).div(normalization)  # (batch, )

        if reduce:
            loss = loss.sum()

        return loss


class NMTCriterion(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self, padding_idx=PAD, label_smoothing=0.0):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:

            self.criterion = nn.KLDivLoss(size_average=False, reduce=False)

        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=padding_idx, reduce=False)

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(1))

    def _compute_loss(self, inputs, labels, **kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)
        # [batch_size * seq_len, d_words]
        scores = self._bottle(inputs)

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()
            # mask of PAD
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            # Do label smoothing
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            # [N, M]
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        return loss
