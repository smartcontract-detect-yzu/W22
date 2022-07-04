import json
import os

import numpy
import torch
from gensim.models import FastText
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class PscDataset(Dataset):

    def __init__(self):

        self.x = None  # 样本数 * 样本最大长度 * 150
        self.y = None  # 样本数 * 1
        self.array = None
        self.max_len = 0
        self.total_cnt = 0
        self.length_limit = 1500
        self.length_limit_cnt = 0
        self.total_length = 0

        """可以在初始化函数当中对数据进行一些操作，比如读取、归一化等"""
        self.data_list = []  # 读取 txt 数据
        self.data_np = None  # 读取 txt 数据
        self.data = []
        self.data_p = []
        self.data_np = []
        self.ponzi_dataset_path = "examples/psc_dataset/p"
        self.no_ponzi_dataset_path = "examples/psc_dataset/np"
        self.contract_model = "examples/psc_dataset/Model/FastText/fasttext_model"
        self.p_pt_save = "examples/psc_dataset/pt_file/p_t/"
        self.np_pt_save = "examples/psc_dataset/pt_file/np_t/"
        self.dataset_pt_save_p = "examples/psc_dataset/pt_file/dataset_p.pt"
        self.dataset_pt_save_np = "examples/psc_dataset/pt_file/dataset_np.pt"
        self.FASTTEXT_MODEL = None
        self.dup_filter = None

        self.filter_init()      # 过滤器
        self.get_max_length()   # 合约最大长度
        self.create_dataset()   # 数据集构建

    def filter_init(self):
        with open("dup_file_nams.json", "r") as f:
            self.dup_filter = json.load(f)

    def create_dataset_v2(self):

        g = os.walk(self.p_pt_save)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.endswith(".pt"):  # 目前仅限solidity文件
                    file_name = os.path.join(path, file_name)
                    data = torch.load(file_name)
                    self.data_list.append(data.astype(float))

        g = os.walk(self.np_pt_save)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.endswith(".pt"):  # 目前仅限solidity文件
                    file_name = os.path.join(path, file_name)
                    data = torch.load(file_name)
                    self.data_list.append(data)

        # b = []
        # for i in range(len(self.data_list)):
        #     b.append(np.array(self.data_list[i]))
        # c = np.array(b)

        self.data_np = np.array(self.data_list)
        self.data = torch.from_numpy(self.data_np)

    def create_dataset_init(self):

        if not os.path.exists(self.dataset_pt_save_p):
            print("start to load ponzi dataset")
            g = os.walk(self.p_pt_save)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".pt"):  # 目前仅限solidity文件
                        file_name = os.path.join(path, file_name)
                        data = torch.load(file_name)
                        self.data_p.append(data)

            print("start to save p dataset")
            torch.save(self.data_p, self.dataset_pt_save_p)

        if not os.path.exists(self.dataset_pt_save_np):
            print("start to load no ponzi dataset")
            g = os.walk(self.np_pt_save)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".pt"):  # 目前仅限solidity文件
                        file_name = os.path.join(path, file_name)
                        data = torch.load(file_name)
                        self.data_np.append(data)

            print("start to save np dataset")
            torch.save(self.data_np, self.dataset_pt_save_np)

    def get_max_length(self):

        g = os.walk(self.ponzi_dataset_path)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.endswith(".psc"):  # 目前仅限solidity文件
                    self.total_cnt += 1
                    file_prefix = file_name.split(".psc")[0]
                    file_name = os.path.join(path, file_name)
                    self._get_max_length(file_name, file_prefix, 1)

        g = os.walk(self.no_ponzi_dataset_path)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.endswith(".psc"):  # 目前仅限solidity文件
                    self.total_cnt += 1
                    file_prefix = file_name.split(".psc")[0]
                    file_name = os.path.join(path, file_name)
                    self._get_max_length(file_name, file_prefix, 0)

        print("最大文件长度：{} 文件个数：{} 平均长度：{}".format(self.max_len, self.total_cnt, (self.total_length/self.total_cnt)))
        print("length_limit {} cnt：{} ".format(self.length_limit, self.length_limit_cnt))

    def _get_max_length(self, psc_file, file_prefix, label):

        with open(psc_file, "r") as f:
            sequence_content = f.readline()

        tokens = list(word_tokenize(sequence_content))
        self.total_length += len(tokens)

        if len(tokens) > self.length_limit:
            self.length_limit_cnt += 1

        elif len(tokens) > self.max_len:
            self.max_len = len(tokens)

    def create_dataset(self):

        self.FASTTEXT_MODEL = FastText.load(self.contract_model)
        with tqdm(total=self.total_cnt) as pbar:
            g = os.walk(self.ponzi_dataset_path)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".psc"):  # 目前仅限solidity文件
                        file_name = os.path.join(path, file_name)
                        self._get_dataset_sample(file_name, 1)
                        pbar.update(1)

            g = os.walk(self.no_ponzi_dataset_path)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".psc"):  # 目前仅限solidity文件
                        file_name = os.path.join(path, file_name)
                        self._get_dataset_sample(file_name, 0)
                        pbar.update(1)

        # print("保存数据")
        # torch.save(self.x, "x.pt")
        # torch.save(self.y, "y.pt")

    def _get_dataset_sample(self, psc_file, label):

        with open(psc_file, "r") as f:
            sequence_content = f.readline()

        tokens = list(word_tokenize(sequence_content))
        if len(tokens) > self.length_limit:
            return

        array = None
        for token in tokens:
            token_data = self.FASTTEXT_MODEL.wv.__getitem__(token)
            token_data = np.array(token_data, dtype=numpy.float)
            token_data = np.array([token_data], dtype=numpy.float)
            if array is None:
                array = token_data
            else:
                array = np.concatenate((array, token_data), axis=0)

        for i in range(self.max_len - len(tokens)):
            token_data = np.zeros(shape=(1, 150), dtype=numpy.float)
            array = np.concatenate((array, token_data), axis=0)

        array = np.array([array], dtype=numpy.float)
        if self.x is None:
            self.x = array
        else:
            self.x = np.concatenate((self.x, array), axis=0)

        if label == 0:
            label = [0., 1.]
            label = np.array([label], dtype=numpy.float)
        else:
            label = [1., 0.]
            label = np.array([label], dtype=numpy.float)

        if self.y is None:
            self.y = np.array(label, dtype=numpy.float)
        else:
            self.y = np.concatenate((self.y, label), axis=0)

    def __len__(self):

        """返回数据集当中的样本个数"""
        return self.x.shape[0]

    def __getitem__(self, index):

        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(self.y[index]).float()
