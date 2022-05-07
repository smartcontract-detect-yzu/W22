import json
import random
import shutil

import textdistance
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
import os
from torch_geometric.io import read_planetoid_data
from torch_geometric.datasets import Planetoid
from infercode.client.infercode_client import InferCodeClient
import os
import logging
from tqdm import tqdm

PREFIX = "examples/ponzi_src/"
JSON_PREFIX = "ponzi_detector/dataset/json/"
EDGE_TYPES = ["cfg_edges", "cdg_edges", "ddg_edges", "dfg_edges"]
edge_type_2_id = {
    "cfg_edges": 0,
    "cdg_edges": 1,
    "ddg_edges": 2,
    "dfg_edges": 3
}

labeled_json_list = {
    "buypool": "labeled_slice_record_buypool.json",
    "deposit": "labeled_slice_record_deposit.json",
    "sad_chain": "labeled_slice_record_sad_chain.json",
    "sad_tree": "labeled_slice_record_sad_tree.json",
    "xblock_dissecting": "labeled_slice_record_xblock_dissecting.json",
    "5900": "labeled_slice_record_5900.json",
    "no_ponzi": "labeled_slice_record_no_ponzi.json",
    "dapp_src": "labeled_slice_record_dapp_src.json"
}


class PonziDataSetCfg(InMemoryDataset):

    def __init__(self,
                 root=None,
                 dataset_type=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.only_ponzi_flag = 0

        self.type = dataset_type
        self.fild_type = "cfg"
        if dataset_type == "cfg":
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
            self.json_files = {
                "ponzi_cfg_function.json": 1,
                "no_ponzi_cfg_function.json": 0
            }
        elif dataset_type == "pdg":
            self.fild_type = "pdg"
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
            self.json_files = {
                "ponzi_cfg_function.json": 1,
                "no_ponzi_cfg_function.json": 0
            }
        elif dataset_type == "etherscan":
            self.root = root + "cfg/" + "etherscan"
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
            self.json_files = {
                "etherscan_ponzi_cfg_function.json": 1
            }
        elif dataset_type == "big":
            self.root = root + "big"
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
            self.json_files = {
                "ponzi_cfg_function.json": 1,
                "no_ponzi_cfg_function.json": 0
            }

        print("构建数据集类型:{}， 执行路径:{}".format(self.type, self.processed_paths))

        self.filter_limit = 5
        self.filter_cnt = {}
        self.function_name = {}

        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.dataset_prefix = "examples/ponzi_src/analyze/"

        self.dup_filter = None
        self.filter_init()

        self.function_dup_filter = None
        self.function_filter_init()

        super(PonziDataSetCfg, self).__init__(root=root,
                                              transform=transform,
                                              pre_transform=pre_transform,
                                              pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def filter_init(self):
        with open("dup_file_nams.json", "r") as f:
            self.dup_filter = json.load(f)

    def _filter_function(self, names_map, function_name, label):

        if label == 1:
            if function_name in names_map:
                names_map[function_name] += 1
                if names_map[function_name] > self.filter_limit:
                    return True
                else:
                    return False
            else:
                names_map[function_name] = 1
                return False

        return False

    def _psc_map_init(self):

        psc_name_map = {}

        for target in self.json_files:
            label = self.json_files[target]
            if label == 1:

                with open(target, "r") as json_file:
                    dataset_infos = json.load(json_file)

                    for dataset_type in dataset_infos:

                        print(self.dataset_prefix)
                        print(dataset_type)
                        dataset_path = self.dataset_prefix + dataset_type + "/"
                        contracts_infos = dataset_infos[dataset_type]
                        for sol_file in contracts_infos:

                            if sol_file in self.dup_filter:
                                continue

                            sol_infos = contracts_infos[sol_file]
                            if len(sol_infos) == 0:
                                continue
                            else:
                                for contract in sol_infos:
                                    functions = sol_infos[contract]
                                    for function in functions:

                                        # if not self._filter_function(names_map, function, label):
                                        psc_file_name = "{}_{}_{}.txt".format(contract, function, "psc")
                                        psc_file_path = dataset_path + "{}/".format(sol_file) + psc_file_name

                                        cfg_file_name = "{}_{}_{}.json".format(contract, function, self.fild_type)
                                        cfg_file_path = dataset_path + "{}/".format(sol_file) + cfg_file_name
                                        if os.path.exists(psc_file_path):
                                            psc_name_map[psc_file_path] = cfg_file_path
        return psc_name_map

    def function_filter_init(self):

        function_content_map = {}
        function_dup_map = {}

        if os.path.exists("dup_cfg_function_nams.json"):
            with open("dup_cfg_function_nams.json", "r") as f:
                self.function_dup_filter = json.load(f)
            return
        else:
            self.function_dup_filter = {}

        psc_name_map = self._psc_map_init()
        with tqdm(total=len(psc_name_map)) as pbar:
            for psc_path in psc_name_map:

                pbar.update(1)

                if psc_path in function_dup_map:
                    continue

                if psc_path in function_content_map:
                    content = function_content_map[psc_path]
                else:
                    with open(psc_path, "r") as f:
                        content = f.read()
                        function_content_map[psc_path] = content

                for compare_psc_path in psc_name_map:

                    if compare_psc_path != psc_path and compare_psc_path not in function_dup_map:

                        if compare_psc_path in function_content_map:
                            compare_content = function_content_map[compare_psc_path]
                        else:
                            with open(compare_psc_path, "r") as f:
                                compare_content = f.read()
                                function_content_map[compare_psc_path] = compare_content

                        sim = textdistance.hamming.normalized_similarity(content, compare_content)
                        if sim > 0.9:
                            # 如果二者的汉明距离大于0.9，说明二者相似
                            function_dup_map[compare_psc_path] = 1
                            dup_cfg = psc_name_map[compare_psc_path]
                            self.function_dup_filter[dup_cfg] = 1

        print("去重之后的结果：{}".format(len(self.function_dup_filter)))
        with open("dup_cfg_function_nams.json", "w+") as f:
            f.write(json.dumps(self.function_dup_filter))

    # 返回原始文件列表
    @property
    def raw_file_names(self):

        names = []
        names_map = {}
        ponzi_cnt = no_ponzi_cnt = 0

        for target in self.json_files:
            label = self.json_files[target]
            with open(target, "r") as json_file:
                dataset_infos = json.load(json_file)

                for dataset_type in dataset_infos:
                    dataset_path = self.dataset_prefix + dataset_type + "/"
                    contracts_infos = dataset_infos[dataset_type]
                    for sol_file in contracts_infos:

                        if sol_file in self.dup_filter:
                            continue

                        sol_infos = contracts_infos[sol_file]
                        if len(sol_infos) == 0:
                            continue
                        else:
                            for contract in sol_infos:
                                functions = sol_infos[contract]
                                for function in functions:

                                    # if not self._filter_function(names_map, function, label):
                                    cfg_file_name = "{}_{}_{}.json".format(contract, function, self.fild_type)
                                    cfg_file_path = dataset_path + "{}/".format(sol_file) + cfg_file_name
                                    if cfg_file_path in self.function_dup_filter:
                                        continue

                                    if os.path.exists(cfg_file_path):
                                        names.append({"name": cfg_file_path, "label": label})
                                        if label == 1:
                                            ponzi_cnt += 1
                                        else:
                                            no_ponzi_cnt += 1

        print("\n [raw_file_names {}] 庞氏骗局样本数量：{}  非庞氏骗局样本数量：{}".format(random.randint(0, 10245555), ponzi_cnt,
                                                                        no_ponzi_cnt))
        return names

    @property
    def processed_file_names(self):
        """
        # 返回需要跳过的文件列表
        """
        return ['data.pt']

    def process(self):

        ponzi_cnt = no_ponzi_cnt = 0
        infercode = infer_code_init()
        data_list = []

        with tqdm(total=len(self.raw_file_names)) as pbar:
            for sample_info in self.raw_file_names:

                pbar.update(1)
                json_file_name = sample_info["name"]

                lable = sample_info["label"]
                if lable == 1:
                    ponzi_cnt += 1
                    y = [1, 0]  # [ponzi, no_ponzi]
                else:
                    no_ponzi_cnt += 1
                    y = [0, 1]

                with open(json_file_name, "r") as f:

                    json_graph = json.load(f)
                    node_vectors = []
                    tmp_edge_attr = {}
                    src = []
                    dst = []

                    if "nodes" in json_graph:
                        nodes_info = json_graph["nodes"]

                        if len(nodes_info) == 0:
                            continue

                        for node_info in nodes_info:
                            expr = node_info["expr"]
                            v = infercode.encode([expr])  # note：infercode一次解析长度小于5
                            node_vectors.append(v[0])

                    for edge_type in EDGE_TYPES:
                        if edge_type in json_graph:
                            cfg_edges_info = json_graph[edge_type]
                            for cfg_edge_info in cfg_edges_info:
                                from_id = cfg_edge_info["from"]
                                to_id = cfg_edge_info["to"]
                                key = "{}_{}".format(from_id, to_id)

                                if key not in tmp_edge_attr:
                                    edge_feature = [0, 0, 0, 0]
                                    edge_feature[edge_type_2_id[edge_type]] = 1
                                    tmp_edge_attr[key] = edge_feature
                                    src.append(from_id)
                                    dst.append(to_id)

                                else:
                                    edge_feature = tmp_edge_attr[key]
                                    edge_feature[edge_type_2_id[edge_type]] = 1
                                    tmp_edge_attr[key] = edge_feature

                edge_attr = []
                for u, v in zip(src, dst):
                    key = "{}_{}".format(u, v)
                    edge_attr.append(tmp_edge_attr[key])

                x = torch.tensor(node_vectors, dtype=torch.float)
                y = torch.tensor([y], dtype=torch.float)

                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                edge_index = torch.tensor([src, dst], dtype=torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, json=json_file_name)
                try:
                    if not data.edge_index.max() < data.x.size(0):
                        print(data.json)
                    else:
                        data_list.append(data)
                except:
                    print(json_file_name)
                    raise RuntimeError("#######################")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print("##########################")
        print("ponzi:{} no_ponzi:{}".format(ponzi_cnt, no_ponzi_cnt))
        print("##########################")
        print("\n\n")

    # 显示属性
    def __repr__(self):
        return '{}()'.format(self.dataname)


def infer_code_init():
    logging.basicConfig(level=logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # Change from -1 to 0 to enable GPU
    infercode = InferCodeClient(language="solidity")
    infercode.init_from_config()
    return infercode
