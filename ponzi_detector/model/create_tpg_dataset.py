import json
import shutil

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
    "dfg_edges": 3,
    "sddg_edges": 4
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


class PonziDataSet(InMemoryDataset):

    def __init__(self,
                 root=None,
                 dataset_type=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.only_ponzi_flag = 0
        self.filter_limit = 5

        if dataset_type == "cfg":
            self.type = dataset_type
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")

        elif dataset_type == "etherscan":
            self.only_ponzi_flag = 1
            self.type = dataset_type
            self.root = root + "etherscan"
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed_etherscan")
            self.json_list = {"etherscan": "labeled_slice_record_etherscan.json"}

        else:
            self.type = "slice"
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
            self.json_list = labeled_json_list

        print("构建数据集类型:{}， 执行路径:{}".format(self.type, self.processed_paths))

        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        self.json_id = 0
        self.file_json_2_id = {}
        self.id_2_json = {}

        self.dup_filter = None
        self.filter_init()

        super(PonziDataSet, self).__init__(root=root,
                                           transform=transform,
                                           pre_transform=pre_transform,
                                           pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def filter_init(self):
        with open("dup_file_nams.json", "r") as f:
            self.dup_filter = json.load(f)

    def _filter_function(self, names_map, function_name, label):

        if label == 1:

            if "buyPool" in function_name:

                if "buyPool" not in names_map:
                    names_map["buyPool"] = 1
                else:
                    names_map["buyPool"] += 1

                if names_map["buyPool"] > self.filter_limit:
                    return True
                else:
                    return False

            elif function_name in names_map:
                names_map[function_name] += 1
                if names_map[function_name] > self.filter_limit:
                    return True
                else:
                    return False
            else:
                names_map[function_name] = 1
                return False

        return False

    # 返回原始文件列表
    @property
    def raw_file_names(self):

        names = []
        cfg_names = {}
        names_map = {}

        ponzi_slice_cnt = 0
        no_ponzi_slice_cnt = 0

        for target in self.json_list:
            json_file = JSON_PREFIX + self.json_list[target]
            print("目标文件:{}".format(json_file))
            with open(json_file, "r") as f:

                dataset_infos = json.load(f)
                for sc_target in dataset_infos:

                    if sc_target in self.dup_filter:
                        continue

                    cfg_names.clear()
                    target_infos = dataset_infos[sc_target]
                    if "slice" not in target_infos:
                        continue

                    if self.type == "cfg":

                        for slice_info in target_infos["slice"]:
                            slice_name = slice_info["name"]
                            slice_split_info = str(slice_name).split("_")

                            func_name = slice_split_info[-2]  # 倒数第二个是函数名

                            contract_name = ""
                            for part_name in slice_split_info[:-2]:  # 之前的都是合约名
                                contract_name += "{}_".format(part_name)

                            cfg_fun_name = contract_name + func_name + "_cfg"
                            if cfg_fun_name not in cfg_names:
                                cfg_names[cfg_fun_name] = 1
                                sc_target_json = cfg_fun_name + ".json"
                                full_name = "{}analyze/{}/{}/{}".format(PREFIX, target, sc_target, sc_target_json)
                                if "tag" in slice_info:
                                    ponzi_slice_cnt += 1
                                    names.append({"name": full_name, "label": 1})
                                else:
                                    no_ponzi_slice_cnt += 1
                                    names.append({"name": full_name, "label": 0})
                    else:
                        json_prefix_map = {}
                        for slice_info in target_infos["slice"]:
                            prefix_json = slice_info["name"]
                            if "tag" in slice_info:
                                json_prefix_map[prefix_json] = 1
                            else:
                                json_prefix_map[prefix_json] = 0

                        sc_target_path = "{}analyze/{}/{}/".format(PREFIX, target, sc_target)
                        for file_name in os.listdir(sc_target_path):
                            if str(file_name).endswith(".json"):

                                # labeled jason: contract_function_sliceid 或者 contract_function_sliceid_external<>
                                sc_fun_slice = str(file_name).split(".json")[0]
                                for json_prefix in json_prefix_map:
                                    if json_prefix in sc_fun_slice:
                                        label = json_prefix_map[json_prefix]

                                        # if self._filter_function(names_map, sc_fun_slice, label):
                                        #     continue

                                        json_file_name = sc_target_path + sc_fun_slice + ".json"
                                        if label == 1:
                                            ponzi_slice_cnt += 1
                                            names.append({"name": json_file_name, "label": label, "address": sc_target})
                                        else:
                                            if not self.only_ponzi_flag:
                                                no_ponzi_slice_cnt += 1
                                                names.append(
                                                    {"name": json_file_name, "label": label, "address": sc_target})

        print("庞氏合约样本数量: {}".format(len(names)))
        print("正样本个数：{}  负样本个数：{}".format(ponzi_slice_cnt, no_ponzi_slice_cnt))
        return names

    # 返回需要跳过的文件列表
    @property
    def processed_file_names(self):
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
                address = sample_info["address"]
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

                        for node_info in nodes_info:
                            # if "@tag" in node_info["cfg_id"]:

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

                self.json_id += 1
                self.file_json_2_id[json_file_name] = self.json_id
                self.id_2_json[self.json_id] = json_file_name

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, address=address, json=json_file_name)
                if not data.edge_index.max() < data.x.size(0):
                    print(data.json)
                else:
                    data_list.append(data)

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


def create_dataset(json_files):
    pass


dataset_info = {
    "cfg": {
        "root": 'examples/ponzi_src/dataset/cfg'
    },
    "sliced": {
        "root": 'examples/ponzi_src/dataset/sliced'
    },
    "etherscan": {
        "root": 'examples/ponzi_src/dataset/etherscan'
    }
}
if __name__ == '__main__':
    d_type = "sliced"
    root_dir = dataset_info[d_type]["root"]

    data = PonziDataSet(dataset_type=d_type, root=root_dir)
    print(data[0])
    print(data.processed_file_names)
