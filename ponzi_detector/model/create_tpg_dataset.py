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
EDGE_TYPES = ["cfg_edges", "cdg_edges", "ddg_edges", "dfg_edges"]
labeled_json_list = {
    "buypool": "labeled_slice_record_buypool.json",
    "deposit": "labeled_slice_record_deposit.json",
    "sad_chain": "labeled_slice_record_sad_chain.json",
    "sad_tree": "labeled_slice_record_sad_tree.json",
    "xblock_dissecting": "labeled_slice_record_xblock_dissecting.json"
}


class PonziDataSet(InMemoryDataset):

    def __init__(self,
                 root=None,
                 dataset_type=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        if dataset_type == "cfg":
            self.type = dataset_type
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
        elif dataset_type == "etherscan":
            self.type = dataset_type
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")
        else:
            self.type = "slice"
            self.root = root
            self.raw = "{}/{}".format(root, "raw")
            self.processed = "{}/{}".format(root, "processed")

        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(PonziDataSet, self).__init__(root=root,
                                           transform=transform,
                                           pre_transform=pre_transform,
                                           pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回原始文件列表
    @property
    def raw_file_names(self):
        names = []
        cfg_names = {}

        if self.type == "etherscan":
            g = os.walk(self.raw)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".json"):
                        full_name = "{}/{}".format(self.raw, file_name)
                        names.append({"name": full_name, "label": 1})
            return names

        for target in labeled_json_list:
            json_file = PREFIX + labeled_json_list[target]
            with open(json_file, "r") as f:

                dataset_infos = json.load(f)
                for sc_target in dataset_infos:

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
                                    names.append({"name": full_name, "label": 1})
                                else:
                                    names.append({"name": full_name, "label": 0})
                    else:
                        for slice_info in target_infos["slice"]:
                            sc_target_json = slice_info["name"] + ".json"
                            full_name = "{}analyze/{}/{}/{}".format(PREFIX, target, sc_target, sc_target_json)
                            if "tag" in slice_info:
                                names.append({"name": full_name, "label": 1})
                            else:
                                names.append({"name": full_name, "label": 0})

        print("庞氏合约样本数量: {}".format(len(names)))
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
                if lable == 1:
                    ponzi_cnt += 1
                    y = [1, 0]  # [ponzi, no_ponzi]
                else:
                    no_ponzi_cnt += 1
                    y = [0, 1]

                with open(json_file_name, "r") as f:

                    json_graph = json.load(f)
                    node_vectors = []
                    edge_dup = {}
                    src = []
                    dst = []

                    if "nodes" in json_graph:
                        nodes_info = json_graph["nodes"]

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

                                if key not in edge_dup:
                                    edge_dup[key] = 1
                                    src.append(from_id)
                                    dst.append(to_id)

                x = torch.tensor(node_vectors, dtype=torch.float)
                y = torch.tensor([y], dtype=torch.float)
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, y=y)
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
