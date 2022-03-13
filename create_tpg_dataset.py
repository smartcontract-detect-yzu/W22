import json

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

EDGE_TYPES = ["cfg_edges", "cdg_edges", "ddg_edges", "dfg_edges"]


class PonziDataSet(InMemoryDataset):

    def __init__(self,
                 root='examples/dataset',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.infercode = infer_code_init()

        self.raw = "{}/{}".format(root, "raw")
        self.processed = "{}/{}".format(root, "processed")
        super(PonziDataSet, self).__init__(root=root,
                                           transform=transform,
                                           pre_transform=pre_transform,
                                           pre_filter=pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回原始文件列表
    @property
    def raw_file_names(self):
        names = [
            "FastBetMultiplier01eth_pay_12.json"
        ]
        return names

    # 返回需要跳过的文件列表
    @property
    def processed_file_names(self):
        return ['data.pt']

    # 下载原生文件
    def download(self):
        return

    def process(self):

        data_list = []

        for json_file_name in self.raw_file_names:

            with open("{}/{}".format(self.raw, json_file_name), "r") as f:

                json_graph = json.load(f)
                node_vectors = []
                edge_dup = {}
                src = []
                dst = []

                if "nodes" in json_graph:
                    nodes_info = json_graph["nodes"]

                    for node_info in nodes_info:
                        expr = node_info["expr"]
                        v = self.infercode.encode([expr])  # note：infercode一次解析长度小于5
                        node_vectors.append(v)

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
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

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


if __name__ == '__main__':
    data = PonziDataSet()
    print(data[0])
    print(data.processed_file_names)
