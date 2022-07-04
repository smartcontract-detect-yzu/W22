import json
import os
import shutil
import time

import torch
from etherscan.client import EmptyResponse
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer
from ponzi_detector.info_analyze.file_psc_analyze import FilePscAnalyzer
from ponzi_detector.model.MTCformer import TextCnn
from ponzi_detector.model.create_cfg_dataset import PonziDataSetCfg
from ponzi_detector.model.create_psc_dataset import PscDataset
from ponzi_detector.model.create_tpg_dataset import PonziDataSet
from ponzi_detector.model.graph_neural_network import CGCClass
from ponzi_detector.tools import OPCODE_MAP
import wsgiref.validate
from etherscan.contracts import Contract
import torchmetrics
from gensim.models.fasttext import FastText
import textdistance

# V1 版本旁氏合约数据库
DATASET_NAMES = [
    "sad_chain",
    "sad_tree",
    "xblock_dissecting",
    "buypool",
    "deposit",
    "5900"
]

# V2 版本旁氏合约数据库
PONZI_DATASET_NAME = [
    "labled_ponzi_src"
]

# V2 版本非旁氏合约数据库
NO_PONZI_DATASET_NAMES = [
    "no_ponzi",
    "dapp_src"
]

PONZI_JSON_MAP = {
    "buypool": "labeled_slice_record_buypool.json",
    "deposit": "labeled_slice_record_deposit.json",
    "sad_chain": "labeled_slice_record_sad_chain.json",
    "sad_tree": "labeled_slice_record_sad_tree.json",
    "xblock_dissecting": "labeled_slice_record_xblock_dissecting.json",
    "5900": "labeled_slice_record_5900.json"
}

NO_PONZI_JSON_MAP = {
    "no_ponzi": "labeled_slice_record_no_ponzi.json",
    "dapp_src": "labeled_slice_record_dapp_src.json"
}

TEST_PONZI_JSON_MAP = {
    "etherscan": "labeled_slice_record_etherscan.json"
}


def _get_features_for_xgboost(name, dataset_lines, tag):
    opcode_frequency = name + "/" + "frequency.json"
    drop_cnt = 0
    id_cnt_map = {}
    with open(opcode_frequency, "r") as f:
        line_infos = []
        info = json.load(f)
        opcodes_info = info["opcode_cnt"]
        total = info["total"]
        for opcode in opcodes_info:

            if opcode in OPCODE_MAP:
                current_id = OPCODE_MAP[opcode]
                current_cnt = opcodes_info[opcode]
                id_cnt_map[current_id] = current_cnt
            else:
                # 只保留76个特征
                drop_cnt += opcodes_info[opcode]
                continue

        total = total - drop_cnt
        for opcode in OPCODE_MAP:

            opcode_id = OPCODE_MAP[opcode]

            if opcode_id in id_cnt_map:
                freq = id_cnt_map[opcode_id] / total
                line_infos.append("{}".format(freq))
            else:
                line_infos.append("{}".format(0.0))

        info_str = tag
        for info in line_infos:
            info_str += ", {}".format(info)
        info_str += "\n"
        dataset_lines.append(info_str)


def calculate_metrics_by_address(preds, labels, address_list, ponzi_label=0):
    ponzi_address = {}
    no_ponzi_address = {}

    tp_address = {}
    fp_address = {}
    tn_address = {}
    fn_address = {}

    for batch_preds, batch_labels, batch_address in zip(preds, labels, address_list):
        for pred, label, address in zip(batch_preds, batch_labels, batch_address):

            # 个数统计
            if label == ponzi_label:
                ponzi_address[address] = 1
            else:
                no_ponzi_address[address] = 1

            # 当前地址还没有预测结果
            if address not in tp_address \
                    and address not in fp_address \
                    and address not in tn_address \
                    and address not in fn_address:

                if label == ponzi_label:
                    if pred == label:
                        tp_address[address] = 1
                    else:
                        fp_address[address] = 1
                else:
                    if pred == label:
                        tn_address[address] = 1
                    else:
                        fn_address[address] = 1

            # 如果存在预测结果
            else:
                if address in tp_address or address in tn_address:  # 已经预测对了，不用重新预测
                    continue

                elif address in fp_address or address in fn_address:  # 如果之前的切片预测错了，现在重新预测

                    # 首先判断当前的预测是否准确
                    if pred == label:  # 预测正确

                        if address in fp_address:
                            fp_address.pop(address)  # 删除错误的结果
                        else:
                            fn_address.pop(address)  # 删除错误的结果

                        if label == ponzi_label:
                            tp_address[address] = 1  # 添加预测正确
                        else:
                            tn_address[address] = 1  # 添加预测正确

                    # 如果还是预测错误 fn or fp, 之前预测错为fn 本次肯定还是fn
                    else:
                        pass  # 不需要更新

    # 计算
    TP = len(tp_address)
    TN = len(tn_address)
    FP = len(fp_address)
    FN = len(fn_address)

    total_data_num = TP + TN + FP + FN

    # 计算acc
    acc = (TP + TN) / (TP + TN + FP + FN)

    # 计算recall
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # 计算precision
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    # 计算f1
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("样本个数大检查: {}   ----- {}".format(len(ponzi_address), len(no_ponzi_address)))
    return acc, recall, precision, f1, total_data_num


def calculate_metrics(preds, labels, ponzi_label=0):
    TP = FP = TN = FN = 0
    ponzi_cnt = no_ponzi_cnt = 0
    for batch_preds, batch_labels in zip(preds, labels):
        for pred, label in zip(batch_preds, batch_labels):

            if label == ponzi_label:
                ponzi_cnt += 1
                if pred == label:
                    TP += 1
                else:
                    FP += 1
            else:
                no_ponzi_cnt += 1
                if pred == label:
                    TN += 1
                else:
                    FN += 1

    total_data_num = TP + TN + FP + FN

    # 计算acc
    acc = (TP + TN) / (TP + TN + FP + FN)

    # 计算recall
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    # 计算precision
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    # 计算f1
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("ppppp:{}  no_ppppppp:{}".format(ponzi_cnt, no_ponzi_cnt))
    return acc, recall, precision, f1, total_data_num


def do_valid(dataset, model, device):
    # 开始验证
    with torch.no_grad():
        model.eval()
        valid_off_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        valid_preds = []
        valid_labels = []
        correct = 0.
        loss = 0.
        criterion = torch.nn.CrossEntropyLoss()
        for data in valid_off_loader:
            data = data.to(device)
            out = model(data)

            pred = out.argmax(dim=1)
            label = data.y.argmax(dim=1)

            valid_preds.append(pred)
            valid_labels.append(label)

            batch_loss = criterion(out, data.y)
            correct += int((pred == label).sum())

            loss += batch_loss

        val_acc = correct / len(valid_off_loader.dataset)
        val_loss = loss / len(valid_off_loader.dataset)
        print("\nnormal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))

        acc, recall, precision, f1, total_num = calculate_metrics(valid_preds, valid_labels)
        print("total:{} \n结果指标\tacc:{} recall:{} precision:{} f1:{}".format(total_num, acc, recall, precision, f1))


class DataSet:

    def __init__(self, name="all"):

        self.name = name
        self.label_json = {}
        self.json_file_name = "slice_record_{}.json".format(name)
        self._get_label_json_file()
        self.labeled_json_dir = "./ponzi_detector/dataset/json/"

        self.api_key = 'api_key.json'
        self.to_download_list = "./examples/download/download_list.txt"
        self.to_download_dir = "./examples/download/"

        self.ponzi_file_names = []  # 数据集正样本集合
        self.no_ponzi_file_names = []  # 数据集负样本集合

        self.pyg_dataset = None  # PYG 数据集 包含训练和测试
        self.pyg_test_dataset = None  # PYG etherscan数据集，用于测试
        self.pyg_test_dataset_2_file = {}
        self.model_save_path = "./ponzi_detector/dataset/saved_model/"

        self.ponzi_cfg_filer_json = "ponzi_cfg_function.json"
        self.no_ponzi_cfg_filer_json = "no_ponzi_cfg_function.json"

        self.dup_filter = None
        self.filter_init()

    def _get_label_json_file(self):

        if self.name == "all":
            for name in DATASET_NAMES:
                json_name = "labeled_slice_record_{}.json".format(name)
                self.label_json[name] = json_name
        else:
            json_name = "labeled_slice_record_{}.json".format(self.name)
            self.label_json[self.name] = json_name

    def filter_init(self):
        with open("dup_file_nams.json", "r") as f:
            self.dup_filter = json.load(f)

    def get_work_dirs(self):

        dataset_prefixs = []
        analyze_prefixs = []

        if self.name != "all":

            dataset_prefixs.append("examples/ponzi_src/{}/".format(self.name))
            analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(self.name))
            cnt = 1
        else:

            for name in DATASET_NAMES:
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

            for name in NO_PONZI_DATASET_NAMES:
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

            cnt = len(dataset_prefixs)

        return dataset_prefixs, analyze_prefixs, cnt

    def get_work_dirs_v2(self):

        labels = []
        dataset_prefixs = []
        analyze_prefixs = []

        if self.name != "all":
            labels.append(1)
            dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", self.name))
            analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(self.name))

        else:
            for name in PONZI_DATASET_NAME:
                labels.append(1)
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

            for name in NO_PONZI_DATASET_NAMES:
                labels.append(0)
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

        return dataset_prefixs, analyze_prefixs, labels

    def get_work_dirs_with_name(self):

        names = []
        dataset_prefixs = []
        analyze_prefixs = []

        if self.name != "all" and self.name != "p":
            names.append(self.name)
            dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", self.name))
            analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(self.name))
            cnt = 1

        elif self.name == "all":

            for name in DATASET_NAMES:
                names.append(name)
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

            for name in NO_PONZI_DATASET_NAMES:
                names.append(name)
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

        elif self.name == "ponzi":
            for name in DATASET_NAMES:
                names.append(name)
                dataset_prefixs.append("{}{}/".format("./examples/ponzi_src/", name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

        cnt = len(dataset_prefixs)
        return names, dataset_prefixs, analyze_prefixs, cnt

    def pre_filter_dataset(self):
        """
        预处理数据集：删除那些没有交易行为的合约
        """
        removed_cnt = 0
        dst_dataset_prefix = "examples/ponzi_src/" + self.name

        dst_dataset_src_dir = dst_dataset_prefix + "/src"
        compile_error_files = []
        no_send_files = []

        g = os.walk(dst_dataset_src_dir)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.endswith(".sol"):
                    solfile_analyzer = SolFileAnalyzer(file_name, dst_dataset_src_dir)
                    solfile_analyzer.do_chdir()

                    print("开始分析: {}".format(file_name))
                    try:
                        ret = solfile_analyzer.do_filter_contract()
                        if ret is False:
                            # 需要删除
                            src = dst_dataset_src_dir + "/" + file_name
                            no_send_files.append(src)
                            removed_cnt += 1
                    except:
                        src = dst_dataset_src_dir + "/" + file_name
                        compile_error_files.append(src)
                        print("error:{}".format(file_name))

                    solfile_analyzer.revert_chdir()

        dst = dst_dataset_prefix + "/compile_error/"
        for compile_error_file in compile_error_files:
            shutil.move(compile_error_file, dst)

        dst = dst_dataset_prefix + "/no_send/"
        for no_send_file in no_send_files:
            shutil.move(no_send_file, dst)

        print("最终删除样本数为：{}".format(removed_cnt))

    def dataset_static(self):

        tem_prefix = "./examples/ponzi_src/analyze/"
        tem_dst_prefix = "./ponzi_detector/dataset/src/"
        address_to_id_map = {}
        ponzi_sc_cnt = 0
        ponzi_slice_cnt = 0
        file_name_id = 1
        g = os.walk(self.labeled_json_dir)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name.endswith(".json"):
                    dataset_name = file_name.split("labeled_slice_record_")[1].split(".json")[0]

                    # 过滤
                    if dataset_name == "etherscan":
                        continue

                    json_file = os.path.join(path, file_name)
                    print("开始分析： [{}]".format(file_name))
                    with open(json_file, "r+") as jf:
                        dataset_info = json.load(jf)
                        for sc_name in dataset_info:
                            sc_info = dataset_info[sc_name]
                            sc_cnt_flag = 0
                            if "slice" in sc_info:
                                for slices_info in sc_info["slice"]:
                                    for slice_info in slices_info:
                                        if "tag" in slice_info:
                                            ponzi_slice_cnt += 1
                                            if sc_cnt_flag == 0:
                                                src_file = tem_prefix + dataset_name + "/" + sc_name + "/" + sc_name + ".sol"
                                                address_to_id_map[file_name_id] = sc_name
                                                dst_file = tem_dst_prefix + str(file_name_id) + ".sol"
                                                shutil.copy(src_file, dst_file)
                                                file_name_id += 1
                                                ponzi_sc_cnt += 1
                                                sc_cnt_flag = 1
        with open("ponzi_id_nams.json", "w+") as f:
            f.write(json.dumps(address_to_id_map))

        print("【统计结果】旁氏合约样本个数：{} 切片个数：{}".format(ponzi_sc_cnt, ponzi_slice_cnt))

    def do_get_ast_json(self, pass_tag=1):

        """
        进行数据集分析，利用solc编译器得到ast json文件

        返回值：
          当前数据集的切片信息，以map形式返回
        """
        dataset_infos = {}
        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".sol"):  # 目前仅限solidity文件
                        src_file = os.path.join(path, file_name)

                        address = file_name.split(".sol")[0]
                        analyze_dir = analyze_prefix + address

                        if not os.path.exists(analyze_dir):
                            os.mkdir(analyze_dir)

                        if not os.path.exists(analyze_dir + "/" + file_name):
                            shutil.copy(src_file, analyze_dir)

                        done_file = analyze_dir + "/ast_done.txt"
                        if os.path.exists(done_file) and pass_tag:
                            print("========={}===========".format(file_name))
                            continue
                        else:

                            if os.path.exists(done_file):
                                os.remove(done_file)

                            print("\033[0;31;40m\t开始分析: {} \033[0m".format(file_name))
                            solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir)

                            solfile_analyzer.do_chdir()

                            # 解析编译器版本，并修改编译器版本
                            solfile_analyzer.do_file_analyze_prepare()

                            # 获得合约的ast 文件
                            solfile_analyzer.get_ast_json()

                            solfile_analyzer.revert_chdir()
                    else:
                        path_file_name = os.path.join(path, file_name)
                        os.remove(path_file_name)

        return dataset_infos

    def get_asm_for_dataset_from_bin(self, pass_tag=0):

        """
        数据集为非源码数据集，而是编译后的字节码数据集
        直接利用 evm disasm 进行反编译
        """

        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]
            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".bin"):
                        src_file = os.path.join(path, file_name)

                        address = file_name.split(".bin")[0]
                        analyze_dir = analyze_prefix + address

                        # 全部作为非旁氏合约样本
                        self.no_ponzi_file_names.append(analyze_dir)

                        if not os.path.exists(analyze_dir):
                            os.mkdir(analyze_dir)

                        if not os.path.exists(analyze_dir + "/" + file_name):
                            shutil.copy(src_file, analyze_dir)

                        done_file = analyze_dir + "/asm_done.txt"
                        pass_file = analyze_dir + "/pass.txt"

                        if os.path.exists(pass_file):
                            continue

                        if os.path.exists(done_file) and pass_tag:
                            print("========={}===========".format(file_name))
                            continue
                        else:

                            if os.path.exists(done_file):
                                os.remove(done_file)

                            solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir, file_type="bin")
                            solfile_analyzer.do_chdir()
                            solfile_analyzer.get_asm_from_bin()
                            solfile_analyzer.get_opcode_frequency_feature()
                            solfile_analyzer.revert_chdir()

    def get_asm_and_opcode_for_dataset(self, pass_tag=0):

        """
        将数据集中的所有.sol文件
        1.通过solc编译成字节码
        2.通过evm disasm将字节码反编译成汇编文件
        """

        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".sol"):
                        src_file = os.path.join(path, file_name)

                        address = file_name.split(".sol")[0]
                        analyze_dir = analyze_prefix + address

                        if not os.path.exists(analyze_dir):
                            os.mkdir(analyze_dir)

                        if not os.path.exists(analyze_dir + "/" + file_name):
                            shutil.copy(src_file, analyze_dir)

                        done_file = analyze_dir + "/asm_done.txt"
                        pass_file = analyze_dir + "/pass.txt"

                        if os.path.exists(pass_file):
                            continue

                        if os.path.exists(done_file) and pass_tag:
                            print("========={}===========".format(file_name))
                            continue
                        else:

                            if os.path.exists(done_file):
                                os.remove(done_file)

                            solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir)
                            solfile_analyzer.do_chdir()
                            solfile_analyzer.do_file_analyze_prepare()
                            solfile_analyzer.get_opcode_and_asm_file()
                            solfile_analyzer.get_opcode_frequency_feature()
                            solfile_analyzer.revert_chdir()

    def clean_up(self):

        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            analyze_prefix = analyze_prefix_list[i]
            g = os.walk(analyze_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".asm") \
                            or file_name.endswith(".json") \
                            or file_name.endswith(".bin") \
                            or file_name.endswith(".dot") \
                            or file_name.endswith(".png") \
                            or file_name.endswith(".txt") \
                            or file_name.endswith(".evm"):
                        src_file = os.path.join(path, file_name)
                        os.remove(src_file)

    def label_file_analyze(self):
        """
        根据标记文件 labeled_slice_record_<dataset>.json
        得到正负样本的名称
        """
        find_target_file_ponzi = {}
        prefix = self.labeled_json_dir
        analyze_prefix = "examples/ponzi_src/analyze/"
        for dataset_type in self.label_json:
            json_file = self.label_json[dataset_type]
            with open(prefix + json_file, "r") as f:
                print("filename: {}".format(json_file))
                dataset_info = json.load(f)
                is_ponzi = 0
                for file_name in dataset_info:

                    if file_name in self.dup_filter:
                        continue

                    contract_info = dataset_info[file_name]
                    if "slice" in contract_info:
                        for slice_info in contract_info["slice"]:

                            if "tag" in slice_info:
                                find_target_file_ponzi[file_name] = 1
                                file_path = analyze_prefix + "{}/".format(dataset_type) + file_name
                                self.ponzi_file_names.append(file_path)
                                with open(file_path + "/is_ponzi.txt", "w+") as f:
                                    f.write("1")
                                is_ponzi = 1
                                break

                        if is_ponzi == 0:
                            file_path = analyze_prefix + "{}/".format(dataset_type) + file_name
                            with open(file_path + "/is_no_ponzi.txt", "w+") as f:
                                f.write("0")
                            self.no_ponzi_file_names.append(file_path)

        if self.name == "all":
            dataset_type = "no_ponzi"
            json_file_path = prefix + "labeled_slice_record_no_ponzi.json"
            with open(json_file_path, "r") as f:
                dataset_info = json.load(f)

                for file_name in dataset_info:

                    if file_name in find_target_file_ponzi:
                        print("标签有误: {}".format(file_name))

                    file_path = analyze_prefix + "{}/".format(dataset_type) + file_name
                    self.no_ponzi_file_names.append(file_path)

                    # contract_info = dataset_info[file_name]
                    # if "slice" in contract_info:
                    #     if len(contract_info["slice"]) != 0:
                    #         file_path = analyze_prefix + "{}/".format(dataset_type) + file_name
                    #         self.no_ponzi_file_names.append(file_path)
                    # elif len(contract_info["slice"]) == 0:
                    #     print("name:{}".format(file_name))
                    #     print("slice:{}".format(contract_info["slice"]))

            print("jfp:{}".format(json_file_path))
        print("样本量：{}  {}".format(len(self.ponzi_file_names), len(self.no_ponzi_file_names)))

    def prepare_for_xgboost(self):

        ponzi_dataset_lines = []
        no_ponzi_dataset_lines = []

        for name in self.ponzi_file_names:
            print("ponzi: {}".format(name))
            _get_features_for_xgboost(name, ponzi_dataset_lines, "1")

        for name in self.no_ponzi_file_names:
            _get_features_for_xgboost(name, no_ponzi_dataset_lines, "0")

        with open("xgboost_dataset_{}_ponzi.csv".format(self.name), "w+") as f:
            f.writelines(ponzi_dataset_lines)

        with open("xgboost_dataset_{}_no_ponzi.csv".format(self.name), "w+") as f:
            f.writelines(no_ponzi_dataset_lines)

    def do_analyze_for_cfg_pdg(self, pass_tag=1):

        if self.name != "etherscan":
            self.ponzi_cfg_filer_json = "ponzi_cfg_function.json"
            self.no_ponzi_cfg_filer_json = "no_ponzi_cfg_function.json"
            with open(self.ponzi_cfg_filer_json) as f:
                ponzi_cfg_filter = json.load(f)

            with open(self.no_ponzi_cfg_filer_json) as f:
                no_ponzi_cfg_filter = json.load(f)
        else:
            self.ponzi_cfg_filer_json = "etherscan_ponzi_cfg_function.json"
            with open(self.ponzi_cfg_filer_json) as f:
                ponzi_cfg_filter = json.load(f)

        names, dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs_with_name()
        for i in range(cnt):
            dataset_name = names[i]
            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]

            # 构建过滤器
            if dataset_name in ponzi_cfg_filter:
                dataset_filters = ponzi_cfg_filter[dataset_name]
            else:
                dataset_filters = no_ponzi_cfg_filter[dataset_name]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:

                    if file_name.endswith(".sol"):  # 目前仅限solidity文件
                        file_name_without_sol = str(file_name).split(".sol")[0]

                        print(file_name_without_sol)
                        if file_name_without_sol in dataset_filters:

                            file_filters = dataset_filters[file_name_without_sol]
                            src_file = os.path.join(path, file_name)

                            address = file_name.split(".sol")[0]
                            analyze_dir = analyze_prefix + address

                            if not os.path.exists(analyze_dir):
                                os.mkdir(analyze_dir)

                            if not os.path.exists(analyze_dir + "/" + file_name):
                                shutil.copy(src_file, analyze_dir)

                            cfg_done_file = analyze_dir + "/cfg_done.txt"
                            pdg_done_file = analyze_dir + "/pdg_done.txt"
                            if pass_tag and os.path.exists(cfg_done_file) and os.path.exists(pdg_done_file):
                                print("========={}===========".format(file_name))
                                continue
                            else:

                                if os.path.exists(cfg_done_file):
                                    os.remove(cfg_done_file)

                                if os.path.exists(pdg_done_file):
                                    os.remove(pdg_done_file)

                                print("\033[0;31;40m\t开始分析: {} \033[0m".format(file_name))
                                print("过滤器：{}".format(file_filters))
                                solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir, cfg_filter=file_filters)

                                # 切换工作目录
                                solfile_analyzer.do_chdir()

                                # 解析编译器版本，并修改编译器版本
                                solfile_analyzer.do_file_analyze_prepare()

                                # 获得cfg和pdg
                                solfile_analyzer.do_analyze_a_file_for_cfg_pdg()

                                # 恢复工作目录
                                solfile_analyzer.revert_chdir()

    def do_get_psc(self, pass_tag=1):

        dataset_prefix_list, analyze_prefix_list, labels = self.get_work_dirs_v2()
        for i in range(len(labels)):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]
            label = labels[i]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:

                    if file_name.endswith(".sol"):  # 目前仅限solidity文件
                        src_file = os.path.join(path, file_name)

                        address = file_name.split(".sol")[0]
                        analyze_dir = analyze_prefix + address

                        if not os.path.exists(analyze_dir):
                            os.mkdir(analyze_dir)

                        if not os.path.exists(analyze_dir + "/" + file_name):
                            shutil.copy(src_file, analyze_dir)

                        done_file = analyze_dir + "/psc_done.txt"

                        if os.path.exists(done_file) and pass_tag:
                            print("========={}===========".format(file_name))
                            continue

                        else:
                            if os.path.exists(done_file):
                                os.remove(done_file)

                            print("\033[0;31;40m\t开始分析: {} \033[0m".format(file_name))
                            solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir)
                            solfile_analyzer.do_chdir()

                            psc_analyzer = FilePscAnalyzer(file_name, label)
                            # psc_analyzer.do_get_psc_for_file(FASTTEXT_MODEL)
                            psc_analyzer.get_psc_from_sol()

                            solfile_analyzer.revert_chdir()

    def get_duplicate_by_distance(self):

        with open("ponzi_id_nams.json", "r") as f:
            id_name_maps = json.load(f)

        content_map_cache = {}
        psc_file_map = {}
        dup_file_map = {}
        dataset_prefix_list, analyze_prefix_list, labels = self.get_work_dirs_v2()
        for i in range(len(labels)):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:

                    if file_name.endswith(".sol"):  # 目前仅限solidity文件

                        address = file_name.split(".sol")[0]
                        analyze_dir = analyze_prefix + address
                        psc_file_path = analyze_dir + "/" + "{}.psc".format(address)
                        psc_file_map[address] = psc_file_path

        with tqdm(total=len(psc_file_map)) as pbar:
            for address in psc_file_map:

                pbar.update(1)

                if address in dup_file_map:
                    continue

                if address in content_map_cache:
                    content = content_map_cache[address]
                else:
                    file_path = psc_file_map[address]
                    with open(file_path, "r") as f:
                        content = f.readline()
                        content_map_cache[address] = content

                for compare_address in psc_file_map:
                    if compare_address != address and compare_address not in dup_file_map:

                        if compare_address in content_map_cache:
                            compare_file_content = content_map_cache[compare_address]
                        else:
                            compare_file_path = psc_file_map[compare_address]
                            with open(compare_file_path, "r") as f:
                                compare_file_content = f.readline()
                                content_map_cache[compare_address] = compare_file_content

                        sim = textdistance.hamming.normalized_similarity(content, compare_file_content)
                        if sim > 0.9:
                            # 如果二者的汉明距离大于0.9，说明二者重复
                            dup_file_map[compare_address] = 1

        print("去重之后的结果：{}".format(len(dup_file_map)))
        dup_name_maps = {}
        for filed_id in dup_file_map:
            dup_name = id_name_maps[filed_id]
            dup_name_maps[dup_name] = 1

        with open("dup_file_nams.json", "w+") as f:
            f.write(json.dumps(dup_name_maps))

    def do_analyze(self, pass_tag=1):

        """
        进行数据集分析

        返回值：
          当前数据集的切片信息，以map形式返回
        """
        failed_file = []
        dataset_infos = {}
        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".sol"):  # 目前仅限solidity文件
                        src_file = os.path.join(path, file_name)

                        address = file_name.split(".sol")[0]
                        analyze_dir = analyze_prefix + address

                        if not os.path.exists(analyze_dir):
                            os.mkdir(analyze_dir)

                        if not os.path.exists(analyze_dir + "/" + file_name):
                            shutil.copy(src_file, analyze_dir)

                        done_file = analyze_dir + "/done_ok.txt"
                        pass_file = analyze_dir + "/pass.txt"
                        error_file = analyze_dir + "/error_pass.txt"

                        if os.path.exists(pass_file) or os.path.exists(error_file):
                            continue

                        if os.path.exists(done_file) and pass_tag:
                            print("========={}===========".format(file_name))
                            continue
                        else:

                            if os.path.exists(done_file):
                                os.remove(done_file)

                            print("\033[0;31;40m\t开始分析: {} \033[0m".format(file_name))
                            solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir)
                            solfile_analyzer.do_chdir()

                            # 解析编译器版本，并修改编译器版本
                            solfile_analyzer.do_file_analyze_prepare()

                            # 字节码特征提取
                            solfile_analyzer.get_opcode_and_asm_file()
                            solfile_analyzer.get_opcode_frequency_feature()

                            # 源代码语义特征分析
                            try:
                                slice_infos = solfile_analyzer.do_analyze_a_file()
                                info = {
                                    "addre": solfile_analyzer.addre,
                                    "slice": slice_infos
                                }
                                dataset_infos[solfile_analyzer.addre] = info

                                with open("done_ok.txt", "w+") as f:
                                    f.write("done")
                            except:
                                print("error file:{}".format(file_name))
                                with open("error_pass.txt", "w+") as f:
                                    f.write("error")
                                failed_file.append(file_name)

                            solfile_analyzer.revert_chdir()
                    else:
                        path_file_name = os.path.join(path, file_name)
                        os.remove(path_file_name)

        return dataset_infos

    def download_solidity_contracts(self):
        """"
        利用etherscan的api下载代码，
        Note：用不了，一直无法连接不知道为什么
        """
        with open(self.api_key, mode='r') as key_file:

            key = json.loads(key_file.read())['key']
            with open(self.to_download_list, "r") as f:

                lines = f.readlines()
                for address in lines:

                    if os.path.exists("{}/{}.null".format(self.to_download_dir, address)) \
                            or os.path.exists("{}/{}.sol".format(self.to_download_dir, address)):
                        continue

                    print("下载：{}".format(address))
                    api = Contract(address=address, api_key=key)
                    try:
                        sourcecode = api.get_sourcecode()
                    except EmptyResponse:
                        continue

                    if len(sourcecode[0]['SourceCode']) == 0:
                        file_name = "{}/{}.null".format(self.to_download_dir, address)
                        with open(file_name, "w+") as f:
                            f.write(sourcecode[0]['SourceCode'])
                    else:
                        file_name = "{}/{}.sol".format(self.to_download_dir, address)
                        with open(file_name, "w+") as f:
                            f.write(sourcecode[0]['SourceCode'])

    def _do_test(self, model, dataset, device, etherscan=0):

        # 开始验证
        with torch.no_grad():
            model.eval()
            valid_off_loader = DataLoader(dataset, batch_size=64, shuffle=True)
            valid_preds = []
            valid_labels = []
            valid_address = []
            records = []
            total_address = {}
            detected_map = {}
            correct = 0.
            loss = 0.
            criterion = torch.nn.CrossEntropyLoss()
            for data in valid_off_loader:
                data = data.to(device)
                out = model(data)

                pred = out.argmax(dim=1)
                label = data.y.argmax(dim=1)

                valid_preds.append(pred)
                valid_labels.append(label)
                valid_address.append(data.address)

                if etherscan == 1:
                    for idx, predict_false in enumerate(torch.ne(pred, label)):
                        file_name = data.json[idx]
                        record_info = "False: {} Name: {}  label:{}  pre:{}".format(predict_false,
                                                                                    file_name,
                                                                                    label[idx],
                                                                                    pred[idx])

                        records.append(record_info)

                        address = data.address[idx]
                        if address not in total_address:
                            total_address[address] = 1

                        if predict_false.item() is False:  # 檢測成功
                            if address not in detected_map:
                                detected_map[address] = 1

                batch_loss = criterion(out, data.y)
                correct += int((pred == label).sum())

                loss += batch_loss

            val_acc = correct / len(valid_off_loader.dataset)
            val_loss = loss / len(valid_off_loader.dataset)
            print("\nnormal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))

            # acc, recall, precision, f1, total_num = calculate_metrics(valid_preds, valid_labels)
            acc, recall, precision, f1, total_num = calculate_metrics_by_address(valid_preds, valid_labels, valid_address)

            print("{} total:{} 结果指标\tacc:{} recall:{} precision:{} f1:{}".format(self.name, total_num, acc, recall,
                                                                                 precision, f1))

            if etherscan == 1:
                print("=============={}/{}========================".format(len(detected_map), len(total_address)))
                if acc >= 0.7:
                    for record in records:
                        print(record)
                    print("===========================================")
        return acc

    def do_learning(self):

        feature_size = self.pyg_dataset[0].x.shape[1]
        edge_attr_size = self.pyg_dataset[0].edge_attr.shape[1]
        print("节点特征大小: {}  边特征大小: {}".format(feature_size, edge_attr_size))

        # 分割数据集
        train_size = int(len(self.pyg_dataset) * 0.7)
        valid_size = len(self.pyg_dataset) - train_size
        print("train_size:{}   valid_size:{}".format(train_size, valid_size))
        train_dataset, valid_dataset = torch.utils.data.random_split(self.pyg_dataset, [train_size, valid_size])

        # 训练数据
        train_off_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # 训练参数
        model_params = {
            "TRAINING_EPOCHS": 256,
            "MODEL_FEAT_SIZE": feature_size,
            "MODEL_LAYERS": 3,
            "MODEL_DROPOUT_RATE": 0,
            "MODEL_EDGE_DIM": edge_attr_size,
            "MODEL_EDGE_DENSE_NEURONS": 8,  # edge_attr_size -> 32  ## best: edge_attr_size->16->8
            "MODEL_EDGE_NEURONS": 8,  # 32 -> 16
            "MODEL_DENSE_NEURONS": 48,  # 100 -> 48
            "MODEL_OUT_CHANNELS": 2  # 每一类的概率
        }

        # 优化器参数
        solver = {
            "SOLVER_LEARNING_RATE": 0.0001,
            "SOLVER_SGD_MOMENTUM": 0.8,
            "SOLVER_WEIGHT_DECAY": 0.001
        }

        # # # 训练参数
        # model_params = {
        #     "TRAINING_EPOCHS": 84,
        #     "MODEL_FEAT_SIZE": feature_size,
        #     "MODEL_LAYERS": 3,
        #     "MODEL_DROPOUT_RATE": 0.1,
        #     "MODEL_DENSE_NEURONS": 48,  # 100 -> 48
        #     "MODEL_EDGE_DIM": edge_attr_size,
        #     "MODEL_OUT_CHANNELS": 2  # 每一类的概率
        # }
        #
        # # 优化器参数
        # solver = {
        #     "SOLVER_LEARNING_RATE": 0.001,
        #     "SOLVER_SGD_MOMENTUM": 0.8,
        #     "SOLVER_WEIGHT_DECAY": 0.001
        # }

        print(model_params)
        print(solver)

        # 构建模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CGCClass(model_params=model_params)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=solver["SOLVER_LEARNING_RATE"])

        # scheduler = StepLR(optimizer, step_size=100, gamma=0.25)

        # 损失函数
        criterion = torch.nn.CrossEntropyLoss()

        # 开始训练
        time_start = time.time()
        epochs = model_params["TRAINING_EPOCHS"]
        for epoch in range(epochs):
            model.train()
            training_loss = 0
            for i, data in enumerate(train_off_loader):
                optimizer.zero_grad()
                data = data.to(device)
                out = model(data)
                target = data.y
                loss = criterion(out, target)
                training_loss += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                # scheduler.step()

            training_loss /= len(train_off_loader.dataset)
            if epoch % 10 == 0 or epoch == (epochs - 1):
                print("epoch {} Training loss: {}".format(epoch, training_loss))

                # print("----------开始验证-----------")
                acc = self._do_test(model, valid_dataset, device)  # 验证集
                # self._do_test(model, self.pyg_test_dataset, device, etherscan=1)  # etherscan

        # time_end = time.time()
        # time_sum = time_end - time_start
        # print("训练时间：{}".format(time_sum))

        # # 保存模型:
        # name = "{}/{}_{}.pt".format(self.model_save_path, "model", str(acc)[2:4])
        # torch.save(model.state_dict(), name)

        # # 开始验证
        # with torch.no_grad():
        #     model.eval()
        #     valid_off_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
        #     valid_preds = []
        #     valid_labels = []
        #     valid_address = []
        #     correct = 0.
        #     loss = 0.
        #     criterion = torch.nn.CrossEntropyLoss()
        #     for data in valid_off_loader:
        #         data = data.to(device)
        #         out = model(data)
        #
        #         pred = out.argmax(dim=1)
        #         label = data.y.argmax(dim=1)
        #
        #         valid_preds.append(pred)
        #         valid_labels.append(label)
        #         valid_address.append(data.address)
        #
        #         batch_loss = criterion(out, data.y)
        #         correct += int((pred == label).sum())
        #
        #         loss += batch_loss
        #
        #     val_acc = correct / len(valid_off_loader.dataset)
        #     val_loss = loss / len(valid_off_loader.dataset)
        #     print("\nnormal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))
        #
        #     acc, recall, precision, f1, total_num = calculate_metrics_by_address(valid_preds,
        #                                                                          valid_labels,
        #                                                                          valid_address)
        #
        #     # acc, recall, precision, f1, total_num = calculate_metrics(valid_preds, valid_labels)
        #     print("total:{} \n结果指标\tacc:{} recall:{} precision:{} f1:{}".format(total_num, acc, recall, precision, f1))

        # 开始测试
        # with torch.no_grad():
        #     model.eval()
        #     test_off_loader = DataLoader(self.pyg_test_dataset, batch_size=64, shuffle=True)
        #     test_preds = []
        #     test_labels = []
        #     test_address = []
        #     correct = 0.
        #     loss = 0.
        #     criterion = torch.nn.CrossEntropyLoss()
        #     for data in test_off_loader:
        #
        #         data = data.to(device)
        #         out = model(data)
        #
        #         pred = out.argmax(dim=1)
        #         label = data.y.argmax(dim=1)
        #
        #         test_preds.append(pred)
        #         test_labels.append(label)
        #         test_address.append(data.address)
        #
        #         # 小数据集的详细结果
        #         for idx, predict_false in enumerate(torch.ne(pred, label)):
        #             file_name = data.json[idx]
        #             print("False: {} Name: {}  label:{}  pre:{}".format(predict_false, file_name, label[idx], pred[idx]))
        #
        #         batch_loss = criterion(out, data.y)
        #         correct += int((pred == label).sum())
        #         loss += batch_loss
        #
        #     val_acc = correct / len(test_off_loader.dataset)
        #     val_loss = loss / len(test_off_loader.dataset)
        #     print("\nnormal test loss: {}\taccuracy:{}".format(val_loss, val_acc))
        #
        #     # acc, recall, precision, f1, total_num = calculate_metrics(test_preds, test_labels)
        #     acc, recall, precision, f1, total_num = calculate_metrics_by_address(test_preds,
        #                                                                          test_labels,
        #                                                                          test_address)
        #     print("total:{} \n结果指标\tacc:{} recall:{} precision:{} f1:{}".format(total_num, acc, recall, precision, f1))

        # # 保存模型:
        # print("----------开始验证-----------")
        # name = "{}/{}_{}.pt".format(self.model_save_path, "model", str(acc)[2:4])
        # torch.save(model.state_dict(), name)

    def prepare_for_cfg_dataset(self, label):

        all_cfg_infos = {}
        cfg_names = {}
        function_cnt = 0

        if label == "ponzi":
            json_file_name = "ponzi_cfg_function.json"
            datasets_map = PONZI_JSON_MAP

        elif label == "no_ponzi":
            json_file_name = "no_ponzi_cfg_function.json"
            datasets_map = NO_PONZI_JSON_MAP

        elif label == "etherscan":
            json_file_name = "etherscan_ponzi_cfg_function.json"
            datasets_map = TEST_PONZI_JSON_MAP
        else:
            datasets_map = None
            raise RuntimeError("错误的数据集")

        for dataset_name in datasets_map:

            dataset_cfg_infos = {}
            all_cfg_infos[dataset_name] = dataset_cfg_infos

            json_file = self.labeled_json_dir + datasets_map[dataset_name]
            print("目标文件:{}".format(json_file))

            with open(json_file, "r") as f:

                dataset_infos = json.load(f)
                for file_name in dataset_infos:

                    if file_name not in dataset_cfg_infos:
                        dataset_cfg_infos[file_name] = {}

                    cfg_names.clear()
                    target_infos = dataset_infos[file_name]

                    if "slice" not in target_infos:
                        continue

                    for slice_info in target_infos["slice"]:

                        if label == "no_ponzi" or "tag" in slice_info:

                            slice_name = slice_info["name"]
                            slice_split_info = str(slice_name).split("_")

                            func_name = slice_split_info[-2]  # 倒数第二个是函数名

                            contract_name = ""
                            for part_name in slice_split_info[:-2]:  # 之前的都是合约名
                                contract_name += "{}_".format(part_name)
                            contract_name = contract_name[:-1]

                            if contract_name not in dataset_cfg_infos[file_name]:
                                dataset_cfg_infos[file_name][contract_name] = []

                            cfg_fun_name = contract_name + func_name + "_cfg"
                            if cfg_fun_name not in cfg_names:
                                cfg_names[cfg_fun_name] = 1
                                function_cnt += 1
                                dataset_cfg_infos[file_name][contract_name].append(func_name)

        print("{} 函数样本数量: {}".format(label, function_cnt))

        with open(json_file_name, "w+") as f:
            f.write(json.dumps(all_cfg_infos))

    def prepare_dataset_for_learning(self):
        """
        为进行神经网络训练创建数据集
        """
        root_dir = 'ponzi_detector/dataset/'
        pyg_dataset = PonziDataSet(root=root_dir, dataset_type="slice")
        self.pyg_dataset = pyg_dataset

        root_dir = 'ponzi_detector/dataset/etherscan'
        pyg_test_dataset = PonziDataSet(root=root_dir, dataset_type="etherscan")
        self.pyg_test_dataset = pyg_test_dataset
        self.pyg_test_dataset_2_file = pyg_test_dataset.id_2_json

    def prepare_dataset_for_learning_cfg(self):
        """
        为进行神经网络训练创建数据集
        """

        root_dir = 'ponzi_detector/dataset/cfg'
        pyg_dataset = PonziDataSetCfg(root=root_dir, dataset_type="cfg")
        self.pyg_dataset = pyg_dataset

        # root_dir = 'ponzi_detector/dataset/cfg/big'
        # pyg_test_dataset = PonziDataSetCfg(root=root_dir, dataset_type="big")
        # self.pyg_test_dataset = pyg_test_dataset

        root_dir = 'ponzi_detector/dataset/cfg/etherscan'
        pyg_test_dataset = PonziDataSetCfg(root=root_dir, dataset_type="etherscan")
        self.pyg_test_dataset = pyg_test_dataset

    def prepare_dataset_for_learning_pdg(self):
        """
        为进行神经网络训练创建数据集
        """
        root_dir = 'ponzi_detector/dataset/pdg'
        pyg_dataset = PonziDataSetCfg(root=root_dir, dataset_type="pdg")
        self.pyg_dataset = pyg_dataset

        # root_dir = 'ponzi_detector/dataset/cfg/big'
        # pyg_test_dataset = PonziDataSetCfg(root=root_dir, dataset_type="big")
        # self.pyg_test_dataset = pyg_test_dataset

        root_dir = 'ponzi_detector/dataset/pdg/etherscan'
        pyg_test_dataset = PonziDataSetCfg(root=root_dir, dataset_type="etherscan")
        self.pyg_test_dataset = pyg_test_dataset

    def prepare_dataset_for_learning_cfg_pdg(self):
        """
        为进行神经网络训练创建数据集
        """

        print("==============learning_for_cfg_pdg=================")

        root_dir = 'ponzi_detector/dataset/cfg_pdg'
        pyg_dataset = PonziDataSetCfg(root=root_dir, dataset_type="cfg_pdg")
        self.pyg_dataset = pyg_dataset

        # root_dir = 'ponzi_detector/dataset/cfg/big'
        # pyg_test_dataset = PonziDataSetCfg(root=root_dir, dataset_type="big")
        # self.pyg_test_dataset = pyg_test_dataset

        root_dir = 'ponzi_detector/dataset/pdg/etherscan'
        pyg_test_dataset = PonziDataSetCfg(root=root_dir, dataset_type="etherscan")
        self.pyg_test_dataset = pyg_test_dataset

    def prepare_dataset_for_psc(self):
        """
        为进行神经网络训练创建数据集
        """
        psc_dataset = PscDataset()
        print(psc_dataset.__len__())

        # 分割数据集
        train_size = int(psc_dataset.__len__() * 0.7)
        valid_size = int(psc_dataset.__len__()) - train_size
        print("train_size:{}   valid_size:{}".format(train_size, valid_size))
        train_dataset, valid_dataset = torch.utils.data.random_split(psc_dataset, [train_size, valid_size])

        train_off_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=False)
        valid_off_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True, drop_last=False)

        # 构建模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TextCnn()
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # 损失函数
        criterion = torch.nn.CrossEntropyLoss()

        epochs = 32
        training_loss = 0
        for epoch in range(1, epochs + 1):
            for i, data in enumerate(train_off_loader):
                optimizer.zero_grad()
                feature = data[0].to(device)
                out = model(feature)
                label = data[1]
                loss = criterion(out, label)
                training_loss += loss.item() * len(label.shape)
                loss.backward()
                optimizer.step()

            training_loss /= len(train_off_loader.dataset)
            print("epoch {} Training loss: {}".format(epoch, training_loss))

        # 开始验证
        with torch.no_grad():
            model.eval()
            valid_preds = []
            valid_labels = []
            correct = 0.
            loss = 0.
            criterion = torch.nn.CrossEntropyLoss()
            for data in valid_off_loader:
                feature = data[0].to(device)
                out = model(feature)
                label = data[1]

                pred = out.argmax(dim=1)
                label = label.argmax(dim=1)

                valid_preds.append(pred)
                valid_labels.append(label)

                # batch_loss = criterion(out, data.y)
                correct += int((pred == label).sum())

                # loss += batch_loss

            val_acc = correct / len(valid_off_loader.dataset)
            # val_loss = loss / len(valid_off_loader.dataset)
            print("\nnormal Validation loss: {}\taccuracy:{}".format(0, val_acc))

            acc, recall, precision, f1, total_num = calculate_metrics(valid_preds, valid_labels)
            print("{} total:{} 结果指标\tacc:{} recall:{} precision:{} f1:{}".format(self.name, total_num, acc, recall,
                                                                                 precision, f1))
