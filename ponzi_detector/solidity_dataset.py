import json
import os
import shutil

import torch
from etherscan.client import EmptyResponse
from torch_geometric.loader import DataLoader

from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer
from ponzi_detector.model.create_tpg_dataset import PonziDataSet
from ponzi_detector.model.graph_neural_network import CGCClass
from ponzi_detector.tools import OPCODE_MAP
import wsgiref.validate
from etherscan.contracts import Contract
import torchmetrics

DATASET_NAMES = [
    "sad_chain",
    "sad_tree",
    "xblock_dissecting",
    "buypool",
    "deposit",
    "5900"
]

NO_PONZI_DATASET_NAMES = [
    "no_ponzi"
]


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

    def _get_label_json_file(self):

        if self.name == "all":
            for name in DATASET_NAMES:
                json_name = "labeled_slice_record_{}.json".format(name)
                self.label_json[name] = json_name

        else:
            json_name = "labeled_slice_record_{}.json".format(self.name)
            self.label_json[self.name] = json_name

    def get_work_dirs(self):

        dataset_prefixs = []
        analyze_prefixs = []

        if self.name != "all":

            dataset_prefixs.append("examples/ponzi_src/{}/".format(self.name))
            analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(self.name))
            cnt = 1
        else:
            for name in zip(DATASET_NAMES, NO_PONZI_DATASET_NAMES):
                dataset_prefixs.append("{}{}/".format(self.labeled_json_dir, name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))

            cnt = len(dataset_prefixs)

        return dataset_prefixs, analyze_prefixs, cnt

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
                                                dst_file = tem_dst_prefix + str(file_name_id) + ".sol"
                                                shutil.copy(src_file, dst_file)
                                                file_name_id += 1
                                                ponzi_sc_cnt += 1
                                                sc_cnt_flag = 1

        print("【统计结果】旁氏合约样本个数：{} 切片个数：{}".format(ponzi_sc_cnt, ponzi_slice_cnt))

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
        prefix = self.labeled_json_dir
        analyze_prefix = "examples/ponzi_src/analyze/"
        for dataset_type in self.label_json:
            json_file = self.label_json[dataset_type]
            with open(prefix + json_file, "r") as f:
                print("filename: {}".format(json_file))
                dataset_info = json.load(f)
                is_ponzi = 0
                for file_name in dataset_info:

                    contract_info = dataset_info[file_name]
                    if "slice" in contract_info:
                        for slice_info in contract_info["slice"]:
                            if "tag" in slice_info:
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
                    contract_info = dataset_info[file_name]
                    if "slice" in contract_info:
                        if len(contract_info["slice"]) != 0:
                            file_path = analyze_prefix + "{}/".format(dataset_type) + file_name
                            self.no_ponzi_file_names.append(file_path)

                        elif len(contract_info["slice"]) == 0:
                            print("name:{}".format(file_name))
                            print("slice:{}".format(contract_info["slice"]))

            print("jfp:{}".format(json_file_path))
        print("样本量：{}  {}".format(len(self.ponzi_file_names), len(self.no_ponzi_file_names)))

    def prepare_for_xgboost(self):

        ponzi_dataset_lines = []
        no_ponzi_dataset_lines = []

        for name in self.ponzi_file_names:
            _get_features_for_xgboost(name, ponzi_dataset_lines, "1")

        for name in self.no_ponzi_file_names:
            _get_features_for_xgboost(name, no_ponzi_dataset_lines, "0")

        with open("xgboost_dataset_{}_ponzi.csv".format(self.name), "w+") as f:
            f.writelines(ponzi_dataset_lines)

        with open("xgboost_dataset_{}_no_ponzi.csv".format(self.name), "w+") as f:
            f.writelines(no_ponzi_dataset_lines)

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
        train_off_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 训练参数
        model_params = {
            "TRAINING_EPOCHS": 64,
            "MODEL_FEAT_SIZE": feature_size,
            "MODEL_LAYERS": 3,
            "MODEL_DROPOUT_RATE": 0.03,
            "MODEL_DENSE_NEURONS": 48,  # 100 -> 48
            "MODEL_EDGE_DIM": edge_attr_size,
            "MODEL_OUT_CHANNELS": 2  # 每一类的概率
        }

        # 优化器参数
        solver = {
            "SOLVER_LEARNING_RATE": 0.001,
            "SOLVER_SGD_MOMENTUM": 0.8,
            "SOLVER_WEIGHT_DECAY": 0.001
        }

        # 构建模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CGCClass(model_params=model_params)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=solver["SOLVER_LEARNING_RATE"],
                                     weight_decay=solver["SOLVER_WEIGHT_DECAY"])

        # 损失函数
        criterion = torch.nn.CrossEntropyLoss()

        # 开始训练
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

            training_loss /= len(train_off_loader.dataset)
            print("epoch {} Training loss: {}".format(epoch, training_loss))

        # 模型训练结果
        metrics_acc = torchmetrics.Accuracy()
        metrics_recall = torchmetrics.Recall(average='none', num_classes=2)
        metrics_precision = torchmetrics.Precision(average='none', num_classes=2)
        # metrics_f1 = torchmetrics.F1(average="macro", num_classes=2)

        # 开始验证
        with torch.no_grad():
            model.eval()
            valid_off_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
            correct = 0.
            loss = 0.
            criterion = torch.nn.CrossEntropyLoss()
            for data in valid_off_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                label = data.y.argmax(dim=1)
                batch_loss = criterion(out, data.y)
                correct += int((pred == label).sum())

                metrics_recall(pred, label)
                metrics_acc(pred, label)
                metrics_precision(pred, label)

                loss += batch_loss

            val_acc = correct / len(valid_off_loader.dataset)
            val_loss = loss / len(valid_off_loader.dataset)
            print("\nnormal Validation loss: {}\taccuracy:{}".format(val_loss, val_acc))

            total_acc = metrics_acc.compute()
            total_recall = metrics_recall.compute()
            total_precision = metrics_precision.compute()
            # total_auc = metrics_f1.compute()
            print("acc:{} recall:{} p:{}".format(total_acc, total_recall, total_precision))

        metrics_acc.reset()
        metrics_recall.reset()
        metrics_precision.reset()

        # 开始测试
        with torch.no_grad():
            model.eval()
            test_off_loader = DataLoader(self.pyg_test_dataset, batch_size=64, shuffle=True)
            correct = 0.
            loss = 0.
            criterion = torch.nn.CrossEntropyLoss()
            for data in test_off_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                label = data.y.argmax(dim=1)
                batch_loss = criterion(out, data.y)
                for idx, predict_false in enumerate(torch.ne(pred, label)):
                    if predict_false:
                        file_name = data.json[idx]

                correct += int((pred == label).sum())

                metrics_recall(pred, label)
                metrics_acc(pred, label)
                metrics_precision(pred, label)

                loss += batch_loss

            val_acc = correct / len(test_off_loader.dataset)
            val_loss = loss / len(test_off_loader.dataset)
            print("\nnormal test loss: {}\taccuracy:{}".format(val_loss, val_acc))
            total_acc = metrics_acc.compute()
            total_recall = metrics_recall.compute()
            total_precision = metrics_precision.compute()
            print("acc:{} recall:{} p:{}".format(total_acc, total_recall, total_precision))


        # 保存模型:
        name = "{}/{}_{}.pt".format(self.model_save_path, "model", str(val_acc)[2:4])
        torch.save(model.state_dict(), name)

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
