import json
import os
import shutil

from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer

DATASET_NAMES = [
    "sad_chain",
    "sad_tree",
    "xblock_dissecting",
    "buypool",
    "deposit"
]


def _get_features_for_xgboost(name, opcode_2_id, opcode_id, dataset_lines, tag):
    opcode_frequency = name + "/" + "frequency.json"
    with open(opcode_frequency, "r") as f:
        line_infos = []
        info = json.load(f)
        opcodes_info = info["opcode_cnt"]
        total = info["total"]
        for opcode in opcodes_info:

            if opcode in opcode_2_id:
                current_id = opcode_2_id[opcode]
            else:
                opcode_2_id[opcode] = opcode_id
                current_id = opcode_id
                opcode_id += 1

            freq = opcodes_info[opcode] / total * 100
            line_infos.append("{}:{}".format(current_id, freq))

        info_str = tag
        for info in line_infos:
            info_str += " {}".format(info)
        info_str += "\n"
        dataset_lines.append(info_str)


class DataSet:

    def __init__(self, name="all"):

        self.name = name
        self.label_json = {}

        self._get_label_json_file()

        self.ponzi_file_names = []  # 数据集正样本集合
        self.no_ponzi_file_names = []  # 数据集负样本集合

    def _get_label_json_file(self):

        if self.name == "all":
            for name in DATASET_NAMES:
                json_name = "labeled_slice_record_{}.json".format(name)
                self.label_json[name] = json_name

        else:
            json_name = "labeled_slice_record_{}.json".format(self.name)
            self.label_json[json_name] = json_name

    def get_work_dirs(self):

        dataset_prefixs = []
        analyze_prefixs = []

        if self.name != "all":

            dataset_prefixs.append("examples/ponzi_src/{}/".format(self.name))
            analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(self.name))
            cnt = 1
        else:

            for name in DATASET_NAMES:
                dataset_prefixs.append("examples/ponzi_src/{}/".format(name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))
            cnt = len(DATASET_NAMES)

        return dataset_prefixs, analyze_prefixs, cnt

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
                    if file_name.endswith(".asm") or file_name.endswith(".bin") or file_name.endswith(".evm"):
                        src_file = os.path.join(path, file_name)
                        os.remove(src_file)

    def label_file_analyze(self):
        """
        根据标记文件 labeled_slice_record_<dataset>.json
        得到正负样本的名称
        """
        prefix = "examples/ponzi_src/"
        analyze_prefix = "examples/ponzi_src/analyze/"
        for dataset_type in self.label_json:
            json_file = self.label_json[dataset_type]
            with open(prefix + json_file, "r") as f:
                print("filename: {}".format(json_file))
                dataset_info = json.load(f)
                for file_name in dataset_info:

                    contract_info = dataset_info[file_name]
                    if "slice" in contract_info:
                        for slice_info in contract_info["slice"]:
                            if "tag" in slice_info:
                                self.ponzi_file_names.append(analyze_prefix + "{}/".format(dataset_type) + file_name)
                                break
                        self.no_ponzi_file_names.append(analyze_prefix + "{}/".format(dataset_type) + file_name)

        print("样本量：{}".format(len(self.ponzi_file_names)))

    def prepare_for_xgboost(self):

        opcode_id = 1
        opcode_2_id = {}
        dataset_lines = []

        for name in self.ponzi_file_names:
            _get_features_for_xgboost(name, opcode_2_id, opcode_id, dataset_lines, "1")

        for name in self.no_ponzi_file_names:
            _get_features_for_xgboost(name, opcode_2_id, opcode_id, dataset_lines, "0")

        with open("xgboost_dataset_{}.txt".format(self.name), "w+") as f:
            f.writelines(dataset_lines)

    def do_analyze(self, pass_tag=1):

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

                        if os.path.exists(pass_file):
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
                            solfile_analyzer.do_analyze_a_file()
                            with open("done_ok.txt", "w+") as f:
                                f.write("done")
                            solfile_analyzer.revert_chdir()
