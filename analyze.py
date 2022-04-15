import argparse
import json
import os

from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer
from ponzi_detector.solidity_dataset import DataSet
from ponzi_detector.tools import Tools


def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('-d', type=str, default=None)
    parser.add_argument('-n', type=str, default=None)
    parser.add_argument('-op', type=str, default="p")

    args = parser.parse_args()
    return args.d, args.n, args.op


if __name__ == '__main__':

    dataset, name, operate = argParse()

    if name is not None:

        test_path = "examples/ponzi/"
        for file in os.listdir(test_path):
            if not file.endswith(".sol") and not file == "ast":
                os.remove(os.path.join(test_path, file))

        file_analyzer = SolFileAnalyzer(file_name=name, work_path=test_path)

        file_analyzer.do_chdir()
        file_analyzer.do_file_analyze_prepare()  # 环境配置
        file_infos = file_analyzer.do_analyze_a_file(test_mode=1)

    if operate == "download":
        tools = Tools()
        tools.download_from_etherscan_by_list()

    if dataset is not None:

        data_set = DataSet(dataset)  # 数据集

        if operate == "learning":
            data_set.prepare_dataset_for_learning()
            data_set.do_learning()

        elif operate == "filter":
            if dataset == "no_ponzi" or dataset == "dapp_src":
                data_set.pre_filter_dataset()

        elif operate == "asm":

            # 将数据集中的sol文件通过solc 编译成bin文件
            # 再利用evm disasm进行编译成opcdoe文件
            data_set.get_asm_and_opcode_for_dataset(pass_tag=1)
            data_set.label_file_analyze()
            data_set.prepare_for_xgboost()  # 生成xgboost格式的数据集

        elif operate == "static":
            data_set.dataset_static()

        elif operate == "clean":
            print("清除数据：{}".format(dataset))
            data_set.clean_up()

        elif operate == "bin":
            # 针对全是编译后的bin文件类型数据集
            data_set.get_asm_for_dataset_from_bin(pass_tag=1)
            data_set.prepare_for_xgboost()

        elif operate == "analyze":
            print("分析数据集：{}".format(dataset))
            dataset_info = data_set.do_analyze(pass_tag=1)
            if len(dataset_info) != 0:
                with open(data_set.json_file_name, "w+") as f:
                    json.dump(dataset_info, f)
        else:
            print("错误的操作符：{}".format(operate))
