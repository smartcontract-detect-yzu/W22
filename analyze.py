import argparse
import os
import shutil
import subprocess
from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from ponzi_detector.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer
from ponzi_detector.semantic_analyze.data_flow_analyzer import DataFlowAnalyzer
from ponzi_detector.semantic_analyze.interprocedural_analyzer import InterproceduralAnalyzer
from ponzi_detector.semantic_analyze.code_graph_constructor import CodeGraphConstructor
from slither import Slither

versions = ['0', '0.1.7', '0.2.2', '0.3.6', '0.4.26', '0.5.17', '0.6.12', '0.7.6', '0.8.6']


def select_solc_version(version_info):
    start = 0

    for i, char in enumerate(version_info):
        if char == '0' and start == 0:
            start = 1
            op_info = version_info[0:i]

            space_cnt = 0
            for c in op_info:
                if c == '^' or c == '>':
                    return versions[int(version_info[i + 2])]

                if c == '=':
                    last_char = version_info[i + 5]
                    if '0' <= last_char <= '9':
                        return version_info[i:i + 6]
                    else:
                        return version_info[i:i + 5]

                if c == ' ':
                    space_cnt += 1

            if space_cnt == len(op_info):
                last_char = version_info[i + 5]

                if '0' <= last_char <= '9':
                    return version_info[i:i + 6]
                else:
                    return version_info[i:i + 5]

    return "auto"


def parse_solc_version(file):
    with open(file, 'r', encoding='utf-8') as contract_code:

        mini = 100
        version_resault = None

        for line in contract_code:
            target_id = line.find("pragma solidity")
            if target_id != -1:
                new_line = line[target_id:]
                version_info = new_line.split("pragma solidity")[1]
                v = select_solc_version(version_info)

                if v[-3] == '.':
                    last_version = int(v[-2:])
                else:
                    last_version = int(v[-1:])

                if mini > last_version:
                    mini = last_version
                    version_resault = v

                return version_resault

                # version_info = version_info.replace('\r', '').replace('\n', '').replace('\t', '')
                # print("info:%s  ---  version:%s" % (version_info, version_resault))

        if version_resault is None:
            version_resault = "0.4.26"

        return version_resault


def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('-t', type=str, default=None)
    parser.add_argument('-n', type=str, default=None)

    args = parser.parse_args()
    return args.t, args.n


if __name__ == '__main__':

    target, name = argParse()
    if name is not None:

        test_path = "examples/ponzi/"
        for file in os.listdir(test_path):
            if not file.endswith(".sol") and not file == "ast":
                os.remove(os.path.join(test_path, file))

        os.chdir(test_path)
        solc_version = parse_solc_version(name)
        print("========={} V: {}".format(name, solc_version))
        subprocess.check_call(["solc-select", "use", solc_version])

        slither = Slither(name)
        for contract in slither.contracts:

            contract_info = ContractInfo(contract)

            for function in contract.functions:

                if function.can_send_eth():

                    # 函数对象
                    function_info = FunctionInfo(contract_info, function)
                    if not function_info.has_trans_stmts():
                        # print("当前函数没有直接调用 .send 或者 .trans, 暂时不进行下一步分析")
                        continue

                    print("\n########################")
                    print("开始分析： {}".format(function_info.name))
                    print("########################\n")

                    # 当前函数的控制流分析器
                    control_flow_analyzer = ControlFlowAnalyzer(contract_info, function_info)

                    # 当前函数的数据流分析器
                    data_flow_analyzer = DataFlowAnalyzer(contract_info, function_info)

                    # 过程间分析器
                    inter_analyzer = InterproceduralAnalyzer(contract_info, function_info)

                    # 为当前函数创建代码图表示构建
                    code_constructor = CodeGraphConstructor(contract_info, function_info)

                    control_flow_analyzer.do_control_dependency_analyze()  # 控制流分析
                    data_flow_analyzer.do_data_semantic_analyze()  # 数据语义分析
                    inter_analyzer.do_interprocedural_analyze_for_state_def()  # 过程间全局变量数据流分析

                    # 语义分析完之后进行数据增强，为切片做准备
                    function_info.construct_dependency_graph()

                    # 切片
                    code_constructor.do_code_slice_by_internal_all_criterias()
                    function_info.debug_png_for_graph("sliced_pdg")

                    flag, _ = inter_analyzer.do_need_analyze_callee()
                    if flag is True:
                        chains = function_info.get_callee_chain()
                        inter_analyzer.graphs_pool_init()
                        inter_analyzer.do_interprocedural_analyze_for_call_chain(chains[1])
