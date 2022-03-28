import argparse
import os
import shutil
import subprocess
from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from ponzi_detector.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer
from ponzi_detector.semantic_analyze.data_flow_analyzer import DataFlowAnalyzer
from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer
from ponzi_detector.semantic_analyze.interprocedural_analyzer import InterproceduralAnalyzer
from ponzi_detector.semantic_analyze.code_graph_constructor import CodeGraphConstructor
from slither import Slither


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

        file_analyzer = SolFileAnalyzer(file_name=name, work_path=test_path)
        file_analyzer.do_file_analyze_prepare()  # 解析前的准备工作

        slither = Slither(name)
        for contract in slither.contracts:

            contract_info = ContractInfo(contract)
            for function in contract.functions:

                if function.can_send_eth():

                    function_info = FunctionInfo(contract_info, function)  # 函数对象
                    if not function_info.has_trans_stmts():  # print("当前函数没有直接调用 .send 或者 .trans, 暂时不进行下一步分析")
                        continue

                    print("\n########################")
                    print("开始分析： {}".format(function_info.name))
                    print("########################\n")

                    control_flow_analyzer = ControlFlowAnalyzer(contract_info, function_info)  # 当前函数的控制流分析器
                    data_flow_analyzer = DataFlowAnalyzer(contract_info, function_info)  # 当前函数的数据流分析器
                    inter_analyzer = InterproceduralAnalyzer(contract_info, function_info)  # 过程间分析器
                    code_constructor = CodeGraphConstructor(contract_info, function_info)  # 为当前函数创建代码图表示构建

                    control_flow_analyzer.do_control_dependency_analyze()  # 控制流分析
                    data_flow_analyzer.do_data_semantic_analyze()  # 数据语义分析
                    inter_analyzer.do_interprocedural_analyze_for_state_def()  # 过程间全局变量数据流分析

                    # 语义分析完之后进行数据增强，为切片做准备
                    function_info.construct_dependency_graph()
                    function_info.debug_png_for_graph(graph_type="pdg")

                    # 切片
                    code_constructor.do_code_slice_by_internal_all_criterias()
                    function_info.debug_png_for_graph("sliced_pdg")

                    # 过程间分析
                    flag, _ = inter_analyzer.do_need_analyze_callee()
                    if flag is True:
                        chains = function_info.get_callee_chain()
                        inter_analyzer.graphs_pool_init()  # 初始化图池，为函数间图合并做准备
                        inter_analyzer.do_interprocedural_analyze_for_call_chain(chains[1])
