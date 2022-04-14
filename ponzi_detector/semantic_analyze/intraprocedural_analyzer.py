import networkx as nx
from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from ponzi_detector.semantic_analyze.code_graph_constructor import CodeGraphConstructor
from ponzi_detector.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer
from ponzi_detector.semantic_analyze.data_flow_analyzer import DataFlowAnalyzer


# 过程内分析器，主要分析当前函数内部的调用情况
class IntraprocedureAnalyzer:

    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo):

        self.name = function_info.name
        self.function_info = function_info
        self.contract_info = contract_info
        self.intra_function_analyzers = {}

    def do_intra_procedure_analyze(self):

        print("\n【START:INTRA】============开始过程内分析:{}==============【START】".format(self.name))

        cfg: nx.MultiDiGraph = self.function_info.cfg
        for node in cfg.nodes:
            node_info = cfg.nodes[node]
            if "called" in node_info:

                called_fid = node_info["called"][0]  # 调用函数的fid
                called_params = node_info["called_params"]
                print("\t过程内函数调用：called 信息：")
                print("\t\tcfg name: {}  node id:{}".format(cfg.graph["name"], node))
                print("\t\tcalled_fid {}".format(called_fid))
                print("\t\tcalled_params {}".format(called_params))

                function_info = self.contract_info.get_function_info_by_fid(called_fid)
                if function_info is None:
                    called_function = self.contract_info.get_function_by_fid(called_fid)

                    if called_function is not None and called_function.is_implemented is not None:
                        print("\n\t【START: 过程内被调用函数分析】 分析对象：{}".format(called_function.name))
                        function_info = FunctionInfo(self.contract_info, called_function)
                    else:
                        # 当前cfg节点为外部的modify函数，没有fid，需要跳过
                        continue
                else:
                    print("\n\t【START: 过程内被调用函数分析】 分析对象：{}".format(function_info.name))

                # 特殊处理
                if function_info.name == "rand":
                    continue

                # 全套大保健
                control_flow_analyzer = ControlFlowAnalyzer(self.contract_info, function_info)
                data_flow_analyzer = DataFlowAnalyzer(self.contract_info, function_info)
                graph_constructor = CodeGraphConstructor(self.contract_info, function_info)

                control_flow_analyzer.do_control_dependency_analyze()  # 控制流分析
                data_flow_analyzer.do_data_semantic_analyze()  # 数据语义分析
                function_info.construct_dependency_graph()  # 生成依赖图，为切片做准备

                info = {
                    "node_id": node,
                    "fid": called_fid,
                    "function_info": function_info,
                    "control_flow_analyzer": control_flow_analyzer,
                    "data_flow_analyzer": data_flow_analyzer,
                    "graph_constructor": graph_constructor
                }

                # 保存,
                self.intra_function_analyzers[called_fid] = info
                self.function_info.register_intra_fun(node, info)

                print("\n\t【END: 过程内被调用函数分析】 分析对象：{}".format(function_info.name))

        print("\n【END:INTRA】============过程内分析完成：{}==============【END】".format(self.name))
