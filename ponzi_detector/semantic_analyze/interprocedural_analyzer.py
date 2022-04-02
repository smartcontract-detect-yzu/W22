import networkx as nx
from typing import Dict, List
from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from ponzi_detector.semantic_analyze.code_graph_constructor import CodeGraphConstructor, do_prepare_before_merge
from ponzi_detector.semantic_analyze.code_graph_constructor import do_merge_graph1_to_graph2
from ponzi_detector.semantic_analyze.code_graph_constructor import debug_get_graph_png
from ponzi_detector.semantic_analyze.code_graph_constructor import do_graph_relabel_before_merge
from ponzi_detector.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer

# 过程间分析器
from ponzi_detector.semantic_analyze.data_flow_analyzer import DataFlowAnalyzer


class InterproceduralAnalyzer:

    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo):

        self.function_info = function_info
        self.contract_info = contract_info

        # 修改全局变量语句集合
        self.external_state_def_nodes_map = None

        # 过程间函数信息存放
        # self.fun_criteria_pair = {}
        self.interprocedural_function_infos: Dict[int, FunctionInfo] = {}

        self.intra_fun_result: Dict[str, List[str]] = {}  # 函数内分析结果
        self.graphs_pool: Dict[str, nx.MultiDiGraph] = {}  # 图池：包含函数内、函数间分析结果

    def graphs_pool_init(self):

        name = self.function_info.name.__str__()
        sliced_graphs_map = self.function_info.get_sliced_pdg()  # 内部函数切片
        for criteria in sliced_graphs_map:
            key = "{}@{}@{}".format(name, "solidity_call", criteria)
            if name not in self.intra_fun_result:
                self.intra_fun_result[name] = [key]
            else:
                self.intra_fun_result[name].append(key)

            self.graphs_pool[key] = sliced_graphs_map[criteria]

    def _do_analyze_for_target_function(self, fid, criteria_content):

        """
        根据给定的切片准则，对函数进行切片
        并且不进行外部依赖分析
        返回切片完成后的图表示

        入参： criteria_name: 切片准则必须包含的字符串内容
        """

        function_info = self.contract_info.get_function_info_by_fid(fid)
        if function_info is None:
            function = self.contract_info.get_function_by_fid(fid)
            function_info = FunctionInfo(self.contract_info, function)

        function_info.get_external_criteria_by_content(criteria_content)

        # 全套大保健，需要优化
        control_flow_analyzer = ControlFlowAnalyzer(self.contract_info, function_info)
        data_flow_analyzer = DataFlowAnalyzer(self.contract_info, function_info)
        # inter_analyzer = InterproceduralAnalyzer(self.contract_info, function_info)
        graph_constructor = CodeGraphConstructor(self.contract_info, function_info)

        control_flow_analyzer.do_control_dependency_analyze()  # 控制流分析
        data_flow_analyzer.do_data_semantic_analyze()  # 数据语义分析
        # inter_analyzer.do_interprocedural_state_analyze()  # 过程间全局变量数据流分析
        function_info.construct_dependency_graph()  # 语义分析完之后进行数据增强，为切片做准备

        # 切片准则为外部函数调用的切片图表示构建
        graphs_map = graph_constructor.do_code_slice_by_criterias_type(criteria_content, criteria_type="external")

        return graphs_map

    def _construct_slice_call_chain_graph(self, chain):

        """
        通过图构建切片粒度过程间调用链
        一个函数中可能包含多个切片准则
        函数名@切片准则@切片位置

        entry  -- f1@c1@1  -- f2@c2@1 -- f3@c3@1 -- f4@c4@1 -- exit
               -- f1@c1@2  -- f2@c2@2 -- f3@c3@3
                           -- f2@c2@3

        """

        print("调用链：{}".format(chain))
        fun_criteria_pair = {}
        # 定义函数，切片准则对 <function id, criteria function id>
        last = None
        for level, target_fun in enumerate(chain):
            if last is not None:
                fun_criteria_pair[last] = target_fun["fid"]
            last = target_fun["fid"]

        print("函数间分析：函数切片对", fun_criteria_pair)

        # 构建图
        # 1.图初始化
        path_graph = nx.DiGraph()
        path_graph.graph["name"] = "函数{}过程间调用图".format(self.function_info.name)
        path_graph.add_node("entry")
        path_graph.add_node("exit")

        tmp = []
        leaf_function_name = None  # 调用图叶子节点
        for level, target_fid in enumerate(fun_criteria_pair):

            level_info = []
            tmp.append(level_info)
            criteria_fid = fun_criteria_pair[target_fid]
            callee_function = self.contract_info.get_function_by_fid(target_fid)
            criteria_function = self.contract_info.get_function_by_fid(criteria_fid)
            leaf_function_name = criteria_function.name

            print("函数 {} 中的切片准则函数 {}".format(callee_function.name, criteria_function.name))
            if callee_function.id not in self.interprocedural_function_infos:
                graphs_map = self._do_analyze_for_target_function(callee_function.id, criteria_function.name)
                for criteria in graphs_map:
                    key = "{}@{}@{}".format(callee_function.name, criteria_function.name, criteria)
                    level_info.append(key)
                    self.graphs_pool[key] = graphs_map[criteria]  # 所有结果入图池

                    path_graph.add_node(key)
                    if level == 0:
                        for to_node in tmp[level]:
                            path_graph.add_edge("entry", to_node)
                    else:
                        for from_node in tmp[level - 1]:
                            for to_node in tmp[level]:
                                path_graph.add_edge(from_node, to_node)

        # 图池中寻找最后函数
        keys = self.intra_fun_result[leaf_function_name]
        leaf_info = []
        print("keys: {}".format(keys))
        for key in keys:
            print("add node: {}".format(key))
            leaf_info.append(str(key))
            path_graph.add_node(str(key))
        tmp.append(leaf_info)

        level = len(fun_criteria_pair)
        for from_node in tmp[level - 1]:
            for to_node in tmp[level]:
                print("from: {} -> to:{}".format(from_node, to_node))
                path_graph.add_edge(from_node, to_node)

        for from_node in tmp[level]:
            path_graph.add_edge(from_node, "exit")
        return path_graph

    def get_callee_called_pairs(self):

        chains = self.function_info.get_callee_chain()

        for chain in chains:
            print("list:{}".format(chain))

        return chains

    def do_need_analyze_callee(self):

        """
        判断当前函数的的.send 和 .transfer是否数据依赖于入参
        func t(input a)
            .send(a)
        说明t受到了其callee的影响
        """

        # https://github.com/smartcontract-detect-yzu/slither/issues/11#issue-1184776553
        if self.function_info.visibility == "public" \
                or self.function_info.visibility == "external":
            return False, None

        input_params_info = self.function_info.get_input_params()
        graphs_infos = self.function_info.get_sliced_pdg()

        for params_name in input_params_info:
            input_param_graph_id = input_params_info[params_name]["key"]
            for criteria in graphs_infos:
                g: nx.DiGraph = graphs_infos[criteria]

                if input_param_graph_id in g.nodes:  # 如果入参在函数的依赖图中
                    print("需要进行跨函数分析： {} {}".format(criteria, g.nodes[criteria]["label"]))
                    return True, criteria

        return False, None

    def do_interprocedural_analyze_for_state_def(self):
        """
        过程间全局变量数据流分析：
            storage VAR
            function A { VAR = 1}
            function B {b = var}
        存在一个跨函数数据流分析 function A --> function B

        Return：
        external_nodes_map - <交易相关全局变量, [交易相关全局变量修改函数]>
        """

        this_function = self.function_info.function
        state_var_write_function_map = self.contract_info.state_var_write_function_map
        state_var_declare_function_map = self.contract_info.state_var_declare_function_map
        transaction_states = self.function_info.transaction_states
        if transaction_states is None:
            raise RuntimeError("请先进行数据流分析，才能获得交易依赖全局变量")

        # 跨函数交易依赖全局变量修改分析
        duplicate = {}
        external_nodes_map = {}  # <key:交易语句, value:外部函数的语句相关信息>

        for trans_criteria in transaction_states:  # 交易相关全局变量作为准则

            external_nodes_map_for_write_functions = {}
            for transaction_state in transaction_states[trans_criteria]:  # 交易语句究竟涉及哪些全局变量
                for trans_stat_var in transaction_state["vars"]:  # 遍历全部全局变量

                    duplicate.clear()

                    stack = [trans_stat_var]
                    duplicate[trans_stat_var] = 1
                    while len(stack) != 0:

                        current_var = stack.pop()
                        if current_var in state_var_write_function_map:
                            write_funs = state_var_write_function_map[current_var]

                        elif current_var in state_var_declare_function_map:

                            if "exp" in state_var_declare_function_map[current_var]:
                                # 此处表示该全局变量只做了声明，没有赋值 （e.g. char [] m;）
                                # print("\t\tDEC_EXP:{}".format(state_var_declare_function_map[current_var]["exp"]))
                                continue
                            elif "fun" in state_var_declare_function_map[current_var]:
                                write_funs = [state_var_declare_function_map[current_var]["fun"]]
                                # print("\t\tDEC_FUN:{}".format(state_var_declare_function_map[current_var]["fun"].name))
                            else:
                                raise RuntimeError("全局变量缺乏定义和修改")

                        else:
                            raise RuntimeError("全局变量缺乏定义和修改")

                        # 当前全局变量可能被多个外部函数修改
                        for write_fun in write_funs:

                            if write_fun.full_name == this_function.full_name:
                                continue  # 过滤非当前函数

                            if write_fun.id is None:
                                continue  # 全局变量初始化函数,在其他地方已经分析过了

                            print("\n外部写全局变量函数{}分析........".format(write_fun.full_name))
                            write_func_info = FunctionInfo(self.contract_info, write_fun)

                            # 记录下该函数中修改current_var的语句
                            def_var_infos = write_func_info.get_state_var_def_stmts_info(current_var)
                            for info in def_var_infos:

                                # 保存外部连接
                                if "ex_data_dep" not in info:
                                    info["ex_data_dep"] = [trans_criteria]
                                else:
                                    info["ex_data_dep"].append(trans_criteria)

                                # 结构体赋值展开
                                self.function_info.struct_assign_stmt_expand(info)

                                # 保存外部语句信息
                                if write_func_info.name not in external_nodes_map_for_write_functions:
                                    external_nodes_map_for_write_functions[write_func_info.name] = [info]
                                else:
                                    external_nodes_map_for_write_functions[write_func_info.name].append(info)

                                # 这些语句又使用了那些全局变量来修改current_var, 进行下一次的分析
                                for var_info in info["info"]:
                                    if var_info["type"] == "state" and var_info["op_type"] == "use":  # 只需要使用的全局变量
                                        for var in var_info["list"]:
                                            if var not in duplicate:
                                                # print("\t\t下一个变量{}".format(var))
                                                duplicate[var] = 1
                                                stack.append(var)

            if len(external_nodes_map_for_write_functions) > 0:  # 没有则不需要
                external_nodes_map[trans_criteria] = external_nodes_map_for_write_functions

        self.external_state_def_nodes_map = external_nodes_map
        self.function_info.external_state_def_nodes_map = external_nodes_map
        return external_nodes_map

    def do_interprocedural_analyze_for_call_chain(self, chain, idx):

        """
        合并调用链上所有函数的CFG
        for graph in chain：
            to_graph = Merge graph to to_graph
        """

        merged_graphs = None
        removed_edges = None
        fid = self.function_info.get_fid()
        if fid not in self.interprocedural_function_infos:
            self.interprocedural_function_infos[fid] = self.function_info  # leaf

        # 根据给定的调用链构建所有的过程间分析路径
        # 一条调用链可能存在多种路径
        print("调用链：{} {}".format(idx, chain))
        path_graph = self._construct_slice_call_chain_graph(chain)
        debug_get_graph_png(path_graph, "{}_函数间调用关系路径图".format(idx))

        # 进行图合并

        paths = nx.all_simple_paths(path_graph, source="entry", target="exit")
        for chain_idx, path in enumerate(list(paths)):

            merged_graph_name = to_graph_key = to_graph = None
            for idx, g_key in enumerate(path[1:-1]):

                if to_graph_key is None:
                    merged_graph_name = g_key
                    to_graph_key = g_key
                else:
                    merged_graph_name += "_{}".format(g_key)
                    g = self.graphs_pool[g_key]
                    g = do_graph_relabel_before_merge(g, g.graph["name"])

                    to_graph_name = self.graphs_pool[to_graph_key].graph["name"]

                    if to_graph is None:
                        to_graph = self.graphs_pool[to_graph_key]
                        to_graph, removed_edges = do_prepare_before_merge(to_graph, to_graph_name)

                    pos_at_to_graph = "{}@{}".format(to_graph_name, str(to_graph_key).split("@")[-1])
                    print("merge {} to {}".format(g.graph["name"], to_graph.graph["name"]))
                    to_graph = do_merge_graph1_to_graph2(g, to_graph, pos_at_to_graph)
                    to_graph_key = g_key

            # 当前path合并后的结果
            to_graph.graph["name"] = merged_graph_name
            to_graph.add_edges_from(removed_edges)

            # 将图中剩余的内部调用展开
            merged_graph = self.do_interprocedural_analyze_without_slice_criteria(to_graph)
            merged_graph.graph["name"] = "Expand_" + merged_graph_name
            debug_get_graph_png(merged_graph, "合并chain_{}".format(chain_idx), dot=False)

        return merged_graphs

    def do_interprocedural_analyze_without_slice_criteria(self, to_graph: nx.MultiDiGraph):
        """
        图中存在外部函数调用，并且这些外部函数不包含切片准则
        将函数的PDG直接嫁接到原始图表示中
        function A {
            call function B  -- 展开
        }

        """

        for node_id in to_graph.nodes:

            # Note: 图中当前节点为函数调用，需要进行展开
            node_info = to_graph.nodes[node_id]
            if "called" in node_info:

                fid = node_info["called"][0]
                function_info = self.contract_info.get_function_info_by_fid(fid)
                if function_info is None:
                    called_function = self.contract_info.get_function_by_fid(fid)
                    function_info = FunctionInfo(self.contract_info, called_function)

                # 目标函数名
                called_function_name = function_info.name

                # 全套大保健，需要优化
                control_flow_analyzer = ControlFlowAnalyzer(self.contract_info, function_info)
                data_flow_analyzer = DataFlowAnalyzer(self.contract_info, function_info)
                inter_analyzer = InterproceduralAnalyzer(self.contract_info, function_info)
                graph_constructor = CodeGraphConstructor(self.contract_info, function_info)

                control_flow_analyzer.do_control_dependency_analyze()  # 控制流分析
                data_flow_analyzer.do_data_semantic_analyze()  # 数据语义分析
                inter_analyzer.do_interprocedural_analyze_for_state_def()  # 过程间全局变量数据流分析
                function_info.construct_dependency_graph()  # 语义分析完之后进行数据增强，为切片做准备

                graph = graph_constructor.do_code_create_without_slice()  # 构图

                graph, removed_semantic_edges = do_prepare_before_merge(graph, called_function_name)
                to_graph = do_merge_graph1_to_graph2(graph, to_graph, node_id)  # 内部函数调用展开
                to_graph.add_edges_from(removed_semantic_edges)

        return to_graph

    def do_inter_function_call_expand(self):

        # 所有的切片
        sliced_graphs = self.function_info.sliced_pdg

        for criteria in sliced_graphs:
            graph = sliced_graphs[criteria]
            self.do_interprocedural_analyze_without_slice_criteria(graph)
