import itertools
import json

import networkx as nx

from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from queue import LifoQueue


def save_graph_to_json_format(graph, key):
    graph_info = {}
    nodes = []
    cfg_edges = []
    cdg_edges = []
    ddg_edges = []
    dfg_edges = []
    cfg_to_graph_id = {}
    graph_id = 0

    file_name = "{}_{}_{}.json".format(graph.graph["contract_name"], graph.graph["name"], key)

    for node in graph.nodes:
        id = graph_id
        cfg_to_graph_id[node] = id
        graph_id += 1
        cfg_id = node
        type = graph.nodes[node]["type"]
        expr = graph.nodes[node]["expression"]
        node_info = {
            "id": id,
            "cfg_id": cfg_id,
            "expr": expr,
            "type": type
        }
        nodes.append(node_info)

    for u, v, d in graph.edges(data=True):

        edge_info = {
            "from": cfg_to_graph_id[u],
            "to": cfg_to_graph_id[v],
            "cfg_from": u,
            "cfg_to": v
        }

        if "type" in d:
            if d["type"] == "ctrl_dependency":
                cdg_edges.append(edge_info)
            if d["type"] == "data_dependency":
                ddg_edges.append(edge_info)
            if d["type"] == "data_flow":
                dfg_edges.append(edge_info)
        else:
            cfg_edges.append(edge_info)

    graph_info["nodes"] = nodes
    graph_info["cfg_edges"] = cfg_edges
    graph_info["cdg_edges"] = cdg_edges
    graph_info["ddg_edges"] = ddg_edges
    graph_info["dfg_edges"] = dfg_edges

    return graph_info, file_name


# 基于PDG的前向依赖发内心
def forward_dependence_analyze(pdg, criteria):
    """
    前向依赖分析:
    切片准则和其所依赖的语句集合
    """

    result = {}
    stack = LifoQueue()

    stack.put(str(criteria))
    result[str(criteria)] = 1

    while stack.qsize() > 0:

        current_stmt = stack.get()  # 栈顶语句出栈，进行依赖分析
        for successor_stmt in pdg.successors(current_stmt):  # 数据依赖 + 控制依赖关系

            for edge_id in pdg[current_stmt][successor_stmt]:
                edge_data = pdg[current_stmt][successor_stmt][edge_id]
                # print("分析：{} -{}- {}".format(current_stmt, edge_data, successor_stmt))

                if "type" in edge_data:  # 控制依赖、数据依赖边
                    # print("DEBUG 节点{} 依赖于 {} as {}".format(current_stmt, successor_stmt, edge_data["type"]))
                    if successor_stmt not in result:
                        result[successor_stmt] = 1
                        stack.put(successor_stmt)  # 压栈
    return result


def _remove_node(g, node):
    # NOTE: 删除节点的原则：边的属性全部继承source A---(1)--B---(2)---C
    # todo: 删除节点B时，边的类型继承A  A---(1)---C
    sources = []
    targets = []

    for source, _ in g.in_edges(node):
        edge = g[source][node]
        # print("from {} to {}(removed)：{}".format(source, node, g[source][node]))
        if "type" not in edge:  # note：过滤：只保留CFG边，依赖关系删除
            sources.append(source)

    for _, target in g.out_edges(node):
        edge = g[node][target]
        # print("from {}(removed) to {}：{}".format(node, target, g[node][target]))
        if "type" not in edge:  # note：过滤：只保留CFG边，依赖关系删除
            targets.append(target)

    # if g.is_directed():
    #     sources = [source for source, _ in g.in_edges(node)]
    #     targets = [target for _, target in g.out_edges(node)]
    # else:
    #     raise RuntimeError("cfg一定是有向图")

    new_edges = itertools.product(sources, targets)
    new_edges_with_data = []
    for cfg_from, cfg_to in new_edges:
        new_edges_with_data.append((cfg_from, cfg_to))

    # new_edges = [(source, target) for source, target in new_edges if source != target]  # remove self-loops
    g.add_edges_from(new_edges_with_data)
    g.remove_node(node)

    return g


def do_slice(graph, reserve_nodes):
    remove_nodes = []
    sliced_graph = nx.MultiDiGraph(graph)

    for cfg_node_id in sliced_graph.nodes:
        if cfg_node_id not in reserve_nodes:
            remove_nodes.append(int(cfg_node_id))

    remove_nodes.sort(reverse=True)
    for remove_node in remove_nodes:
        sliced_graph = _remove_node(sliced_graph, str(remove_node))

    return sliced_graph


# 代码图表示构建器
class CodeGraphConstructor:
    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo):

        self.function_info = function_info
        self.contract_info = contract_info

        self.slice_graphs = {}
        self.external_node_id = {}

    def _add_edges_for_graph(self, g, reserved_nodes):

        new_edges = []
        semantic_edges = self.function_info.semantic_edges
        for semantic_type in semantic_edges:
            for edge in semantic_edges[semantic_type]:
                if str(edge[0]) in reserved_nodes and str(edge[1]) in reserved_nodes:
                    new_edges.append(edge)

        g.add_edges_from(new_edges)

        first_node = None
        for node in g.nodes:
            first_node = node
            break

        return first_node, g

    def _get_slice_criterias(self, criteria):

        current_criteria_set = [criteria]

        criterias_append = self.function_info.append_criterias
        msg_value_stmts = self.function_info.criterias_msg

        # 交易相关全局变量语义补充
        if criteria in criterias_append:
            for append_criteria in criterias_append[criteria]:
                current_criteria_set += append_criteria

        # 保留使用msg.value的语句
        for msg_value_stmt in msg_value_stmts:
            current_criteria_set.append(msg_value_stmt)

        return current_criteria_set

    def reserved_nodes_for_a_criteria(self, criteria):

        criteria_set = self._get_slice_criterias(criteria)
        print("切片准则：{}".format(criteria_set))

        # 针对每个切片准则进行前向依赖分析
        reserved_nodes = {}
        pdg = self.function_info.pdg
        for criteria_stmt in criteria_set:
            criteria_reserved_nodes = forward_dependence_analyze(pdg, criteria_stmt)
            for reserved_node in criteria_reserved_nodes:
                if reserved_node not in reserved_nodes:
                    reserved_nodes[reserved_node] = 1

        # 循环体结构保留
        loop_stmts = self.function_info.loop_stmts
        for loop_struct in loop_stmts:
            loop_from = loop_struct["from"]
            loop_to = loop_struct["to"]
            if loop_from in reserved_nodes and loop_to not in reserved_nodes:
                print("保存loop {}-{}".format(loop_from, loop_to))
                reserved_nodes[loop_to] = 1

        return reserved_nodes

    def _add_external_nodes(self, sliced_pdg, criteria):

        external_id = 0
        first_id = current_id = previous_id = None

        # 外部节点来源1：const_init
        const_init = self.function_info.const_var_init
        for const_var in const_init:

            new_id = "{}@{}".format(str(external_id), "tag")
            external_id += 1

            if previous_id is None:
                first_id = new_id
                previous_id = new_id
                current_id = new_id
            else:
                previous_id = current_id
                current_id = new_id

            sliced_pdg.add_node(new_id,
                                label=const_init[const_var],
                                expression=const_init[const_var],
                                type=const_init[const_var])

            if previous_id != current_id:
                sliced_pdg.add_edge(previous_id, current_id, color="black")

        external_state_map = self.function_info.external_state_def_nodes_map
        if criteria in external_state_map:
            for external_node in reversed(external_state_map[criteria]):

                if "expand" in external_node:
                    for expand_stmt in external_node["expand"]:

                        new_id = "{}@{}".format(str(external_id), "tag")
                        external_id += 1

                        if previous_id is None:
                            first_id = new_id
                            previous_id = new_id
                            current_id = new_id
                        else:
                            previous_id = current_id
                            current_id = new_id

                        sliced_pdg.add_node(new_id,
                                            label=expand_stmt,
                                            expression=expand_stmt,
                                            type=external_node["type"])

                        if previous_id != current_id:
                            sliced_pdg.add_edge(previous_id, current_id, color="black")
                else:
                    # print("外部节点: {}".format(external_node))
                    external_id += 1
                    new_id = "{}@{}".format(str(external_id), "tag")

                    if previous_id is None:
                        first_id = new_id
                        previous_id = new_id
                        current_id = new_id
                    else:
                        previous_id = current_id
                        current_id = new_id

                    sliced_pdg.add_node(new_id,
                                        label=external_node["expression"],
                                        expression=external_node["expression"],
                                        type=external_node["type"])

                    if previous_id != current_id:
                        sliced_pdg.add_edge(previous_id, current_id, color="black")

        return first_id, current_id

    def _add_reenter_edges(self, sliced_pdg, first_id):

        # 获得切片之后的控制流图 sliced_cfg
        sliced_cfg = nx.MultiDiGraph(sliced_pdg)
        sliced_cfg.graph["name"] = sliced_pdg.graph["name"]
        for u, v, k, d in sliced_pdg.edges(data=True, keys=True):
            if "type" in d:
                sliced_cfg.remove_edge(u, v, k)

        stmts_var_info_maps = self.function_info.stmts_var_info_maps
        for node_id in sliced_cfg.nodes:
            if sliced_cfg.out_degree(node_id) == 0:  # 叶子节点列表

                # print("from {} to {}".format(node_id, first_id))
                # 所有的叶子节点 --> 函数本身的 entry point
                sliced_pdg.add_edge(node_id, first_id, color="yellow", label="re_enter")

                if node_id not in stmts_var_info_maps or first_id not in stmts_var_info_maps:
                    # 新增语句 缺少数据流信息，暂不进行分析
                    continue

                # 由于新加了re_enter边，需要分析 node_id 和 first_id之间是否存在数据依赖
                # node_id -> first_id 需要分析 first_id 使用的数据是否依赖于 node_id
                previous_def = {}
                stmt_var_infos_def = stmts_var_info_maps[node_id]
                for var_info in stmt_var_infos_def:
                    if var_info["op_type"] == "def":
                        for var in var_info["list"]:
                            previous_def[var] = 1

                stmt_var_infos_use = stmts_var_info_maps[first_id]
                for var_info in stmt_var_infos_use:
                    if var_info["op_type"] == "use":
                        for var in var_info["list"]:
                            if var in previous_def:
                                sliced_pdg.add_edge(first_id, node_id, color="green", type="data_dependency")

    def do_code_slice_by_function_criterias(self):

        # 切片之前的准备工作
        self.function_info.get_all_criterias()
        if self.function_info.pdg is None:
            self.function_info.construct_dependency_graph()

        if self.function_info.pdg is None:
            raise RuntimeError("please construct the pdg before do slice")

        for criteria in self.function_info.criterias:
            # 计算需要保留的节点
            reserved_nodes = self.reserved_nodes_for_a_criteria(criteria)

            # 在原始CFG中去除其它节点
            sliced_cfg = do_slice(self.function_info.cfg, reserved_nodes)

            # 为切片后的cfg添加语义边，构成切片后的属性图
            first_node, sliced_pdg = self._add_edges_for_graph(sliced_cfg, reserved_nodes)

            # 保存
            self.slice_graphs[criteria] = sliced_pdg
            self.function_info.sliced_pdg[criteria] = sliced_pdg

            # TODO 外部节点
            new_first_id, external_last_id = self._add_external_nodes(sliced_pdg, criteria)
            if external_last_id is not None:
                sliced_pdg.add_edge(external_last_id, first_node, color="black")

            # reentry edge 重入边，保存一个函数可以执行多次的语义
            if new_first_id is not None:
                self._add_reenter_edges(sliced_pdg, new_first_id)
            else:
                self._add_reenter_edges(sliced_pdg, first_node)

            # 保存为json格式
            graph_info, file_name = save_graph_to_json_format(sliced_cfg, criteria)
            with open(file_name, "w+") as f:
                f.write(json.dumps(graph_info))
