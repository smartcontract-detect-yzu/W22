import itertools
import os
import subprocess
import json
from typing import Dict
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot

from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from queue import LifoQueue


def graph_clean_up(graph: nx.DiGraph):
    # 孤儿节点删除
    for node_id in graph.nodes:
        if graph.in_degree(node_id) == 0 and graph.out_degree(node_id) == 0:
            graph.remove_node(node_id)

    # TODO: BUG: entry-->entry 循环的bug暂时规避
    to_remove_edges = []
    for u, v in graph.edges():

        if u == v:
            label = graph.nodes[u]["label"]
            if label == "ENTRY" or label == "EXIT":
                to_remove_edges.append((u,v))
                # print("u:{} v:{} label:{}".format(u, v, label))

    graph.remove_edges_from(to_remove_edges)

def add_graph_node_info(graph: nx.MultiDiGraph, idx, key, value):
    node = graph.nodes[idx]
    if key not in node:
        node[key] = value
    else:
        raise RuntimeError("所添加的 key 重复了")


def _get_cfg_from_pdg(pdg: nx.MultiDiGraph):
    # 获得切片之后的控制流图 sliced_cfg
    cfg = nx.MultiDiGraph(pdg)
    cfg.graph["name"] = cfg.graph["name"]
    for u, v, k, d in pdg.edges(data=True, keys=True):
        if "type" in d:
            cfg.remove_edge(u, v, k)
    return cfg


def _add_entry_point_for_graph(graph: nx.MultiDiGraph):
    entry_points = []

    name = graph.graph["name"]
    for node_id in graph.nodes:
        if graph.in_degree(node_id) == 0:  # 入口节点
            entry_points.append(node_id)
    graph.add_node("{}@entry".format(name), label="ENTRY")

    for entry_point in entry_points:
        graph.add_edge("{}@entry".format(name), entry_point)

    return graph


def _add_exit_point_for_graph(graph: nx.MultiDiGraph):
    exit_points = []

    name = graph.graph["name"]
    for node_id in graph.nodes:
        if graph.out_degree(node_id) == 0:  # 出口节点
            exit_points.append(node_id)
    graph.add_node("{}@exit".format(name), label="EXIT")

    for exit_point in exit_points:
        graph.add_edge(exit_point, "{}@exit".format(name))

    return graph


def do_graph_relabel_before_merge(graph: nx.MultiDiGraph, prefix: str):
    """
    将两个图合并之前，需要先修改图的node id
    避免两个图的node id出现相同的情况
    """
    graph.graph["relabel"] = "{}@".format(prefix)
    return nx.relabel_nodes(graph, lambda x: "{}@{}".format(prefix, x))


def do_prepare_before_merge(graph: nx.MultiDiGraph, prefix: str):
    """
    将两个图合并之前，需要先修改图的node id
    避免两个图的node id出现相同的情况
    """
    graph.graph["relabel"] = "{}@".format(prefix)
    g = nx.relabel_nodes(graph, lambda x: "{}@{}".format(prefix, x))

    g = _get_cfg_from_pdg(g)  # 原始操作在CFG上完成
    g = _add_entry_point_for_graph(g)
    g = _add_exit_point_for_graph(g)

    return g


def do_merge_graph1_to_graph2(graph: nx.MultiDiGraph, to_graph: nx.MultiDiGraph, pos_at_to_graph):
    if "relabel" not in graph.graph or "relabel" not in to_graph.graph:
        raise RuntimeError("请先进行do_graph_relabel_before_merge，再进行merge操作")

    # 原始操作在CFG上完成
    g1 = _get_cfg_from_pdg(graph)
    g1 = _add_entry_point_for_graph(g1)
    g1 = _add_exit_point_for_graph(g1)
    debug_get_graph_png(g1, "g1", dot=True)

    g2 = to_graph
    debug_get_graph_png(g2, "g2", dot=True)

    sources = []
    for source, _ in g2.in_edges(pos_at_to_graph):
        sources.append(source)

    targets = []
    for _, target in g2.out_edges(pos_at_to_graph):
        targets.append(target)

    name = g1.graph["name"]
    to_name = g2.graph["name"]

    joint_graph: nx.MultiDiGraph = nx.union(g1, g2)
    joint_graph.graph["name"] = "{}@{}@{}_".format(g2.graph["name"], g1.graph["name"], pos_at_to_graph)
    # joint_graph.graph["name"] = "{}".format(g2.graph["name"])
    joint_graph.graph["relabel"] = "{}_{}".format(g1.graph["relabel"], g2.graph["relabel"])

    for src in sources:
        joint_graph.add_edge(src, "{}@entry".format(name))

    for target in targets:
        joint_graph.add_edge("{}@exit".format(name), target)

    joint_graph.remove_node(pos_at_to_graph)

    # 规避 https://github.com/smartcontract-detect-yzu/slither/issues/9
    graph_clean_up(joint_graph)

    return joint_graph


def debug_get_graph_png(graph: nx.Graph, postfix, dot=False):
    dot_name = "{}_{}.dot".format(graph.graph["name"], postfix)
    cfg_name = "{}_{}.png".format(graph.graph["name"], postfix)
    nx_dot.write_dot(graph, dot_name)
    subprocess.check_call(["dot", "-Tpng", dot_name, "-o", cfg_name])
    if dot is False:
        os.remove(dot_name)


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


# 基于PDG的前向依赖分析
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
    input_tmp_prefix = 10000
    input_temp_map = {}
    sliced_graph = nx.MultiDiGraph(graph)

    for cfg_node_id in sliced_graph.nodes:
        if cfg_node_id not in reserve_nodes:
            if "input_" not in cfg_node_id:

                remove_nodes.append(int(cfg_node_id))
            else:

                # 入参节点的ID是 input_开头，无法进行排序，此处进行特殊处理
                # 可以先删除，其出度入度都不大
                input_tmp_prefix += 1
                input_temp_map[input_tmp_prefix] = cfg_node_id
                remove_nodes.append(input_tmp_prefix)

    # 加速策略：优先删除id较大节点（叶子节点）
    # TODO：其实应该按照出度+入度的大小排列
    remove_nodes.sort(reverse=True)
    for remove_node in remove_nodes:

        # 还原
        if remove_node in input_temp_map:
            remove_node = input_temp_map[remove_node]

        sliced_graph = _remove_node(sliced_graph, str(remove_node))

    return sliced_graph


# 代码图表示构建器
class CodeGraphConstructor:
    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo):

        self.function_info = function_info
        self.contract_info = contract_info

        self.slice_graphs = {}
        self.external_node_id = {}

        # <criteria, graphs>
        self.external_slice_graphs: Dict[int, nx.MultiDiGraph] = {}

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

    def _expand_criteria_by_semantic(self, criteria):

        """
        根据切片准则进行语义增强
        保留更多的切片准则
        """

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

    def reserved_nodes_for_a_criteria(self, criteria, criteria_type="all"):

        # 首先根据切片类型判断使用需要对切片准则进行语义增强
        if criteria_type is "all":
            criteria_set = self._expand_criteria_by_semantic(criteria)
        else:
            criteria_set = [criteria]
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

    def _const_var_filter_by_sliced_graph(self, sliced_pdg):

        candidate_const_var = []

        const_init = self.function_info.const_var_init
        for graph_node in sliced_pdg.nodes:

            if "input_" in graph_node:
                continue

            var_infos = self.function_info.stmts_var_info_maps[str(graph_node)]
            for var_info in var_infos:
                if "list" in var_info:
                    for var in var_info["list"]:
                        if str(var) in const_init:
                            candidate_const_var.append(str(var))

        return candidate_const_var

    def _add_external_nodes(self, sliced_pdg, criteria):

        """
        外部节点来源：
        1.  self.function_info.const_var_init --> 常数的定义
        2.  self.function_info.external_state_def_nodes_map --> 交易涉及全局变量的外部修改
        """

        external_id = 0
        first_id = current_id = previous_id = None

        # 外部节点来源1：const_init

        const_init = self.function_info.const_var_init
        candidate_const_var = self._const_var_filter_by_sliced_graph(sliced_pdg)  # Note: 需要判断当前图表示经过切片后剩余的节点究竟涉及那些常数
        for const_var in candidate_const_var:

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
                sliced_pdg.add_edge(node_id, first_id, color="yellow", label="re_enter", type="re_enter")

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

    def do_code_slice_by_criterias_type(self, criteria_type="external"):

        criterias = self.function_info.get_criteria_by_type(criteria_type)
        for criteria in criterias:
            # 计算需要保留的节点
            reserved_nodes = self.reserved_nodes_for_a_criteria(criteria, criteria_type="external")

            # 在原始CFG中去除其它节点
            sliced_cfg = do_slice(self.function_info.cfg, reserved_nodes)

            # 为切片后的cfg添加语义边，构成切片后的属性图
            first_node, sliced_pdg = self._add_edges_for_graph(sliced_cfg, reserved_nodes)

            debug_get_graph_png(sliced_pdg, "external_{}".format(criteria))

            # 保存到当前的图构建器中
            self.external_slice_graphs[criteria] = sliced_pdg

        return self.external_slice_graphs

    def do_code_slice_by_internal_all_criterias(self):
        """

        """

        # 切片之前的准备工作
        self.function_info.get_all_internal_criterias()

        if self.function_info.pdg is None:
            self.function_info.construct_dependency_graph()

        if self.function_info.pdg is None:
            raise RuntimeError("please construct the pdg before do slice")

        for criteria in self.function_info.criterias:

            # 计算需要保留的节点
            reserved_nodes = self.reserved_nodes_for_a_criteria(criteria, criteria_type="all")

            # 在原始CFG中去除其它节点
            sliced_cfg = do_slice(self.function_info.cfg, reserved_nodes)

            # 为切片后的cfg添加语义边，构成切片后的属性图
            first_node, sliced_pdg = self._add_edges_for_graph(sliced_cfg, reserved_nodes)

            # 保存
            self.slice_graphs[criteria] = sliced_pdg
            self.function_info.sliced_pdg[criteria] = sliced_pdg

            # 入函数间分析器池

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

    def do_code_create_without_slice(self):

        return self.function_info.cfg
