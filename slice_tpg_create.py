import argparse
import itertools
import json

import shutil
import subprocess
from queue import LifoQueue

import os
from slither.core.expressions import Identifier

from slither import Slither
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot
from slither.core.cfg.node import NodeType
from slither.core.cfg.node import Node as Slither_Node
from slither.core.declarations.function import Function

EXAMPLE_PERFIX = "examples/ponzi/"
EXAMPLE_ANALYZE_PERFIX = "examples/analyze_test/"

DATASET_PERFIX = "examples/ponzi_dataset/"
ANALYZE_PERFIX = "examples/analyze/"

SAD_CHAIN_DATASET_PERFIX = "examples/ponzi_dataset_sad/chain/"
SAD_CHAIN_ANALYZE_PERFIX = "examples/analyze_sad/chain/"

BUYPOOL_DATASET_PERFIX = "examples/keys_dataset/key_buypool/"
BUYPOOL_ANALYZE_PERFIX = "examples/keys_analyze/key_buypool/"

DEPOSIT_DATASET_PERFIX = "examples/keys_dataset/key_deposit/"
DEPOSIT_ANALYZE_PERFIX = "examples/keys_analyze/key_deposit/"

SAD_TREE_DATASET_PERFIX = "examples/ponzi_dataset_sad/tree/"
SAD_TREE_ANALYZE_PERFIX = "examples/analyze_sad/tree/"

DEBUG_PNG = 0
PRINT_PNG = 1
VERSION = 2
ONLY_CFG_FLAG = 0

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

        if version_resault == None:
            version_resault = "0.4.26"
        return version_resault


def argParse():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('-t', type=str, default=None)
    parser.add_argument('-n', type=str, default=None)

    args = parser.parse_args()
    return args.t, args.n

    # return args.f, args.cfg, args.var, args.stmts


def debug_get_ddg_and_cdg(cfg, cdg_edges, ddg_edges, data_flows, ddg_edges_loop):
    debug = 0

    pdg = nx.MultiDiGraph(cfg)
    pdg.add_edges_from(cdg_edges)
    pdg.add_edges_from(ddg_edges)
    pdg.add_edges_from(data_flows)
    pdg.add_edges_from(ddg_edges_loop)
    debug_get_graph_png(pdg, "pdg", cur_dir=True)

    if debug:
        # 生成控制依赖图
        cdg = nx.DiGraph()
        cdg.add_edges_from(cdg_edges)
        debug_get_graph_png(cdg, "cdg")

        # 生成数据依赖图
        ddg = nx.DiGraph()
        ddg.add_edges_from(ddg_edges)
        debug_get_graph_png(ddg, "ddg")

        # 数据依赖图loop增强版本
        ddg.add_edges_from(ddg_edges_loop)
        debug_get_graph_png(ddg, "ddg_loop")

        # 数据流
        ddg_f = nx.MultiDiGraph(cfg)
        ddg_f.add_edges_from(ddg_edges)
        ddg_f.add_edges_from(data_flows)
        debug_get_graph_png(ddg_f, "ddg_f")


def _debug_stat_var_info(state_var_read_function_map, state_var_write_function_map, state_var_declare_function_map):
    if DEBUG_PNG == 0:
        return

    print(u"===全局变量定义信息：")
    for var in state_var_declare_function_map:
        print("\t定义变量{}".format(str(var)))

        if "exp" in state_var_declare_function_map[var]:
            print("\t\t{}".format(state_var_declare_function_map[var]["exp"]))

        if "fun" in state_var_declare_function_map[var]:
            print("\t\t{}".format(state_var_declare_function_map[var]["full_expr"]))

    print("===全局变量读信息：")
    for var in state_var_read_function_map:

        print("读变量{}".format(str(var)))
        for func in state_var_read_function_map[var]:
            print("\t{}".format(func.name))

    print("===全局变量写信息：")
    for var in state_var_write_function_map:

        print("写变量{}".format(str(var)))
        for func in state_var_write_function_map[var]:
            print("\t{}".format(func.name))


def debug_get_graph_png(graph: nx.Graph, postfix, cur_dir):
    if PRINT_PNG == 0:
        return

    if not cur_dir:
        dot_name = EXAMPLE_PERFIX + "{}_{}.dot".format(graph.graph["name"], postfix)
        nx_dot.write_dot(graph, dot_name)
        cfg_name = EXAMPLE_PERFIX + "{}_{}.png".format(graph.graph["name"], postfix)
    else:
        dot_name = "{}_{}.dot".format(graph.graph["name"], postfix)
        cfg_name = "{}_{}.png".format(graph.graph["name"], postfix)
        nx_dot.write_dot(graph, dot_name)

    subprocess.check_call(["dot", "-Tpng", dot_name, "-o", cfg_name])
    os.remove(dot_name)


def _get_png(target_fun):
    cfg_dot_file = "{}_cfg.dot".format(target_fun.name)
    cfg_png = "{}_cfg.png".format(target_fun.name)
    target_fun.cfg_to_dot(cfg_dot_file)
    subprocess.check_call(["dot", "-Tpng", cfg_dot_file, "-o", cfg_png])

    # dom_tree_dot_file = EXAMPLE_PERFIX + "{}_dom.dot".format(target_fun.name)
    # dom_png = EXAMPLE_PERFIX + "{}_dom.png".format(target_fun.name)
    # target_fun.dominator_tree_to_dot(dom_tree_dot_file)
    # subprocess.check_call(["dot", "-Tpng", dom_tree_dot_file, "-o", dom_png])

    return cfg_dot_file


def _recheck_vars_in_expression(stmt_expression, vars):
    """
    规避SSA数据分析的bug
    利用字符串匹配的方式重新计算变量是否真的在语句中

    入参1：当前语句
    入参2：slither解析出的当前语句使用变量
    """
    ret_vars = []
    miss_vars = []
    for var in vars:
        if var in stmt_expression:

            ret_vars.append(var)
        else:
            miss_vars.append(var)

    if len(miss_vars) != 0:
        print("\n\t==ERROR IN DATA DEF-USE==")
        print("\t\t语句：{}".format(stmt_expression))
        print("\t\t变量：{}".format(miss_vars))
        print("\t==ERROR IN DATA DEF-USE==\n")

    return ret_vars


def get_function_cfg(contract_name, function):
    cfg_dot_file = _get_png(function)
    cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
    os.remove(cfg_dot_file)
    cfg.graph["name"] = function.name
    cfg.graph["contract_name"] = contract_name

    for node in function.nodes:
        cfg_node = cfg.nodes[str(node.node_id)]
        cfg_node["expression"] = node.expression.__str__()
        cfg_node["type"] = node.type.__str__()
    return cfg


def _stmt_var_info(stmt_info, state_defs, const_var_init, state_var_declare_function_map):
    stmt_var_info = []

    expression = str(stmt_info.expression)

    no_write = no_read = 0
    if stmt_info.type == NodeType.IFLOOP:
        no_write = 1  # if语句不许写

    # 局部变量读 use
    read_local_vars = [str(var) for var in stmt_info.local_variables_read]
    if len(read_local_vars) != 0:
        rechecked_read_local_vars = _recheck_vars_in_expression(expression, read_local_vars)
        stmt_var_info.append({"list": rechecked_read_local_vars, "type": "local", "op_type": "use"})

    # 全局变量读 use
    read_state_vars = [str(var) for var in stmt_info.state_variables_read]
    if len(read_state_vars) != 0:
        rechecked_read_state_vars = _recheck_vars_in_expression(expression, read_state_vars)
        for var in rechecked_read_state_vars:
            if var in state_var_declare_function_map and "full_expr" in state_var_declare_function_map[var]:
                if var not in const_var_init:
                    const_var_init[var] = state_var_declare_function_map[var]["full_expr"]
        stmt_var_info.append({"list": rechecked_read_state_vars, "type": "state", "op_type": "use"})

    # 当前语句声明的变量
    if no_write == 0 and stmt_info.variable_declaration is not None:
        declare_vars = [str(stmt_info.variable_declaration)]
        rechecked_declare_var = _recheck_vars_in_expression(expression, declare_vars)
        for var in rechecked_declare_var:
            if var in state_var_declare_function_map and "full_expr" in state_var_declare_function_map[var]:
                if var not in const_var_init:
                    const_var_init[var] = state_var_declare_function_map[var]["full_expr"]
        stmt_var_info.append({"list": rechecked_declare_var, "type": "local", "op_type": "def"})

    # 当前语句局部变量写 def
    write_local_vars = [str(var) for var in stmt_info.local_variables_written]
    if no_write == 0 and len(write_local_vars) != 0:
        rechecked_write_local_vars = _recheck_vars_in_expression(expression, write_local_vars)
        stmt_var_info.append({"list": rechecked_write_local_vars, "type": "local", "op_type": "def"})

    # 全局变量写 def
    write_state_vars = [str(var) for var in stmt_info.state_variables_written]
    if no_write == 0 and len(write_state_vars) != 0:
        rechecked_write_state_vars = _recheck_vars_in_expression(expression, write_state_vars)
        for stat_var in rechecked_write_state_vars:

            if stat_var in state_var_declare_function_map and "full_expr" in state_var_declare_function_map[stat_var]:
                if stat_var not in const_var_init:
                    const_var_init[stat_var] = state_var_declare_function_map[stat_var]["full_expr"]

            if stat_var not in state_defs:
                state_defs[stat_var] = [stmt_info.node_id]
            else:
                state_defs[stat_var].append(stmt_info.node_id)
        stmt_var_info.append({"list": rechecked_write_state_vars, "type": "state", "op_type": "def"})

    return stmt_var_info


def _preprocess_for_dependency_analyze(or_cfg, function, state_var_declare_function_map):
    """
    语句预处理，提前有用的信息方便下一步的分析

    Parameters:
    cfg      - 当前函数的cfg
    function - 当前函数的slither分析结果

    Return：
    if_stmts # 条件语句列表
    stmts_var_info_maps # 各语句变量使用情况
    stmts_send_eth # 存在交易行为的语句列表
    stmts_loops # 循环语句列表
    if_paris # IF 与 END_IF
    node_id_2_id # node_id 与 cfg id
    state_defs # 全局变量定义
    const_var_init = {}  # 涉及的全局变量初始化语句
    msg_value_stmt = {}  # 使用了msg.value的语句
    """

    cfg = nx.DiGraph(or_cfg)
    stack = []
    remove_edges = []

    # 需要返回的变量
    if_stmts = []  # 条件语句列表
    stmts_var_info_maps = {}  # 各语句变量使用情况
    stmts_send_eth = {}  # 存在交易行为的语句列表
    stmts_loops = []  # 循环语句列表
    if_paris = {}  # IF 与 END_IF
    node_id_2_id = {}  # node_id 与 cfg id
    state_defs = {}  # 全局变量定义
    const_var_init = {}  # 涉及的全局变量初始化语句
    msg_value_stmt = {}  # 使用了msg.value的语句

    for id, stmt in enumerate(function.nodes):
        # 链表下标和节点ID是不同的
        node_id_2_id[stmt.node_id] = id

    for id, stmt in enumerate(function.nodes):

        # 语句的变量使用情况
        stmts_var_info_maps[str(stmt.node_id)] = _stmt_var_info(stmt, state_defs, const_var_init,
                                                                state_var_declare_function_map)

        if DEBUG_PNG == 1:
            print("语句：{}".format(stmt.expression))
            print("变量使用：{}".format(stmts_var_info_maps[str(stmt.node_id)]))
            print("============\n")

        if "msg.value" in stmt.expression.__str__():
            msg_value_stmt[str(stmt.node_id)] = {
                "exp": stmt.expression.__str__()
            }

        if stmt.can_send_eth():
            if ".transfer(" in str(stmt.expression):
                trans_info = str(stmt.expression).split(".transfer(")
                to = trans_info[0]
                eth = trans_info[1].strip(")")

            elif ".send(" in str(stmt.expression):
                trans_info = str(stmt.expression).split(".send(")
                to = trans_info[0]
                eth = trans_info[1].strip(")")
            else:
                eth = None
                to = None

            if eth is None and to is None:
                print("调用函数")
                pass
            else:  # 防止出现调用函数的情况
                stmts_send_eth[str(stmt.node_id)] = {
                    "to": to,
                    "eth": eth,
                    "exp": stmt.expression.__str__()
                }
                print("=== 切片准则：{} at {}@{} ===".format(stmt.expression, or_cfg.graph["name"], stmt.node_id))
                print("发送以太币 {} 到 {}\n".format(eth, to))
                print("变量使用: {}".format(stmts_var_info_maps[str(stmt.node_id)]))

        if stmt.type == NodeType.IF:
            stack.append(str(stmt.node_id))
            if_stmts.append(str(stmt.node_id))

        if stmt.type == NodeType.STARTLOOP:

            # begin_loop --> if_loop
            for suc_node_id in cfg.successors(str(stmt.node_id)):

                # NOTE 规避 node_id和function.nodes下标不一致的问题
                list_index = node_id_2_id[int(suc_node_id)]
                if function.nodes[list_index].type == NodeType.IFLOOP:
                    stack.append(str(suc_node_id))
                    if_stmts.append(str(suc_node_id))

        if stmt.type == NodeType.ENDIF or stmt.type == NodeType.ENDLOOP:
            if_start = stack.pop()
            if if_start not in if_paris:
                if_paris[if_start] = str(stmt.node_id)
            else:
                raise RuntimeError("IF END_IF 配对失败")

        if stmt.type == NodeType.IFLOOP:
            for pre_node_id in cfg.predecessors(str(stmt.node_id)):

                # IF_LOOP 的前驱节点中非 START_LOOP 的节点到IF_LOOP的边需要删除
                list_index = node_id_2_id[int(pre_node_id)]
                if function.nodes[list_index].type != NodeType.STARTLOOP:
                    remove_edges.append((pre_node_id, str(stmt.node_id)))

                    # 记录循环体的起止节点：循环执行的路径起止点
                    stmts_loops.append({"from": str(stmt.node_id), "to": pre_node_id})

    if len(remove_edges) != 0:
        cfg.remove_edges_from(remove_edges)

    # 2: 给CFG中的所有叶子节点添加exit子节点作为函数退出的标识符
    leaf_nodes = []
    for cfg_node_id in cfg.nodes:
        if cfg.out_degree(cfg_node_id) == 0:  # 叶子节点列表
            leaf_nodes.append(cfg_node_id)
    # debug_get_graph_png(cfg, "cfg_noloop")

    cfg.add_node("EXIT_POINT", label="EXIT_POINT")
    for leaf_node in leaf_nodes:
        cfg.add_edge(leaf_node, "EXIT_POINT")

    # debug_get_graph_png(cfg, "cfg_exit")

    return cfg, if_stmts, stmts_var_info_maps, stmts_send_eth, stmts_loops, \
           if_paris, node_id_2_id, state_defs, const_var_init, msg_value_stmt


def _get_control_dependency_relations(cfg, if_stmts, predom_relations, function, node_id_2_id):
    """
    根据控制流和前向支配计算当前函数的控制依赖关系
    Y is control dependent on X
    ⇔
    there is a path in the CFG from X to Y
    that doesn’t contain the immediate forward dominator of X
    其中x代表了分支控制语句：if or if_loop
    y代表x的所有后继节点

    c-->b且c-->a,基于就近原则选择依赖点既c-->b
    if(a){
        if(b){
            c
        }
    }
    """
    control_dep_edges = {}
    control_dep_relations = []
    for x in if_stmts:
        predom_node = predom_relations[x]
        cfg_paths = nx.all_simple_paths(cfg, source=x, target="EXIT_POINT")

        for cfg_path in list(cfg_paths):

            for y in cfg_path[1:-1]:
                list_index = node_id_2_id[int(y)]
                node_info = function.nodes[list_index]

                # 虚拟节点暂时不进行控制依赖分析
                if node_info.type != NodeType.ENDIF \
                        and node_info.type != NodeType.ENDLOOP \
                        and node_info.type != NodeType.STARTLOOP:
                    if y != predom_node:  # does’t contain the immediate forward dominator
                        key = "{}-{}".format(y, x)  # y控制依赖于x
                        if key not in control_dep_edges:
                            control_dep_edges[key] = 1
                            length = nx.shortest_path_length(cfg, x, y)
                            control_dep_relations.append({'from': y, "to": x, 'color': 'red', 'distance': length})

    control_dep_edges.clear()
    for cdg_edge in control_dep_relations:
        from_node = cdg_edge["from"]
        to_node = cdg_edge["to"]
        distance = cdg_edge["distance"]

        # NOTE: 就近原则，永远依赖于较近的那个
        if from_node not in control_dep_edges:
            control_dep_edges[from_node] = {"to": to_node, "distance": distance}
        else:
            old_distance = control_dep_edges[from_node]["distance"]
            if old_distance > distance:
                control_dep_edges[from_node] = {"to": to_node, "distance": distance}

    return control_dep_edges


def if_exit_fliter(simple_cfg, from_node, to_node, if_paris):
    """
    过滤掉终止的控制流依赖
    if(a) -- A -- end_if -- B
    此时 B 并不依赖于 if(a)
    """

    path_cnt = end_if_cnt = 0
    if to_node in if_paris:
        end_if = if_paris[to_node]

        # NOTE: 控制依赖关系和cfg关系方向相反
        cfg_paths = nx.all_simple_paths(simple_cfg, source=to_node, target=from_node)
        for cfg_path in cfg_paths:
            path_cnt += 1
            for node in cfg_path:
                if node == end_if:
                    end_if_cnt += 1
                    break

        if path_cnt == end_if_cnt:
            return True  # 每条路径都经过 END_IF, 则控制依赖就不存在
        else:
            return False
    else:
        # 有可能没有配对
        return False


def get_control_dependency_relations(simple_cfg, if_stmts, function, if_paris, node_id_2_id):
    """
    利用前向支配关系生成控制流依赖
    1.生成前向支配关系
    2.控制流依赖关系计算
    """

    # 前向支配关系生成，reverse_cfg的入口为 "EXIT_POINT"
    reverse_cfg = simple_cfg.reverse()
    predom_relations = nx.algorithms.immediate_dominators(reverse_cfg, "EXIT_POINT")
    del predom_relations["EXIT_POINT"]  # 删除EXIT_POINT，因为这是虚拟节点

    # 控制流依赖关系计算 <key:from, value:{"to": to_node, "distance": distance}>
    control_dep_relations = _get_control_dependency_relations(simple_cfg, if_stmts, predom_relations, function,
                                                              node_id_2_id)

    cdg_edges = []
    for from_node in control_dep_relations:
        to_node = control_dep_relations[from_node]["to"]

        # NOTE: 控制依赖边方向和控制流边方向相反
        if not if_exit_fliter(simple_cfg, from_node, to_node, if_paris):
            cdg_edges.append((from_node, to_node, {'color': "red", "type": "ctrl_dependency"}))
        else:
            print("过滤控制流边{}-{}".format(from_node, to_node))
            pass
    return cdg_edges


def _get_ddg_edges(data_dependency_relations):
    duplicate = {}
    ddg_edges = []

    for edge_info in data_dependency_relations:

        if edge_info["from"] == edge_info["to"]:
            continue

        key = "{}-{}".format(edge_info["from"], edge_info["to"])
        if key not in duplicate:
            duplicate[key] = 1
            ddg_edges.append((edge_info["from"], edge_info["to"], {'color': "green", "type": "data_dependency"}))

    # print("DEBUG 数据依赖：", ddg_edges)
    return ddg_edges


def _get_data_dependency_relations_by_path(path, stmts_var_info_maps):
    var_def_use_chain = {}
    data_dependency_relations = []

    for stmt in path[:-1]:  # 去尾，避免 EXIT_POINT
        stmt_id = str(stmt)
        stmt_var_infos = stmts_var_info_maps[stmt_id]

        for var_info in stmt_var_infos:
            for var_name in var_info["list"]:
                info = {"id": stmt_id, "var_type": var_info["type"], "op_type": var_info["op_type"]}
                if var_name not in var_def_use_chain:
                    chain = [info]
                    var_def_use_chain[var_name] = chain
                else:
                    var_def_use_chain[var_name].append(info)

    # 计算当前执行路径下的def_use chain分析
    for var in var_def_use_chain:
        last_def = None

        chain = var_def_use_chain[var]
        for chain_node in chain:
            if chain_node["op_type"] == "def":
                last_def = chain_node["id"]
            else:
                if last_def is not None:
                    edge_info = {"from": chain_node["id"], "to": last_def}
                    data_dependency_relations.append(edge_info)

    return data_dependency_relations


def get_data_dependency_relations(cfg, stmts_var_info_maps):
    """
    数据依赖解析：获得 def-use chain
    """
    var_def_use_chain = {}
    data_dependency_relations = []

    cfg_paths = nx.all_simple_paths(cfg, source="0", target="EXIT_POINT")
    for path in cfg_paths:
        var_def_use_chain.clear()

        for stmt in path[:-1]:  # 去尾，避免EXIT_POINT
            stmt_id = str(stmt)
            stmt_var_infos = stmts_var_info_maps[stmt_id]

            for var_info in stmt_var_infos:
                for var_name in var_info["list"]:
                    info = {"id": stmt_id, "var_type": var_info["type"], "op_type": var_info["op_type"]}
                    if var_name not in var_def_use_chain:
                        chain = [info]
                        var_def_use_chain[var_name] = chain
                    else:
                        var_def_use_chain[var_name].append(info)

        # 计算当前执行路径下的def_use_chain分析
        for var in var_def_use_chain:
            last_def = None
            chain = var_def_use_chain[var]
            for chain_node in chain:

                if chain_node["op_type"] == "def":
                    last_def = chain_node["id"]
                else:
                    if last_def is not None:
                        edge_info = {"from": chain_node["id"], "to": last_def}
                        data_dependency_relations.append(edge_info)

    ddg_edges = _get_ddg_edges(data_dependency_relations)
    return ddg_edges


def get_data_dependency_relations_forloop(simple_cfg, stmts_var_info_maps, transaction_stmts, loop_stmts):
    duplicate = {}
    ddg_edges = []

    for criteria in transaction_stmts:

        loop_paths = loop_structure_extreact(simple_cfg, loop_stmts, criteria)
        for path in loop_paths:
            loop_data_deps = _get_data_dependency_relations_by_path(path, stmts_var_info_maps)

            for edge_info in loop_data_deps:

                if edge_info["from"] == edge_info["to"]:
                    continue

                key = "{}-{}".format(edge_info["from"], edge_info["to"])
                if key not in duplicate:
                    duplicate[key] = 1

                    print("LOOP DATA: {}-{}".format(edge_info["from"], edge_info["to"]))

                    ddg_edges.append(
                        (edge_info["from"], edge_info["to"], {'color': "green", "type": "data_dependency"})
                    )

    return ddg_edges


def _data_flow_reorder(current_stmt_vars):
    use_vars = []
    def_vars = []

    # issue2: 由于数据流分析是逆向分析，导致变量关系需要先分析def 再分析use
    for var_info in current_stmt_vars:
        if var_info["op_type"] == "use":
            use_vars.append(var_info)
        elif var_info["op_type"] == "def":
            def_vars.append(var_info)

    return def_vars + use_vars


def data_flow_analyze(cfg, stmts_var_info_maps):
    """
    交易行为相关数据流分析
    transaction：<to, money>
    反向分析控制流，得到数据流
    """
    data_flow_edges = []
    data_flow_map = {}
    duplicate = {}
    def_info = {}
    use_info = {}

    cfg_paths = nx.all_simple_paths(cfg, source="0", target="EXIT_POINT")
    for cfg_path in cfg_paths:
        def_info.clear()
        use_info.clear()

        # [{"list": rechecked_read_state_vars, "type": "state", "op_type": "use"}]
        for from_node in reversed(cfg_path):

            if from_node == 'EXIT_POINT':
                continue

            _current_stmt_vars = stmts_var_info_maps[from_node]
            current_stmt_vars = _data_flow_reorder(_current_stmt_vars)
            for var_info in current_stmt_vars:

                # 如果当前语句有写操作，查询之前语句对该变量是否有读操作
                # 写变量：该变量出读操作栈
                if var_info["op_type"] == "def":
                    for var in var_info["list"]:

                        if var in use_info:
                            for to_node in use_info[var]:

                                if from_node == to_node:
                                    continue

                                key = "{}-{}".format(from_node, to_node)
                                if key not in duplicate:
                                    duplicate[key] = 1
                                    if to_node not in data_flow_map:
                                        data_flow_map[to_node] = [from_node]
                                    else:
                                        data_flow_map[to_node].append(from_node)

                                    data_flow_edges.append(
                                        (from_node, to_node, {'color': "blue", "type": "data_flow"}))

                            # kill：中断上一次def的flow 出读操作栈
                            del use_info[var]

                        def_info[var] = from_node

                # 读变量：压栈
                if var_info["op_type"] == "use":
                    for var in var_info["list"]:

                        if var not in use_info:
                            use_info[var] = [from_node]
                        else:
                            use_info[var].append(from_node)

    if DEBUG_PNG == 1:
        dfg = nx.MultiDiGraph(cfg)
        dfg.add_edges_from(data_flow_edges)
        debug_get_graph_png(dfg, "dfg", cur_dir=True)

    print("数据流:", data_flow_map)
    return data_flow_edges, data_flow_map


def _forward_analyze(cfg, criteria):
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
        for successor_stmt in cfg.successors(current_stmt):  # 数据依赖 + 控制依赖关系

            for edge_id in cfg[current_stmt][successor_stmt]:
                edge_data = cfg[current_stmt][successor_stmt][edge_id]
                # print("分析：{} -{}- {}".format(current_stmt, edge_data, successor_stmt))

                if "type" in edge_data:
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


def do_slice(cfg, reserve_nodes):
    remove_nodes = []
    sliced_cfg = nx.DiGraph(cfg)

    for cfg_node_id in sliced_cfg.nodes:
        if cfg_node_id not in reserve_nodes:
            remove_nodes.append(int(cfg_node_id))

    remove_nodes.sort(reverse=True)
    for remove_node in remove_nodes:
        sliced_cfg = _remove_node(sliced_cfg, str(remove_node))

    return sliced_cfg


def save_sliced_pdg_to_json(graph):
    graph_info = {}
    nodes = []
    cfg_edges = []
    cdg_edges = []
    ddg_edges = []
    dfg_edges = []
    cfg_to_graph_id = {}
    graph_id = 0
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

    return graph_info


def reserved_nodes_for_criteria(pdg, criteria, criterias_append, msg_value_stmts, loop_stmts):
    criteria_set = [criteria]

    # 交易相关全局变量语义补充
    if criteria in criterias_append:
        for append_criteria in criterias_append[criteria]:
            criteria_set += append_criteria

    # 保留使用msg.value的语句
    for msg_value_stmt in msg_value_stmts:
        criteria_set.append(msg_value_stmt)

    # 针对每个切片准则进行前向依赖分析
    reserved_nodes = {}
    print("切片准则：{}".format(criteria_set))
    for criteria_stmt in criteria_set:
        criteria_reserved_nodes = _forward_analyze(pdg, criteria_stmt)
        for reserved_node in criteria_reserved_nodes:
            if reserved_node not in reserved_nodes:
                reserved_nodes[reserved_node] = 1

    # 循环体保留
    for loop_struct in loop_stmts:
        loop_from = loop_struct["from"]
        loop_to = loop_struct["to"]
        if loop_from in reserved_nodes and loop_to not in reserved_nodes:
            print("保存loop {}-{}".format(loop_from, loop_to))
            reserved_nodes[loop_to] = 1

    return reserved_nodes


def external_graph_node(sliced_pdg, criteria, external_state_map):
    # 外部节点信息, 添加到cfg中
    external_id = 0
    current_id = None
    previous_id = None

    if criteria in external_state_map:
        for external_node in reversed(external_state_map[criteria]):

            # print("外部节点: {}".format(external_node))
            external_id += 1
            new_id = "{}@{}".format(str(external_id), "tag")

            if previous_id is None:
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

    return current_id


def external_struct_expand_graph_node(sliced_pdg, criteria, const_init, external_state_map):
    # 外部节点信息, 添加到cfg中
    external_id = 0
    first_id = None
    current_id = None
    previous_id = None

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


def add_reenter_edges(sliced_pdg, first_id, criteria, stmts_var_info_maps):
    sliced_cfg = nx.MultiDiGraph(sliced_pdg)
    sliced_cfg.graph["name"] = sliced_pdg.graph["name"]

    for u, v, k, d in sliced_pdg.edges(data=True, keys=True):
        if "type" in d:
            sliced_cfg.remove_edge(u, v, k)

    debug_get_graph_png(sliced_cfg, "sliced_cfg_{}".format(criteria), cur_dir=True)

    for node_id in sliced_cfg.nodes:
        if sliced_cfg.out_degree(node_id) == 0:  # 叶子节点列表

            # print("from {} to {}".format(node_id, first_id))
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


def program_slice(cfg, semantic_edges, loop_stmts, criterias, criterias_append, msg_value_stmts,
                  external_state_map, const_init, stmts_var_info_maps):
    pdg = nx.MultiDiGraph(cfg)

    for semantic_type in semantic_edges:
        if semantic_type == "ctrl_dep" \
                or semantic_type == "data_dep" \
                or semantic_type == "loop_data_dep":
            pdg.add_edges_from(semantic_edges[semantic_type])

    for criteria in criterias:
        reserved_nodes = reserved_nodes_for_criteria(pdg, criteria, criterias_append, msg_value_stmts, loop_stmts)
        sliced_cfg = do_slice(cfg, reserved_nodes)

        new_edges = []
        for semantic_type in semantic_edges:
            for edge in semantic_edges[semantic_type]:
                if str(edge[0]) in reserved_nodes and str(edge[1]) in reserved_nodes:
                    new_edges.append(edge)

        sliced_pdg = nx.MultiDiGraph(sliced_cfg)
        sliced_pdg.add_edges_from(new_edges)

        first_node = None
        for node in sliced_pdg.nodes:
            first_node = node
            break

        # 外部节点信息, 添加到cfg中: 是否包含了结构体展开操作
        new_first_id, external_last_id = external_struct_expand_graph_node(sliced_pdg, criteria, const_init,
                                                                           external_state_map)
        if external_last_id is not None:
            sliced_pdg.add_edge(external_last_id, first_node, color="black")

        # reentry edge 重入边，保存一个函数可以执行多次的语义
        if new_first_id is not None:
            add_reenter_edges(sliced_pdg, new_first_id, criteria, stmts_var_info_maps)
        else:
            add_reenter_edges(sliced_pdg, first_node, criteria, stmts_var_info_maps)

        # 保存为json格式
        graph_info = save_sliced_pdg_to_json(sliced_pdg)
        graph_json_file = "{}_{}_{}.json".format(cfg.graph["contract_name"], cfg.graph["name"], criteria)
        with open(graph_json_file, "w+") as f:
            f.write(json.dumps(graph_info))

        debug_get_graph_png(sliced_pdg, "ssliced_pdg_{}".format(criteria), cur_dir=True)


def loop_structure_extreact(simple_cfg, loop_structures, criteria):
    """
    循环体执行
    for(循环条件){
        A ; criteria ;B ;C ;D
    }
    存在执行路径 <criteria, B, C, D, 循环条件>, 需要分析该路径的数据依赖关系，而B C D会对criteria造成影响
    存在执行路径 <B, C, D, 循环条件, criteria>, 需要分析该路径的数据依赖关系，而B C D会对criteria造成影响
    """
    loop_reverse_paths = []
    for loop_structure in loop_structures:

        src = loop_structure["from"]
        dst = loop_structure["to"]

        cfg_paths = nx.all_simple_paths(simple_cfg, source=src, target=dst)
        for cfg_path in cfg_paths:

            for index, path_node in enumerate(cfg_path):
                if path_node == str(criteria):
                    loop_exe_path = cfg_path[index + 1:] + [path_node] + ["EXIT_POINT"]  # 将初始节点(切片准则)放在最后
                    loop_reverse_paths.append(loop_exe_path)
                    break

    return loop_reverse_paths


def transaction_data_flow_analyze(stmts_var_info_maps, data_flow_map, transaction_stmts):
    trans_stats = {}

    for trans_stmt in transaction_stmts:
        stack = [trans_stmt]
        trans_state_infos = []
        while len(stack) != 0:
            to_id = stack.pop()
            if to_id in data_flow_map:
                for from_id in data_flow_map[to_id]:

                    stmt_var_infos = stmts_var_info_maps[from_id]
                    for var_info in stmt_var_infos:
                        if var_info["op_type"] == "use" and var_info["type"] == "state":
                            trans_state_infos.append({"vars": var_info["list"], "stmt_id": from_id})

                    stack.append(from_id)

        trans_stats[trans_stmt] = trans_state_infos
        # print("DEBUG 切片准则：{}".format(trans_stmt))
        # print("DEBUG 涉及全局变量信息:{}".format(trans_stats[trans_stmt]))

    return trans_stats


def state_criteria_add(transaction_states, state_defs):
    criteria_append = {}
    dup = {}
    for transaction_stmt in transaction_states:
        dup.clear()
        trans_states = transaction_states[transaction_stmt]
        criteria_append[transaction_stmt] = []
        for state_info in trans_states:

            states = state_info["vars"]
            for state in states:
                if state in state_defs and state not in dup:
                    dup[state] = 1
                    criteria_append[transaction_stmt].append(state_defs[state])

    return criteria_append


def _new_struct_analyze(stmt, struct_name, structs_info):
    struct_end_pos = None
    new_struct_elems = []
    stack = []
    new_stmts = []

    expr = str(stmt.expression).rsplit(struct_name + "(")[1]
    expr = "(" + expr
    for index, char in enumerate(expr):
        if index == 0:
            if char != "(":
                print("\n[ERROR]语句不是左括号开头: {}".format(stmt.expression))
                print("\n[ERROR]语句不是左括号开头: {}".format(expr))
                raise RuntimeError("语句不是左括号开头")
            else:
                stack.append("(_first")
        elif char == "(":
            stack.append(char)
        elif char == ")":
            top_char = stack.pop()
            if top_char == "(_first":
                struct_end_pos = index
                break

    _struct_expr = expr[1:struct_end_pos].split(",")
    left_expr = str(stmt.expression).split(struct_name)[0]
    left_expr = "{}{}{}".format(left_expr, struct_name, expr[struct_end_pos + 1:])

    struct_info = structs_info[struct_name]
    for i, elem in enumerate(struct_info.elems_ordered):
        elem_type = elem.type.__str__()
        elem_name = elem.name.__str__()
        elem_content = _struct_expr[i].__str__()

        if elem_content.startswith(elem_type):
            elem_content = elem_content.strip("{}(".format(elem_type))[:-1]

        elem_content = "{}.{} = {}".format(struct_name, elem_name, elem_content)
        new_stmts.append(elem_content)
        new_struct_elems.append(elem_content)

    new_stmts.append(left_expr)

    print("\n==原始语句:", stmt.expression.__str__())
    for new_stmt in new_stmts:
        print("\t\t", new_stmt)

    return new_struct_elems, left_expr, new_stmts


def _new_struct(node, structs_info):
    for ir in node.irs:

        if "new " in str(ir):  # 如果当前语句进行了结构体定义
            struct_name = str(ir).split("new ")[1].split("(")[0]
            if struct_name not in structs_info:
                # print("\n====异常结构体名称：", struct_name)
                # print(node.expression.__str__())
                # print(ir)
                # print("当前结构体：", [str(s_n) for s_n in structs_info])
                continue
            else:
                return struct_name
    return None


def struct_analyze(external_state_map, structs_info, const_var_init, state_var_declare_function_map):
    for stmt_id in external_state_map:

        stmts_array = external_state_map[stmt_id]
        for stmt_info in stmts_array:

            node = stmt_info['node']
            for v in node.state_variables_read:
                if str(v) in state_var_declare_function_map and "full_expr" in state_var_declare_function_map[str(v)]:
                    if str(v) not in const_var_init:
                        const_var_init[str(v)] = state_var_declare_function_map[str(v)]["full_expr"]

            struct_name = _new_struct(node, structs_info)
            if struct_name is not None:
                _, _, new_stmts = _new_struct_analyze(node, struct_name, structs_info)
                stmt_info["expand"] = new_stmts

    print("\n=======常数初始化:")
    for var in const_var_init:
        print("\t", const_var_init[var])


def _function_property(function):
    print("type {}".format(function.type.__str__()))
    print("function_type {}".format(function.function_type.__str__()))
    print("{}".format(function.external_calls_as_expressions))
    pass


def _debug_irs_for_stmt(node):
    if DEBUG_PNG == 0:
        return

    for ir in node.irs:
        print(str(ir))
    print("\r")
    for ir_ssa in node.irs_ssa:
        print(str(ir_ssa))

    print("\n\tFUNC_EXP:{}".format(node.expression))
    for var in node.variables_written:
        print("\tWRITE:{} {}".format(var.type, var.name))
    for var in node.variables_read:
        print("\tREAD:{} {}".format(var.type, var.name))


def interprocedural_state_analyze(this_function,
                                  transaction_states,
                                  state_var_write_function_map,
                                  state_var_declare_function_map):
    """
    Parameters:
    this_function -- 当前函数的信息
    transaction_states -- 当前函数依赖中交易行为依赖的全局变量
    state_var_write_function_map -- 当前智能合约中全局变量修改操作和函数的对应表
    state_var_declare_function_map -- 当前智能合约中全局变量声明操作和函数的对应表

    Return：
    external_nodes_map - <交易相关全局变量, [交易相关全局变量修改函数]>
    """

    # 跨函数交易依赖全局变量修改分析
    external_nodes_map = {}
    duplicate = {}
    for trans_criteria in transaction_states:
        external_nodes_map[trans_criteria] = []
        for transaction_state in transaction_states[trans_criteria]:
            for trans_stat_var in transaction_state["vars"]:
                duplicate.clear()
                stack = [trans_stat_var]
                duplicate[trans_stat_var] = 1
                while len(stack) != 0:

                    current_var = stack.pop()
                    state_defs_t = {}
                    const_var_init_t = {}

                    if current_var in state_var_write_function_map:
                        write_funs = state_var_write_function_map[current_var]
                    elif current_var in state_var_declare_function_map:
                        if "exp" in state_var_declare_function_map[current_var]:

                            # print("\t\tDEC_EXP:{}".format(state_var_declare_function_map[current_var]["exp"]))
                            continue
                        elif "fun" in state_var_declare_function_map[current_var]:

                            write_funs = [state_var_declare_function_map[current_var]["fun"]]
                            # print("\t\tDEC_FUN:{}".format(state_var_declare_function_map[current_var]["fun"].name))
                        else:
                            raise RuntimeError("全局变量缺乏定义和修改")

                    else:
                        raise RuntimeError("全局变量缺乏定义和修改")

                    for write_fun in write_funs:

                        if write_fun.full_name == this_function.full_name:
                            continue  # 过滤非当前函数

                        # print("\t\tFun name is", write_fun.full_name)
                        for node in write_fun.nodes:
                            # print("\t\tFUNC_EXP:{}".format(node.expression))
                            # print("\t\tSTAT_VAR:{}\n".format([str(v) for v in node.state_variables_written]))
                            state_defs_t.clear()
                            for v in node.state_variables_written:

                                if current_var == str(v):
                                    var_infos = _stmt_var_info(node,
                                                               state_defs_t,
                                                               const_var_init_t,
                                                               state_var_declare_function_map)

                                    _debug_irs_for_stmt(node)

                                    node_expression = node.expression.__str__()  # 语句
                                    node_type = node.type.__str__()  # 类型

                                    external_nodes_map[trans_criteria].append({
                                        "state_var": current_var,
                                        "expression": node_expression,
                                        "type": node_type,
                                        "info": var_infos,
                                        "fun": write_fun,
                                        "func_name": write_fun.name,
                                        "node": node
                                    })

                                    for var_info in var_infos:
                                        if var_info["type"] == "state" and var_info["op_type"] == "use":
                                            for var in var_info["list"]:
                                                if var not in duplicate:
                                                    # print("\t\t下一个变量{}".format(var))
                                                    duplicate[var] = 1
                                                    stack.append(var)

    return external_nodes_map


def interprocedural_function_analyze(simple_cfg, function, called_by_callee):
    for node in function.nodes:
        key = "{}_{}".format(function.name.__str__(), node.node_id.__str__())
        if key in called_by_callee:
            called_name = called_by_callee[key]["name"]
            called_function = called_by_callee[key]["function"]


def _analyze_function(contract_name,
                      function, structs_info,
                      called_by_callee,
                      state_var_write_function_map,
                      state_var_declare_function_map):
    print("\n##################################")
    print("##### 合约名 {}  ##".format(contract_name))
    print("##### 函数名 {}  ##".format(function.name))
    print("####################################")

    # 语义边集合
    semantic_edges = {}

    # 获得控制流图
    cfg = get_function_cfg(contract_name, function)

    if ONLY_CFG_FLAG == 1:
        graph_info = save_sliced_pdg_to_json(cfg)  # 保存为json格式
        graph_json_file = "{}_{}_cfg.json".format(cfg.graph["contract_name"], cfg.graph["name"])
        with open(graph_json_file, "w+") as f:
            f.write(json.dumps(graph_info))
            return ["only cfg"]

    # 预处理
    simple_cfg, if_stmts, stmts_var_info_maps, transaction_stmts, \
    loop_stmts, if_paris, node_id_2_id, state_defs, const_var_init, msg_value_stmts \
        = _preprocess_for_dependency_analyze(cfg, function, state_var_declare_function_map)

    if len(transaction_stmts) == 0: return transaction_stmts

    #  TODO:函数过程间分析 interprocedural_function_analyze(simple_cfg, function, called_by_callee)

    debug_get_graph_png(simple_cfg, "simple_cfg", cur_dir=True)

    # 数据流分析
    data_flow_edges, data_flow_map = data_flow_analyze(simple_cfg, stmts_var_info_maps)
    semantic_edges["data_flow"] = data_flow_edges

    # 根据交易语句获得与交易相关的全局变量
    transaction_states = transaction_data_flow_analyze(stmts_var_info_maps, data_flow_map, transaction_stmts)

    # 每个交易语句相关的全局变量作为补充切片准则，补充交易语句的语义
    criteria_append = state_criteria_add(transaction_states, state_defs)

    # 函数间全局变量修改关系分析
    external_state_map = interprocedural_state_analyze(function,
                                                       transaction_states,
                                                       state_var_write_function_map,
                                                       state_var_declare_function_map)

    # 结构体展开, 常数定义
    const_var_init.clear()  # TODO: 有BUG 该功能暂时不能用
    struct_analyze(external_state_map, structs_info, const_var_init, state_var_declare_function_map)

    # 数据依赖生成
    ddg_edges = get_data_dependency_relations(simple_cfg, stmts_var_info_maps)
    semantic_edges["data_dep"] = ddg_edges

    # 控制依赖关系
    cdg_edges = get_control_dependency_relations(simple_cfg, if_stmts, function, if_paris, node_id_2_id)
    semantic_edges["ctrl_dep"] = cdg_edges

    # 循环体内部的数据流依赖
    loop_ddg_edges = get_data_dependency_relations_forloop(simple_cfg,
                                                           stmts_var_info_maps,
                                                           transaction_stmts,
                                                           loop_stmts)
    semantic_edges["loop_data_dep"] = loop_ddg_edges

    debug_get_ddg_and_cdg(cfg, cdg_edges, ddg_edges, data_flow_edges, loop_ddg_edges)

    # 程序切片
    program_slice(cfg, semantic_edges, loop_stmts, transaction_stmts, criteria_append, msg_value_stmts,
                  external_state_map, const_var_init, stmts_var_info_maps)

    return transaction_stmts


def state_vars_info(function,
                    state_var_declare_function_map,
                    state_var_read_function_map,
                    state_var_write_function_map):
    # 全局变量定义
    if function.is_constructor or function.is_constructor_variables:
        for node in function.nodes:
            for v in node.state_variables_written:
                full_exp = "{} {}".format(str(v.type), node.expression)
                state_var_declare_function_map[str(v)] = {
                    "fun": function,
                    "expr": node.expression,
                    "full_expr": full_exp
                }

    else:
        # 全局变量读
        for v in function.state_variables_read:
            if str(v) not in state_var_read_function_map:
                state_var_read_function_map[str(v)] = [function]
            else:
                state_var_read_function_map[str(v)].append(function)

        # 全局变量写
        for v in function.state_variables_written:

            if not function.can_send_eth():
                continue  # NOTE:对于参与交易的函数，下面会进行重点分析

            if str(v) not in state_var_write_function_map:
                state_var_write_function_map[str(v)] = [function]
            else:
                state_var_write_function_map[str(v)].append(function)


def stat_vars_delcare_without_assign(contract, state_var_declare_function_map):
    for v in contract.state_variables:
        if v.expression is None:
            exp = str(v.type) + " " + str(Identifier(v))
            state_var_declare_function_map[str(Identifier(v))] = {"type": str(v.type), "exp": exp}


def _preprocess_for_contract(contract):
    state_var_declare_function_map = {}  # <全局变量名称, slither.function>
    state_var_read_function_map = {}  # <全局变量名称, slither.function>
    state_var_write_function_map = {}  # <全局变量名称, slither.function>
    structs_info = {}  # <结构体名称, StructureContract>
    called_by_callee = {}  # <函数, <调用者, 调用位置>>

    # 结构体定义信息抽取
    for structure in contract.structures:
        print("结构体名称：{}".format(structure.name))
        structs_info[structure.name] = structure

    # 变量声明
    stat_vars_delcare_without_assign(contract,
                                     state_var_declare_function_map)

    call_chain = []

    # NOTE: 寻找真实调用<>.send<> 和 <>.transfer<>接口的函数
    send_function_map = {}
    for function in contract.functions:
        for node in function.nodes:
            if ".transfer(" in str(node.expression) or ".send(" in str(node.expression):
                if function.name not in send_function_map:
                    send_function_map[function.id] = {
                        "id": function.id,
                        "name": function.name,
                        "function": function,
                        "exp": node.expression,
                        "node": node
                    }

    # 调用图
    call_graph = nx.DiGraph()
    call_graph.graph["name"] = contract.name
    edges = []
    duplicate = {}
    fid_2_gid = {}
    node_id = 0
    for function in contract.functions:

        if function.id not in fid_2_gid:
            call_graph.add_node(node_id, label=function.name, fid=function.id)
            fid_2_gid[function.id] = node_id
            node_id += 1

        from_node = fid_2_gid[function.id]
        for internal_call in function.internal_calls:

            if isinstance(internal_call, Function):
                if internal_call.id not in fid_2_gid:
                    call_graph.add_node(node_id, label=internal_call.name, fid=internal_call.id)
                    fid_2_gid[internal_call.id] = node_id
                    node_id += 1

                to_node = fid_2_gid[internal_call.id]
                if "{}-{}".format(from_node, to_node) not in duplicate:
                    duplicate["{}-{}".format(from_node, to_node)] = 1
                    edges.append((from_node, to_node))
    call_graph.add_edges_from(edges)
    debug_get_graph_png(call_graph, "call_graph", cur_dir=True)

    entry_points = []
    for node_id in call_graph.nodes:
        if call_graph.in_degree(node_id) == 0:  # 入口节点
            entry_points.append(node_id)

    end_ponits = []
    for fid in send_function_map:
        end_ponits.append(fid_2_gid[fid])

    for src in entry_points:
        for dst in end_ponits:
            call_graph_paths = nx.all_simple_paths(call_graph, source=src, target=dst)
            for call_path in call_graph_paths:
                function_path = []
                for path_node in call_path:
                    function_path.append(call_graph.nodes[path_node]["label"])

                print("调用链: {}".format(function_path))


    call_chain = []
    # for entry_point in entry_points:
    #     for
    #     cfg_paths = nx.all_simple_paths(call_graph, source=x, target="EXIT_POINT")

    for function in contract.functions:
        # 全局变量：声明/读取/写
        state_vars_info(function,
                        state_var_declare_function_map,
                        state_var_read_function_map,
                        state_var_write_function_map)

    # 全局变量调试信息打印
    _debug_stat_var_info(state_var_read_function_map,
                         state_var_write_function_map,
                         state_var_declare_function_map)

    return state_var_declare_function_map, \
           state_var_read_function_map, \
           state_var_write_function_map, \
           structs_info, \
           called_by_callee


def analyze_contract(contract_file):
    slice_record = []
    slither = Slither(contract_file)

    for contract in slither.contracts:

        state_var_declare_function_map, \
        state_var_read_function_map, \
        state_var_write_function_map, \
        structs_info, \
        called_by_callee = _preprocess_for_contract(contract)

        for function in contract.functions:

            if function.can_send_eth():
                slices_tag = _analyze_function(contract.name,
                                               function,
                                               structs_info,
                                               called_by_callee,
                                               state_var_write_function_map,
                                               state_var_declare_function_map)
                if ONLY_CFG_FLAG == 0:
                    for slice_id in slices_tag:
                        exp = slices_tag[slice_id]["exp"]
                        name = "{}_{}_{}".format(contract.name, function.name, slice_id)
                        slice_record.append({"name": name, "exp": exp})

    for record in slice_record:
        print("{}".format(record))

    return slice_record


defined_target_list = [
    "sad_chain",
    "sad_tree",
    "xblock_dissecting",
    "buypool",
    "deposit",
    "buypool",
    "etherscan"
]


def _get_work_dir(target):
    if VERSION == 2:
        dataset_prefix = "examples/ponzi_src/{}/".format(target)
        analyze_prefix = "examples/ponzi_src/analyze/{}/".format(target)
        return dataset_prefix, analyze_prefix

    if target == "sad_chain":
        dataset_prefix = SAD_CHAIN_DATASET_PERFIX
        analyze_prefix = SAD_CHAIN_ANALYZE_PERFIX

    elif target == "sad_tree":
        dataset_prefix = SAD_TREE_DATASET_PERFIX
        analyze_prefix = SAD_TREE_ANALYZE_PERFIX

    elif target == "xblock":
        dataset_prefix = DATASET_PERFIX
        analyze_prefix = ANALYZE_PERFIX

    elif target == "buypool":
        dataset_prefix = BUYPOOL_DATASET_PERFIX
        analyze_prefix = BUYPOOL_ANALYZE_PERFIX

    elif target == "deposit":
        dataset_prefix = DEPOSIT_DATASET_PERFIX
        analyze_prefix = DEPOSIT_ANALYZE_PERFIX

    else:
        raise RuntimeError("-t 没有指定数据集")

    return dataset_prefix, analyze_prefix


def analyze_dataset(target):
    pwd = os.getcwd()
    slices_map = {}

    dataset_prefix, analyze_prefix = _get_work_dir(target)
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

                done_file = analyze_dir + "/done_ok.txt"
                pass_file = analyze_dir + "/pass.txt"
                pass_tag = 0

                if os.path.exists(pass_file):
                    continue

                if os.path.exists(done_file) and pass_tag:

                    print("========={}===========".format(file_name))
                    continue
                else:

                    # 首先删除 done_ok.txt 并 切换工作目录
                    if os.path.exists(done_file):
                        os.remove(done_file)

                    # 切换工作目录
                    os.chdir(analyze_dir)

                    # solc 版本解析
                    solc_version = parse_solc_version(file_name)
                    if solc_version == "0.4.6": solc_version = "0.4.26"
                    print("========={} V: {}".format(file_name, solc_version))
                    subprocess.check_call(["solc-select", "use", solc_version])

                    # 分析
                    slices = analyze_contract(file_name)
                    if address in slices_map:
                        raise RuntimeError("重复了")
                    else:
                        slices_map[address] = {
                            "addre": address,
                            "slice": slices
                        }

                    with open("done_ok.txt", "w+") as f:
                        f.write("done")
                    os.chdir(pwd)  # 还原工作目录

    with open("slice_record_{}.json".format(target), "w") as dump_f:
        json.dump(slices_map, dump_f)


if __name__ == '__main__':

    target, name = argParse()

    if name is not None:

        # 生成png文件
        PRINT_PNG = 1

        src_prex = "examples/ponzi_dataset/"
        test_path = "examples/ponzi/"

        for file in os.listdir(test_path):
            if not file.endswith(".sol") and not file == "ast":
                os.remove(os.path.join(test_path, file))

        if not os.path.exists(test_path + name):
            shutil.copy(src_prex + name, test_path)

        os.chdir(test_path)
        solc_version = parse_solc_version(name)
        print("========={} V: {}".format(name, solc_version))
        subprocess.check_call(["solc-select", "use", solc_version])
        slices_record = analyze_contract(name)

    else:

        # 不生成png文件
        PRINT_PNG = 0

        if target == "all":
            for defined_target in defined_target_list:
                analyze_dataset(defined_target)
        elif target == "all_cfg":
            ONLY_CFG_FLAG = 1
            for defined_target in defined_target_list:
                analyze_dataset(defined_target)
        else:
            analyze_dataset(target)

    # slither = Slither(EXAMPLE_PERFIX + '0x09515cb5e3acaef239ab83d78b2f3e3764fcab9b.sol')
    # slither = Slither(EXAMPLE_PERFIX + 'test.sol')
    # slither = Slither(EXAMPLE_PERFIX + '0x6aae3fd4e1545e9865bfdc92c032cb2c712fb125.sol')
