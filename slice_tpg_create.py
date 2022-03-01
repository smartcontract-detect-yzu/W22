import itertools
import os
import subprocess
from queue import LifoQueue

from slither import Slither
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot
from slither.core.cfg.node import NodeType

EXAMPLE_PERFIX = "examples/ponzi/"
DEBUG_PNG = 1


def debug_get_ddg_and_cdg(cfg, cdg_edges, ddg_edges, data_flows, ddg_edges_loop):
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


def debug_get_graph_png(graph: nx.Graph, postfix):
    if DEBUG_PNG != 1:
        return

    dot_name = EXAMPLE_PERFIX + "{}_{}.dot".format(graph.graph["name"], postfix)
    nx_dot.write_dot(graph, dot_name)

    cfg_name = EXAMPLE_PERFIX + "{}_{}.png".format(graph.graph["name"], postfix)

    subprocess.check_call(["dot", "-Tpng", dot_name, "-o", cfg_name])
    os.remove(dot_name)


def get_png(target_fun):
    cfg_dot_file = EXAMPLE_PERFIX + "{}_cfg.dot".format(target_fun.name)
    cfg_png = EXAMPLE_PERFIX + "{}_cfg.png".format(target_fun.name)
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
        print("\n==ERROR IN DATA DEF-USE==")
        print("\t语句：{}".format(stmt_expression))
        print("\t变量：{}".format(miss_vars))
        print("==ERROR IN DATA DEF-USE==\n")

    return ret_vars


def get_function_cfg(function):
    cfg_dot_file = get_png(function)
    cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
    os.remove(cfg_dot_file)
    cfg.graph["name"] = function.name

    for node in function.nodes:
        cfg_node = cfg.nodes[str(node.node_id)]
        cfg_node["expression"] = node.expression.__str__()
        cfg_node["type"] = node.type.__str__()
    return cfg


def _stmt_var_info(stmt_info):
    stmt_var_info = []

    expression = str(stmt_info.expression)

    no_write = no_read = 0
    if stmt_info.type == NodeType.IFLOOP:
        no_write = 1  # if语句不许写

    # 当前语句声明的变量
    if no_write == 0 and stmt_info.variable_declaration is not None:
        declare_vars = [str(stmt_info.variable_declaration)]
        rechecked_declare_var = _recheck_vars_in_expression(expression, declare_vars)
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
        stmt_var_info.append({"list": rechecked_write_state_vars, "type": "state", "op_type": "def"})

    # 局部变量读 use
    read_local_vars = [str(var) for var in stmt_info.local_variables_read]
    if len(read_local_vars) != 0:
        rechecked_read_local_vars = _recheck_vars_in_expression(expression, read_local_vars)
        stmt_var_info.append({"list": rechecked_read_local_vars, "type": "local", "op_type": "use"})

    # 全局变量读 use
    read_state_vars = [str(var) for var in stmt_info.state_variables_read]
    if len(read_state_vars) != 0:
        rechecked_read_state_vars = _recheck_vars_in_expression(expression, read_state_vars)
        stmt_var_info.append({"list": rechecked_read_state_vars, "type": "state", "op_type": "use"})

    return stmt_var_info


def _preprocess_for_dependency_analyze(or_cfg, function):
    """
    1.破环：将CFG中的循环结构消除
        IF_LOOP的前驱节点，并且该前驱节点不是BEGIN_LOOP
        BEGIN_LOOP --> IF_LOOP --> EXPRESSION
                         <----LOOP------|
    2.每条语句的变量使用情况解析，方便后续数据依赖分析
    3.exit：给CFG中的所有叶子节点添加exit子节点作为函数退出的标识符

    Parameters:
    cfg      - 当前函数的cfg
    function - 当前函数的slither分析结果

    Return：
    cfg - 没有循环体，且添加了函数退出节点的控制流图
    if_stmts - 所有条件语句
    stmts_var_info_maps - 所有语句的变量使用情况
    stmts_send_eth - 所有交易语句
    """

    cfg = nx.DiGraph(or_cfg)
    stack = []

    # 1:破环 将CFG中的循环结构消除
    if_stmts = []
    remove_edges = []
    stmts_send_eth = []
    stmts_loops = []
    stmts_var_info_maps = {}
    if_paris = {}
    for stmt in function.nodes:

        # 2:当前语句的变量使用情况
        stmts_var_info_maps[str(stmt.node_id)] = _stmt_var_info(stmt)

        if stmt.can_send_eth():
            stmts_send_eth.append(stmt.node_id)
            print("==== 切片准则：{}".format(stmt.expression))
            print("\t 变量读：{}".format([str(v) for v in stmt.variables_read]))
            print("\t 变量写：{}".format([str(v) for v in stmt.variables_written]))

        if stmt.type == NodeType.IF:
            stack.append(str(stmt.node_id))
            if_stmts.append(str(stmt.node_id))

        if stmt.type == NodeType.STARTLOOP:
            # begin_loop --> if_loop
            for suc_node_id in cfg.successors(str(stmt.node_id)):
                if function.nodes[int(suc_node_id)].type == NodeType.IFLOOP:
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
                if function.nodes[int(pre_node_id)].type != NodeType.STARTLOOP:
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

    return cfg, if_stmts, stmts_var_info_maps, stmts_send_eth, stmts_loops, if_paris


def _get_control_dependency_relations(cfg, if_stmts, predom_relations, function):
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
                node_info = function.nodes[int(y)]

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
        print("错误语句：{} to {}".format(from_node, to_node))
        raise RuntimeError("控制依赖并没有指向条件语句")


def get_control_dependency_relations(simple_cfg, if_stmts, function, if_paris):
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
    control_dep_relations = _get_control_dependency_relations(simple_cfg, if_stmts, predom_relations, function)

    cdg_edges = []
    for from_node in control_dep_relations:
        to_node = control_dep_relations[from_node]["to"]

        # NOTE: 控制依赖边方向和控制流边方向相反
        if not if_exit_fliter(simple_cfg, from_node, to_node, if_paris):
            cdg_edges.append((from_node, to_node, {'color': "red", "type": "ctrl_dependency"}))
        else:
            print("过滤控制流边{}-{}".format(from_node, to_node))
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

    # print("数据依赖：", ddg_edges)
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
                    ddg_edges.append(
                        (edge_info["from"], edge_info["to"], {'color': "green", "type": "data_dependency"}))

    return ddg_edges


def trans_data_flow_analyze(cfg, stmts_var_info_maps, transaction_stmts):
    """
    交易行为相关数据流分析
    transaction：<to, money>
    反向分析控制流，得到数据流
    """
    data_flow_edges = []
    duplicate = {}
    for trans_stmt in transaction_stmts:

        cfg_paths = nx.all_simple_paths(cfg, source="0", target=str(trans_stmt))
        for cfg_path in cfg_paths:

            def_info = {}
            use_info = {}

            for from_node in reversed(cfg_path):

                # [{"list": rechecked_read_state_vars, "type": "state", "op_type": "use"}]
                current_stmt_vars = stmts_var_info_maps[from_node]
                for var_info in current_stmt_vars:

                    # 读变量：压栈
                    if var_info["op_type"] == "use":
                        for var in var_info["list"]:
                            if var not in use_info:
                                use_info[var] = [from_node]
                            else:
                                use_info[var].append(from_node)

                    # 如果当前语句有写操作，查询之前语句对该变量是否有读操作
                    # 写变量：所有读操作出栈
                    if var_info["op_type"] == "def":
                        for var in var_info["list"]:
                            if var in use_info:
                                for to_node in use_info[var]:
                                    key = "{}-{}".format(from_node, to_node)
                                    if key not in duplicate:
                                        duplicate[key] = 1
                                        data_flow_edges.append(
                                            (from_node, to_node, {'color': "yellow", "type": "data_flow"}))
                                del use_info[var]

                            # kill：中断上一次def的flow
                            def_info[var] = from_node

    # print("数据流:", data_flow_edges)
    return data_flow_edges


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
                # print("分析：{} -{}- {}".format(current_stmt, edge_data,successor_stmt))

                if "type" in edge_data:
                    # print("节点{} 依赖于 {} as {}".format(current_stmt, successor_stmt, edge_data["type"]))
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


def program_slice(cfg, semantic_edges, criterias):
    pdg = nx.MultiDiGraph(cfg)
    for semantic_type in semantic_edges:
        if semantic_type == "ctrl_dep" \
                or semantic_type == "data_dep" \
                or semantic_type == "loop_data_dep":
            pdg.add_edges_from(semantic_edges[semantic_type])

    for criteria in criterias:
        # 针对每个切片准则进行前向依赖分析
        reserved_nodes = _forward_analyze(pdg, criteria)

        sliced_cfg = do_slice(cfg, reserved_nodes)

        # debug_get_graph_png(sliced_cfg, "sliced_cfg_{}".format(criteria))

        new_edges = []
        for semantic_type in semantic_edges:
            for edge in semantic_edges[semantic_type]:
                if str(edge[0]) in reserved_nodes and str(edge[1]) in reserved_nodes:
                    new_edges.append(edge)

        sliced_pdg = nx.MultiDiGraph(sliced_cfg)
        sliced_pdg.add_edges_from(new_edges)

        debug_get_graph_png(sliced_pdg, "sliced_pdg_{}".format(criteria))


def loop_structure_extreact(simple_cfg, loop_structures, criteria):
    """
    循环体执行
    for(循环条件){
        A ; criteria ;B ;C ;D
    }
    存在执行路径 <criteria, B, C, D, 循环条件>, 需要分析该路径的数据依赖关系
    """
    loop_reverse_paths = []
    for loop_structure in loop_structures:

        src = loop_structure["from"]
        dst = loop_structure["to"]

        cfg_paths = nx.all_simple_paths(simple_cfg, source=src, target=dst)
        for cfg_path in cfg_paths:

            for index, path_node in enumerate(cfg_path):
                if path_node == str(criteria):
                    loop_exe_path = cfg_path[index + 1:] + [path_node] + ["EXIT_POINT"]  # 将初始节点放在最后
                    loop_reverse_paths.append(loop_exe_path)
                    break
    return loop_reverse_paths


def create_pdg(function):
    semantic_edges = {}

    # 获得控制流图
    cfg = get_function_cfg(function)

    # 预处理
    simple_cfg, if_stmts, stmts_var_info_maps, transaction_stmts, loop_stmts, if_paris \
        = _preprocess_for_dependency_analyze(cfg, function)

    print("if 配对结果：{}".format(if_paris))

    # 数据流分析
    data_flow_edges = trans_data_flow_analyze(simple_cfg, stmts_var_info_maps, transaction_stmts)
    semantic_edges["data_flow"] = data_flow_edges

    # 控制依赖关系
    cdg_edges = get_control_dependency_relations(simple_cfg, if_stmts, function, if_paris)
    semantic_edges["ctrl_dep"] = cdg_edges

    # 数据依赖生成
    ddg_edges = get_data_dependency_relations(simple_cfg, stmts_var_info_maps)
    semantic_edges["data_dep"] = ddg_edges

    # 循环体内部的数据流依赖
    loop_ddg_edges = get_data_dependency_relations_forloop(simple_cfg, stmts_var_info_maps, transaction_stmts,
                                                           loop_stmts)
    semantic_edges["loop_data_dep"] = loop_ddg_edges

    # debug_get_ddg_and_cdg(cfg, cdg_edges, ddg_edges, data_flow_edges, loop_ddg_edges)

    # 程序切片
    program_slice(cfg, semantic_edges, transaction_stmts)

if __name__ == '__main__':

    slither = Slither(EXAMPLE_PERFIX + '0x09515cb5e3acaef239ab83d78b2f3e3764fcab9b.sol')
    # slither = Slither(EXAMPLE_PERFIX + 'test.sol')
    # slither = Slither(EXAMPLE_PERFIX + '0x6aae3fd4e1545e9865bfdc92c032cb2c712fb125.sol')

    for contract in slither.contracts:
        print("当前合约名称{}  当前合约类型：{}".format(contract.name, contract.contract_kind))

        for function in contract.functions:
            if function.can_send_eth():
                print("\n目标函数：{}".format(function.name))
                create_pdg(function)
