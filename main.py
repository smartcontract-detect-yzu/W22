import itertools
import subprocess

from slither.core.declarations import FunctionContract

from slither.slithir.variables import ReferenceVariable
from slither.slithir.operations import OperationWithLValue
from slither.core.expressions import (
    AssignmentOperationType,
    AssignmentOperation,
    Identifier,
)
from slither.core.cfg.node import NodeType
from slither.slither import Slither
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot
from queue import LifoQueue

EXAMPLE_PERFIX = "examples/ponzi/"
DEBUG_PNG = 1


def get_graph_png(graph: nx.Graph, postfix):
    if DEBUG_PNG != 1:
        return

    dot_name = EXAMPLE_PERFIX + "{}_{}.dot".format(graph.graph["name"], postfix)
    nx_dot.write_dot(graph, dot_name)

    cfg_name = EXAMPLE_PERFIX + "{}_{}.png".format(graph.graph["name"], postfix)

    subprocess.check_call(["dot", "-Tpng", dot_name, "-o", cfg_name])


def add_exit_point(cfg: nx.DiGraph):
    leaves = []

    # 所有叶子节点
    for cfg_node in cfg.nodes:
        if cfg.out_degree(cfg_node) == 0:  # 叶子节点列表
            leaves.append(cfg_node)

    # 添加函数的 EXIT_POINT
    cfg.add_node("EXIT_POINT", label="EXIT_POINT")

    # 将所有叶子节点连接到EXIT_POINT
    for leaf_node in leaves:
        cfg.add_edge(leaf_node, "EXIT_POINT")


def create_forward_dominators(cfg: nx.DiGraph):
    # 前向cfg：将cfg反转
    reverse_cfg = cfg.reverse()

    # 获得前向支配关系： immediate forward dominator for each node
    ifdom_list = nx.algorithms.immediate_dominators(reverse_cfg, "EXIT_POINT")
    del ifdom_list["EXIT_POINT"]
    return ifdom_list


# CDG = CFG + FDT 计算 条件语句相关控制依赖
def calc_control_dependency_relation(cfg, if_statements, ifdom_list, stmt_infos):
    control_dep_edge = {}  # 去重
    cdg_edges = []  # 所有候选的控制依赖边信息

    for if_statement_id in if_statements:
        stmt_id = str(if_statement_id)
        ifdom = ifdom_list[stmt_id]
        # print("ifdom of {} is {}".format(stmt_id, ifdom))

        cfg_paths = nx.all_simple_paths(cfg, source=stmt_id, target="EXIT_POINT")

        # Y is control dependent on X ⇔ there is a path in the CFG from X to Y that doesn’t contain the immediate
        # forward dominator of X
        for path in list(cfg_paths):
            # 掐头去尾：计算当前if语句以下所有语句对当前if语句的依赖
            # 掐头：保证只计算if语句之后的语句与当前if语句的关系,去掉当前if语句
            # 去尾：删除函数退出点
            for node in path[1:-1]:
                stmt_info = stmt_infos[node]

                if stmt_info.type != NodeType.ENDIF and stmt_info.type != NodeType.ENDLOOP:
                    if node != ifdom:
                        key = "{}-{}".format(node, stmt_id)
                        if key not in control_dep_edge:
                            control_dep_edge[key] = 1
                            length = nx.shortest_path_length(cfg, stmt_id, node)
                            cdg_edges.append({'from': node, "to": stmt_id, 'color': 'red', 'distance': length})
                            # print("{} 控制依赖于 {}, 距离是: {}".format(node, stmt_id, length))
                    else:
                        break

    control_dep_edge.clear()
    for cdg_edge in cdg_edges:
        from_node = cdg_edge["from"]
        to_node = cdg_edge["to"]
        distance = cdg_edge["distance"]
        if from_node not in control_dep_edge:
            control_dep_edge[from_node] = {"to": to_node, "distance": distance}
        else:
            old_distance = control_dep_edge[from_node]["distance"]
            if old_distance > distance:
                control_dep_edge[from_node] = {"to": to_node, "distance": distance}

    for from_node in control_dep_edge:
        cfg.add_edge(from_node, control_dep_edge[from_node]["to"], color="red", type="ctrl_dependency")


def control_dependency(cfg: nx.DiGraph, if_id, node_infos):
    # 前向支配关系构建 Immediate forward dominators
    ifdom_list = create_forward_dominators(cfg)

    # 控制流依赖计算：并删减一部分与if语句无关控制依赖
    calc_control_dependency_relation(cfg, if_id, ifdom_list, node_infos)

    # 带控制依赖边的cfg
    get_graph_png(cfg, "cdg")


def FDT2CDG(cfg: nx.DiGraph, if_id, node_infos):
    leafs = []
    control_dep_edge = {}
    cdg_edges = []

    idoms = nx.algorithms.immediate_dominators(cfg, "0")
    print("后向支配关系：", idoms)

    # 查询当前cfg所有叶子节点
    for cfg_node in cfg.nodes:
        if cfg.out_degree(cfg_node) == 0:  # 叶子节点列表
            leafs.append(cfg_node)

    cfg.add_node("EXIT_POINT", label="EXIT_POINT")
    for leaf_node in leafs:
        cfg.add_edge(leaf_node, "EXIT_POINT")

    get_graph_png(cfg, "cfg")

    reverse_cfg = cfg.reverse()

    get_graph_png(reverse_cfg, "reverse")

    ifdoms = nx.algorithms.immediate_dominators(reverse_cfg, "EXIT_POINT")
    del ifdoms["EXIT_POINT"]
    print("前向支配关系：", ifdoms)

    # FDT
    fdt = nx.DiGraph(name=cfg.graph["name"])
    fdt.add_nodes_from(reverse_cfg.nodes)
    for s in ifdoms:
        fdt.add_edge(ifdoms[s], s)

    get_graph_png(fdt, "fdt")

    # CDG = CFG + FDT 计算 条件语句相关控制依赖
    for id in if_id:
        ifdom = ifdoms[str(id)]
        print("ifdom of {} is {}".format(id, ifdom))
        cfg_paths = nx.all_simple_paths(cfg, source=str(id), target="EXIT_POINT")

        # Y is control dependent on X ⇔ there is a path in the CFG from X to Y that doesn’t contain the immediate
        # forward dominator of X
        for path in list(cfg_paths):
            for node in path[1:-1]:
                node_info = node_infos[node]
                if node_info.type != NodeType.ENDIF and node_info.type != NodeType.ENDLOOP:
                    if node != ifdom:
                        key = "{}-{}".format(node, str(id))
                        if key not in control_dep_edge:
                            control_dep_edge[key] = 1
                            length = nx.shortest_path_length(cfg, str(id), node)
                            cdg_edges.append({'from': node, "to": str(id), 'color': 'red', 'distance': length})
                            print("{} 控制依赖于 {}, 距离是: {}".format(node, id, length))
                    else:
                        break

    control_dep_edge.clear()
    for cdg_edge in cdg_edges:
        from_node = cdg_edge["from"]
        to_node = cdg_edge["to"]
        distance = cdg_edge["distance"]
        if from_node not in control_dep_edge:
            control_dep_edge[from_node] = {"to": to_node, "distance": distance}
        else:
            old_distance = control_dep_edge[from_node]["distance"]
            if old_distance > distance:
                control_dep_edge[from_node] = {"to": to_node, "distance": distance}

    for from_node in control_dep_edge:
        cfg.add_edge(from_node, control_dep_edge[from_node]["to"], color="red")

    get_graph_png(cfg, "cdg")


def get_png(target_fun):
    cfg_dot_file = EXAMPLE_PERFIX + "{}_cfg.dot".format(target_fun.name)
    cfg_png = EXAMPLE_PERFIX + "{}_cfg.png".format(target_fun.name)
    dom_tree_dot_file = EXAMPLE_PERFIX + "{}_dom.dot".format(target_fun.name)
    dom_png = EXAMPLE_PERFIX + "{}_dom.png".format(target_fun.name)

    target_fun.cfg_to_dot(cfg_dot_file)
    target_fun.dominator_tree_to_dot(dom_tree_dot_file)
    subprocess.check_call(["dot", "-Tpng", cfg_dot_file, "-o", cfg_png])
    subprocess.check_call(["dot", "-Tpng", dom_tree_dot_file, "-o", dom_png])

    return cfg_dot_file


def break_loop(cfg: nx.DiGraph, loop_circle, node_infos):
    remove_edges = []
    for loop_tell in loop_circle:
        for pred_node in loop_circle[loop_tell]:
            if node_infos[pred_node].type != NodeType.STARTLOOP:
                remove_edges.append((pred_node, loop_tell))

    cfg.remove_edges_from(remove_edges)


def debug_vars(stmt_infos, stmts_var_miss_maps):
    for stmt in stmt_infos:

        stmt_info = stmt_infos[stmt]
        stmt_id = str(stmt_info.node_id)

        # if stmt_id in stmts_var_miss_maps:
        #     print("错误变量：", stmts_var_miss_maps[stmt_id])

        if stmt_info.expression:
            print("======Expression: {}".format(stmt_info.expression))
            print("\t语句类型：{} 语句ID {}".format(stmt_info.type, stmt_id))
            print("\t变量声明：", str(stmt_info.variable_declaration))
            print("\t局部变量读：", [str(v) for v in stmt_info.local_variables_read])
            print("\t局部变量写：", [str(v) for v in stmt_info.local_variables_written])
            print("\t全局变量读：", [str(v) for v in stmt_info.state_variables_read])
            print("\t全局变量写：", [str(v) for v in stmt_info.state_variables_written])
            print("\tSlithIR:")
            for ir in stmt_info.irs:
                print("\t\t{}".format(ir))

    for stmt in stmts_var_miss_maps:
        print("错误变量：", stmts_var_miss_maps[stmt])


def _fix_data_def_use_error(stmt_id, stmt_type, miss_info, stmt_expression, vars):
    """
    规避SSA数据分析的bug

    """
    ret_vars = []
    miss_vars = []
    miss_ret = 0

    for var in vars:
        if var in stmt_expression:
            ret_vars.append(var)
        else:
            miss_vars.append(var)

    if len(miss_vars) != 0:
        miss_info[stmt_id] = ["语句:{}\n 变量：{}".format(stmt_expression, miss_vars)]
        miss_ret = 1

    return ret_vars, miss_ret


def stmts_var_infos(stmt_infos):
    stmts_var_info_maps = {}
    stmts_var_miss_maps = {}

    for stmt in stmt_infos:

        stmt_var_info = []

        stmt_info = stmt_infos[stmt]
        stmt_type = stmt_info.type
        stmt_id = str(stmt_info.node_id)
        expression = str(stmt_info.expression)

        no_write = no_read = 0
        if stmt_info.type == NodeType.IFLOOP:
            no_write = 1  # if语句不许写

        # 声明的变量 def
        if no_write == 0 and stmt_info.variable_declaration is not None:
            declar_var = [str(stmt_info.variable_declaration)]
            fixed_declar_var, ret_miss = _fix_data_def_use_error(stmt_id, stmt_type, stmts_var_miss_maps, expression,
                                                                 declar_var)
            # print("声明变量：", declar_var)
            stmt_var_info.append({"list": fixed_declar_var, "type": "local", "op_type": "def"})

        # 局部变量写 def
        write_local_var = [str(var) for var in stmt_info.local_variables_written]
        if no_write == 0 and len(write_local_var) != 0:
            # print("写局部变量：",write_local_var)
            fixed_write_local_var, ret_miss = _fix_data_def_use_error(stmt_id, stmt_type, stmts_var_miss_maps,
                                                                      expression,
                                                                      write_local_var)
            stmt_var_info.append({"list": fixed_write_local_var, "type": "local", "op_type": "def"})

        # 全局变量写 def
        write_state_var = [str(var) for var in stmt_info.state_variables_written]
        if no_write == 0 and len(write_state_var) != 0:
            fixed_write_state_var, ret_miss = _fix_data_def_use_error(stmt_id, stmt_type, stmts_var_miss_maps,
                                                                      expression,
                                                                      write_state_var)
            stmt_var_info.append({"list": fixed_write_state_var, "type": "state", "op_type": "def"})

        # 局部变量读 use
        read_local_var = [str(var) for var in stmt_info.local_variables_read]
        if len(read_local_var) != 0:
            fixed_read_local_var, ret_miss = _fix_data_def_use_error(stmt_id, stmt_type, stmts_var_miss_maps,
                                                                     expression,
                                                                     read_local_var)
            stmt_var_info.append({"list": fixed_read_local_var, "type": "local", "op_type": "use"})

        # 全局变量读 use
        read_state_var = [str(var) for var in stmt_info.state_variables_read]
        if len(read_state_var) != 0:
            fixed_read_state_var, ret_miss = _fix_data_def_use_error(stmt_id, stmt_type, stmts_var_miss_maps,
                                                                     expression,
                                                                     read_state_var)
            stmt_var_info.append({"list": fixed_read_state_var, "type": "state", "op_type": "use"})

        # 记录下当前语句使用的变量信息
        stmts_var_info_maps[stmt_id] = stmt_var_info

    return stmts_var_info_maps, stmts_var_miss_maps


def data_depency(cfg: nx.DiGraph, stmt_infos):
    var_def_use_chain = {}
    ddg_edges_info = []

    stmts_var_info_maps, stmts_var_miss_maps = stmts_var_infos(stmt_infos)
    # debug_vars(stmt_infos, stmts_var_miss_maps)

    cfg_paths = nx.all_simple_paths(cfg, source="0", target="EXIT_POINT")
    for path in cfg_paths:

        var_def_use_chain.clear()  # 每条可能的执行路径都单独计算

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

        # 计算当前执行路径下的def_use chain分析
        for var in var_def_use_chain:
            chain = var_def_use_chain[var]
            last_def = None
            for chain_node in chain:
                if chain_node["op_type"] == "def":
                    last_def = chain_node["id"]
                else:
                    if last_def is not None:
                        edge_info = {"from": chain_node["id"], "to": last_def}
                        ddg_edges_info.append(edge_info)

    dulicate = {}
    ddg_edges = []
    for edge_info in ddg_edges_info:

        if edge_info["from"] == edge_info["to"]:
            continue

        key = "{}-{}".format(edge_info["from"], edge_info["to"])
        if key not in dulicate:
            dulicate[key] = 1
            ddg_edges.append((edge_info["from"], edge_info["to"], {'color': "green", "type": "data_dependency"}))
    cfg.add_edges_from(ddg_edges)
    get_graph_png(cfg, "cdg_ddg")


# 前向遍历
def _forward_analyze(cfg, criteria):
    result = {}

    stack = LifoQueue()
    stack.put(criteria)

    while stack.qsize() > 0:
        stmt = stack.get()
        for v in cfg.successors(stmt):  # 后继节点
            edge_data = cfg.get_edge_data(stmt, v)[0]
            if "type" in edge_data:

                print("节点 {} - 前驱 {} -  {}".format(stmt, v, edge_data["type"]))
                if v not in result:
                    result[v] = 1
                    stack.put(v)

    return result


def forward_analyze(cfg, criterias, stmt_infos):
    state_related = {}
    duplicate = {}

    for criteria in criterias:

        result = _forward_analyze(cfg, criteria)
        for related_node in result:
            if related_node not in duplicate:
                duplicate[related_node] = 1
                stmt_info = stmt_infos[related_node]

                for v in stmt_info.state_variables_read:
                    if str(v) not in state_related:
                        state_related[str(v)] = 1

                for v in stmt_info.state_variables_written:
                    if str(v) not in state_related:
                        state_related[str(v)] = 1

    print("\t分析结果：涉及的全局变量：{}\n".format(state_related))
    return state_related


def get_function_cfg(function):
    cfg_dot_file = get_png(function)
    cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
    cfg.graph["name"] = function.name

    for node in function.nodes:
        cfg_node = cfg.nodes[str(node.node_id)]
        cfg_node["expression"] = node.expression.__str__()
        cfg_node["type"] = node.type.__str__()
    return cfg


def _pre_process_function(cfg, function):
    if_ids = []
    criteria = []
    loop_circle = {}
    stmt_infos = {}

    for stmt in function.nodes:
        stmt_infos[str(stmt.node_id)] = stmt

        if stmt.can_send_eth():
            criteria.append(str(stmt.node_id))
            print("\t交易语句：{} 变量：{}".format(stmt.expression, [str(v) for v in stmt.variables_read]))

        if stmt.type == NodeType.IFLOOP:
            loop_circle[str(stmt.node_id)] = cfg.predecessors(str(stmt.node_id))

        if stmt.type == NodeType.IF or stmt.type == NodeType.IFLOOP:
            if_ids.append(stmt.node_id)

    break_loop(cfg, loop_circle, stmt_infos)  # 破环
    add_exit_point(cfg)  # 添加函数的退出节点

    return if_ids, criteria, stmt_infos


def _debug_stat_var_info(state_var_read_function_map, state_var_write_function_map, state_var_declare_function_map):
    print("===全局变量定义信息：")
    for var in state_var_declare_function_map:
        print("\t定义变量{}".format(str(var)))

        if "exp" in state_var_declare_function_map[var]:
            print("\t\t{}".format(state_var_declare_function_map[var]["exp"]))

        if "fun" in state_var_declare_function_map[var]:
            print("\t\t{}".format(state_var_declare_function_map[var]["fun"].name))

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


def _do_slice_remove_node(g, node):
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


def do_slice(function, slice_criterias):
    print("对函数 {} 进行切片!!!!!!!!!!!!!!!!!!".format(function.full_name))

    cfg = get_function_cfg(function)

    if_ids, criteria, stmt_infos = _pre_process_function(cfg, function)  # 预处理

    control_dependency(cfg, if_ids, stmt_infos)  # 控制依赖

    data_depency(cfg, stmt_infos)  # 数据依赖

    reserve_nodes = {}
    remove_nodes_map = {}
    remove_nodes = []
    for slice_criteria in slice_criterias:

        reserve_nodes[slice_criteria] = 1
        cfg.nodes[slice_criteria]["color"] = "red"

        # 基于切片准则的前向依赖分析
        depency_nodes = _forward_analyze(cfg, slice_criteria)
        print("准则：{} 结果：{}".format(slice_criteria, [node_id for node_id in depency_nodes]))
        for node in depency_nodes:
            if node not in reserve_nodes:
                reserve_nodes[node] = 1
                cfg.nodes[node]["color"] = "red"

    for cfg_node_id in cfg.nodes:
        if cfg_node_id not in reserve_nodes:
            remove_nodes_map[cfg_node_id] = 1
            if cfg_node_id == "EXIT_POINT":
                remove_nodes.append(9999)
            else:
                remove_nodes.append(int(cfg_node_id))

    print("函数 {} 切片后：".format(function.full_name))
    print("\t需要保留的节点为：{}".format([node_id for node_id in reserve_nodes]))
    print("\t需要删除的节点为：{}".format([node_id for node_id in remove_nodes_map]))
    get_graph_png(cfg, "cdg_ddg")

    remove_nodes.sort(reverse=True)
    print("remove_nodes:{}".format(remove_nodes))
    sliced_cfg = cfg

    for remove_node in remove_nodes:
        if remove_node == 9999:
            remove_node_id = "EXIT_POINT"
        else:
            remove_node_id = str(remove_node)

        sliced_cfg = _do_slice_remove_node(cfg, remove_node_id)

    get_graph_png(sliced_cfg, "cdg_ddg_slice")
    return sliced_cfg


# 函数间数据依赖分析
def interfunction_analyze(sliced_functions, sliced_cfgs:[nx.DiGraph]):
    stat_var_write_info = {}
    stat_var_read_info = {}

    for function, sliced_cfg in zip(sliced_functions, sliced_cfgs):

        for node in function.nodes:

            # write
            for v in node.state_variables_written:
                info = {"type": "w", "fun": function.name, "cfg_id": node.node_id}
                if str(node.node_id) in sliced_cfg.nodes:  # 未被切片掉

                    if str(v) not in stat_var_write_info:
                        stat_var_write_info[str(v)] = [info]
                    else:
                        stat_var_write_info[str(v)].append(info)

            # read
            for v in node.state_variables_read:
                info = {"type": "r", "fun": function.name, "cfg_id": node.node_id}

                if str(node.node_id) in sliced_cfg.nodes:  # 未被切片掉
                    if str(v) not in stat_var_read_info:
                        stat_var_read_info[str(v)] = [info]
                    else:
                        stat_var_read_info[str(v)].append(info)

    #print("跨函数分析写操作：{}".format(stat_var_write_info))
    #print("跨函数分析读操作：{}".format(stat_var_read_info))

    return stat_var_write_info, stat_var_read_info


def _debug_function_and_slice_criterias(functions, functions_slice_criterias):
    sliced_cfgs = []
    sliced_functions = []

    print("=====需要进行切片的函数为：")
    for function in functions:

        if function.full_name in functions_slice_criterias:
            print("\t重点分析函数：{}".format(function.full_name))
            for sc in functions_slice_criterias[function.full_name]:
                stmt = function.nodes[int(sc)]
                print("\t\t切片准则：{}".format(stmt.expression))

            sliced_cfg = do_slice(function, functions_slice_criterias[function.full_name])
            sliced_functions.append(function)
            sliced_cfgs.append(sliced_cfg)

    stat_var_write_info, stat_var_read_info = interfunction_analyze(sliced_functions, sliced_cfgs)
    combine_cfgs(sliced_cfgs, stat_var_write_info, stat_var_read_info)


def combine_cfgs(sliced_cfgs, stat_var_write_info, stat_var_read_info):
    TPG = nx.DiGraph(name="test")

    print("需要合并的cfg个数：{}".format(len(sliced_cfgs)))

    for cfg in sliced_cfgs:

        new_edges = []
        function_name = cfg.graph["name"]
        print("函数名称：{}".format(function_name))

        for node_id in cfg.nodes:
            new_node_id = "{}@{}".format(node_id, function_name)
            old_node_info = cfg.nodes[node_id]
            TPG.add_node(new_node_id, label=old_node_info["expression"], type=old_node_info["type"], fun=function_name)

        for edge in cfg.edges:

            from_node = edge[0]
            to_node = edge[1]
            edge_id = edge[2]
            info = cfg.edges[from_node, to_node, edge_id]

            if to_node != from_node:
                new_from_node = "{}@{}".format(from_node, function_name)
                new_to_node = "{}@{}".format(to_node, function_name)
                if "type" in info:
                    edge_type = info["type"]
                    edge_color = info["color"]
                else:
                    edge_type = "ctrl_flow"
                    edge_color = "black"
                new_edges.append((new_from_node, new_to_node, {'type': edge_type, 'color': edge_color}))

        TPG.add_edges_from(new_edges)

    # write ---> read
    cross_function_edges = []
    for var in stat_var_write_info:
        if var in stat_var_read_info:
            write_infos = stat_var_write_info[var]
            read_infos = stat_var_read_info[var]

            for write_info in write_infos:
                write_function_name = write_info["fun"]
                write_cfg_id = write_info["cfg_id"]
                from_node_id = "{}@{}".format(write_cfg_id, write_function_name)

                for read_info in read_infos:
                    read_function_name = read_info["fun"]
                    read_cfg_id = read_info["cfg_id"]
                    to_node_id = "{}@{}".format(read_cfg_id, read_function_name)

                    # 跨函数的全局变量依赖
                    if write_function_name != read_function_name:
                        cross_function_edges.append((from_node_id, to_node_id, {'type': "state_cross", 'color': "blue"}))
    TPG.add_edges_from(cross_function_edges)
    get_graph_png(TPG, "TPG")


if __name__ == '__main__':

    slither = Slither(EXAMPLE_PERFIX + '0x09515cb5e3acaef239ab83d78b2f3e3764fcab9b.sol')

    for contract in slither.contracts:

        state_var_declare_function_map = {}  # 全局变量定义函数
        state_var_write_function_map = {}  # 全局变量写函数列表
        state_var_read_function_map = {}  # 全局变量读函数列表
        transaction_state_vars = []  # 与交易行为相关的全局变量
        transaction_state_vars_stmts = {}
        eth_send_functions = {}
        functions_slice_criterias = {}  # 每个函数对应的切片准则：阶段1、4都会生成切片准则

        # NOTE: stat_var <==> function 对应表
        print("\n=====阶段1：stat_var <==> function 对应表=====\n")

        for v in contract.state_variables:
            if v.expression is None:
                exp = str(v.type) + " " + str(Identifier(v))
                print("expression:", exp)
                state_var_declare_function_map[str(Identifier(v))] = {"exp": exp}

        for function in contract.functions:

            get_function_cfg(function)

            # 全局变量定义
            if function.is_constructor or function.is_constructor_variables:
                for v in function.state_variables_written:
                    if str(v) not in state_var_declare_function_map:
                        state_var_declare_function_map[str(v)] = {"fun": function}
                    else:
                        if "fun" not in state_var_declare_function_map[str(v)]:
                            state_var_declare_function_map[str(v)] = {"fun": function}
                        else: # 一个全局变量只能又一个定义的地方
                            print("全局变量 {} 在 {} 和 {} 重复定义".format(str(v), function.name, state_var_declare_function_map[str(v)]))
                            raise RuntimeError("重复定义")
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

        _debug_stat_var_info(state_var_read_function_map, state_var_write_function_map, state_var_declare_function_map)

        # NOTE: 分析与交易相关函数，抽取交易相关全局变量（transaction state find）
        print("\n=====阶段2：分析与交易相关函数，提取 交易直接相关全局变量=====\n")
        for function in contract.functions:
            if function.can_send_eth():
                print("\t开始分析函数：{}".format(function.name.__str__()))
                eth_send_functions[function.full_name.__str__()] = 1

                cfg = get_function_cfg(function)

                if_ids, criteria, stmt_infos = _pre_process_function(cfg, function)  # 预处理

                if function.full_name not in functions_slice_criterias:
                    functions_slice_criterias[function.full_name] = criteria
                else:
                    functions_slice_criterias[function.full_name] += criteria

                control_dependency(cfg, if_ids, stmt_infos)  # 控制依赖

                data_depency(cfg, stmt_infos)  # 数据依赖

                transaction_state_vars += forward_analyze(cfg, criteria, stmt_infos)  # 交易相关全局变量提取

        # NOTE: 分析交易相关全局变量定义（transaction state declare）
        print("\t交易直接相关全局变量列表：\n\t{}".format([v for v in transaction_state_vars]))

        # NOTE: 分析交易相关全局变量定义（transaction state declare）
        print("\n=====阶段3：分析涉及全局变量的定义函数=====\n")
        for transaction_state in transaction_state_vars:  # 交易相关全局变量
            if transaction_state not in state_var_declare_function_map:
                raise RuntimeError("交易相关全局变量", str(transaction_state), "缺少定义")
            else:
                if "fun" in state_var_declare_function_map[transaction_state]:
                    declare_function: FunctionContract = state_var_declare_function_map[transaction_state]["fun"]
                    for node in declare_function.nodes:
                        for var in node.state_variables_written:
                            if str(var) == transaction_state:
                                print("\t语句：{}".format(node.expression))

                if "exp" in state_var_declare_function_map[transaction_state]:
                    expression = state_var_declare_function_map[transaction_state]["exp"]
                    print("\t语句（未赋值）：{}".format(expression))

        print("\n=====阶段4：写交易直接相关全局变量函数列表=====\n")
        for transaction_state in transaction_state_vars:  # 交易相关全局变量
            if transaction_state in state_var_write_function_map:
                print("\t 写全局变量{}的函数为：".format(transaction_state))

                for write_fun in state_var_write_function_map[transaction_state]:
                    print("\t\tfun name is", write_fun.full_name)
                    for node in write_fun.nodes:

                        if node.expression is not None and len(node.state_variables_written) != 0:
                            print("\t\tFUNC_EXP:{}".format(node.expression))
                            print("\t\tSTAT_VAR:{}\n".format([str(v) for v in node.state_variables_written]))

                        for v in node.state_variables_written:
                            if transaction_state == str(v):
                                print("\t\t写全局变量{}的语句是：{}, CFG_ID为{}\n".format(transaction_state, node.expression,
                                                                               node.node_id))
                                if write_fun.full_name not in functions_slice_criterias:
                                    functions_slice_criterias[write_fun.full_name] = [str(node.node_id)]
                                else:
                                    functions_slice_criterias[write_fun.full_name].append(str(node.node_id))

        _debug_function_and_slice_criterias(contract.functions, functions_slice_criterias)

        # 函数切片
