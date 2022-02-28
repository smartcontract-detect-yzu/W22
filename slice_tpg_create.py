import subprocess

from slither import Slither
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot
from slither.core.cfg.node import NodeType

EXAMPLE_PERFIX = "examples/ponzi/"
DEBUG_PNG = 1


def debug_get_ddg_and_cdg(cfg, cdg_edges, ddg_edges, data_flows):
    # 生成控制依赖图
    cdg = nx.DiGraph(cfg)
    cdg.remove_edges_from(list(cdg.edges()))
    cdg.add_edges_from(cdg_edges)
    debug_get_graph_png(cdg, "cdg")

    # 生成数据依赖图
    ddg = nx.DiGraph(cfg)
    ddg.remove_edges_from(list(ddg.edges()))
    ddg.add_edges_from(ddg_edges)
    debug_get_graph_png(ddg, "ddg")

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
        print("ERROR IN DATA DEF-USE:")
        print("语句：{}".format(stmt_expression))
        print("变量：{}\n".format(miss_vars))

    return ret_vars


def get_function_cfg(function):
    cfg_dot_file = get_png(function)
    cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
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

    # 1:破环 将CFG中的循环结构消除
    if_stmts = []
    remove_edges = []
    stmts_send_eth = []
    stmts_var_info_maps = {}
    for stmt in function.nodes:

        # 2:当前语句的变量使用情况
        stmts_var_info_maps[str(stmt.node_id)] = _stmt_var_info(stmt)

        if stmt.can_send_eth():
            stmts_send_eth.append(stmt.node_id)

        if stmt.type == NodeType.IF or stmt.type == NodeType.IFLOOP:
            if_stmts.append(str(stmt.node_id))

        if stmt.type == NodeType.IFLOOP:
            for pre_node_id in cfg.predecessors(str(stmt.node_id)):

                # IF_LOOP 的前驱节点中非 START_LOOP 的节点到IF_LOOP的边需要删除
                if function.nodes[int(pre_node_id)].type != NodeType.STARTLOOP:
                    remove_edges.append((pre_node_id, str(stmt.node_id)))

    if len(remove_edges) != 0:
        cfg.remove_edges_from(remove_edges)

    # 2: 给CFG中的所有叶子节点添加exit子节点作为函数退出的标识符
    leaf_nodes = []
    for cfg_node_id in cfg.nodes:
        if cfg.out_degree(cfg_node_id) == 0:  # 叶子节点列表
            leaf_nodes.append(cfg_node_id)
    debug_get_graph_png(cfg, "cfg_noloop")

    cfg.add_node("EXIT_POINT", label="EXIT_POINT")
    for leaf_node in leaf_nodes:
        cfg.add_edge(leaf_node, "EXIT_POINT")

    debug_get_graph_png(cfg, "cfg_exit")

    return cfg, if_stmts, stmts_var_info_maps, stmts_send_eth


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


def get_control_dependency_relations(simple_cfg, if_stmts, function):
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
        cdg_edges.append((from_node, to_node, {'color': "red", "type": "ctrl_dependency"}))

    return cdg_edges


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

    duplicate = {}
    ddg_edges = []
    for edge_info in data_dependency_relations:

        if edge_info["from"] == edge_info["to"]:
            continue

        key = "{}-{}".format(edge_info["from"], edge_info["to"])
        if key not in duplicate:
            duplicate[key] = 1
            ddg_edges.append((edge_info["from"], edge_info["to"], {'color': "green", "type": "data_dependency"}))

    print("数据依赖：", ddg_edges)
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
                                        data_flow_edges.append((from_node, to_node, {'color': "yellow", "type": "data_flow"}))
                                del use_info[var]

                            # kill：中断上一次def的flow
                            def_info[var] = from_node

    print("数据流:", data_flow_edges)
    return data_flow_edges


def create_pdg(function):
    # 获得控制流图
    cfg = get_function_cfg(function)

    # 预处理
    simple_cfg, if_stmts, stmts_var_info_maps, transaction_stmts = _preprocess_for_dependency_analyze(cfg, function)

    # 数据流分析
    data_flow_edges = trans_data_flow_analyze(simple_cfg, stmts_var_info_maps, transaction_stmts)

    # 控制依赖关系
    cdg_edges = get_control_dependency_relations(simple_cfg, if_stmts, function)

    # 数据依赖生成
    ddg_edges = get_data_dependency_relations(simple_cfg, stmts_var_info_maps)

    debug_get_ddg_and_cdg(cfg, cdg_edges, ddg_edges, data_flow_edges)


if __name__ == '__main__':

    slither = Slither(EXAMPLE_PERFIX + '0x09515cb5e3acaef239ab83d78b2f3e3764fcab9b.sol')

    for contract in slither.contracts:
        print("当前合约名称{}  当前合约类型：{}".format(contract.name, contract.contract_kind))

        for function in contract.functions:
            if function.can_send_eth():
                print("目标函数：{}".format(function.name))
                create_pdg(function)
