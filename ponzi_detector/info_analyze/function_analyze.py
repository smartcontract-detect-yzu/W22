import os

import networkx as nx
import networkx.drawing.nx_pydot as nx_dot

from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from slither.core.declarations.function import Function
from slither.core.cfg.node import NodeType, Node
import subprocess


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
        # print("\n\t==ERROR IN DATA DEF-USE==")
        # print("\t\t语句：{}".format(stmt_expression))
        # print("\t\t变量：{}".format(miss_vars))
        # print("\t==ERROR IN DATA DEF-USE==\n")
        pass

    return ret_vars


EXAMPLE_PERFIX = "examples/ponzi/"


def debug_get_graph_png(graph: nx.Graph, postfix, cur_dir):
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


class FunctionInfo:

    def __init__(self, contract_info: ContractInfo, function: Function):

        self.contract_info = contract_info
        self.can_send_ether = function.can_send_eth()
        self.function = function
        self.name = function.name

        # 语义相关 初始信息
        self.if_stmts = []  # 条件语句列表
        self.stmts_var_info_maps = {}  # 各语句变量使用情况
        self.transaction_stmts = {}  # 存在交易行为的语句列表 transaction_stmts
        self.loop_stmts = []  # 循环语句列表
        self.if_paris = {}  # IF 与 END_IF
        self.node_id_2_id = {}  # function.node.node_id -> cfg id
        self.id_2_node_id = {}  # function.node.node_id <- cfg id
        self.state_def_stmts = {}  # 全局变量write(def)语句
        self.const_var_init = {}  # 涉及的全局变量初始化语句
        self.msg_value_stmt = {}  # 使用了msg.value的语句

        # 交易全局变量，需要通过数据流分析器获取
        self.transaction_states = None  # <交易语句, [交易涉及的全部全局变量]>

        # 外部函数修改全局变量语句集合
        self.external_state_def_nodes_map = None

        # 切片准则
        self.criterias = None  # 交易语句
        self.criterias_msg = None  # msg.value
        self.append_criterias = None  # 交易相关全局变量，依赖数据流分析器 DataFlowAnalyzer

        # 图表示
        self.cfg = None
        self.simple_cfg = None
        self.pdg = None
        self.sliced_pdg = {}

        # 不同的语义边
        self.semantic_edges = {}

        # 初始化
        self.function_info_analyze()

    def _get_msg_value_stmt(self, stmt):

        if "msg.value" in stmt.expression.__str__():
            self.msg_value_stmt[str(stmt.node_id)] = {
                "exp": stmt.expression.__str__()
            }

    def _get_function_cfg(self):

        cfg_dot_file = "{}_cfg.dot".format(self.function.name)
        self.function.cfg_to_dot(cfg_dot_file)

        cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
        os.remove(cfg_dot_file)
        cfg.graph["name"] = self.function.name
        cfg.graph["contract_name"] = self.contract_info.name

        for node in self.function.nodes:
            cfg_node = cfg.nodes[str(node.node_id)]
            cfg_node["expression"] = node.expression.__str__()
            cfg_node["type"] = node.type.__str__()

        return cfg

    def _get_node_id_2_cfg_id(self):

        for index, stmt in enumerate(self.function.nodes):

            # 链表下标和节点ID是不同的
            self.node_id_2_id[stmt.node_id] = index
            self.id_2_node_id[index] = stmt.node_id
            if stmt.node_id != index:
                # print("NOTE: {} with ID: {} is at index:{}".format(stmt.expression.__str__(), stmt.node_id, index))
                pass

    def _get_stat_defs(self, stmt_info, write_state_vars):

        for stat_var in write_state_vars:

            if stat_var not in self.state_def_stmts:
                self.state_def_stmts[stat_var] = [stmt_info.node_id]
            else:
                self.state_def_stmts[stat_var].append(stmt_info.node_id)

    def _get_stmt_const_var_init(self, state_vars):

        state_declare = self.contract_info.state_var_declare_function_map
        for var in state_vars:
            if var in state_declare and "full_expr" in state_declare[var]:
                if var not in self.const_var_init:
                    self.const_var_init[var] = state_declare[var]["full_expr"]

    def __stmt_var_info(self, stmt_info: Node):

        stmt_var_info = []
        expression = str(stmt_info.expression)

        # if语句不许写 https://github.com/smartcontract-detect-yzu/slither/issues/8
        no_write = 1 if stmt_info.type == NodeType.IFLOOP else 0

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
            self._get_stmt_const_var_init(rechecked_read_state_vars)  # 查看当前使用的全局变量的初始化情况

        # 当前语句声明的变量
        if no_write == 0 and stmt_info.variable_declaration is not None:
            declare_vars = [str(stmt_info.variable_declaration)]
            rechecked_declare_var = _recheck_vars_in_expression(expression, declare_vars)
            stmt_var_info.append({"list": rechecked_declare_var, "type": "local", "op_type": "def"})
            self._get_stmt_const_var_init(rechecked_declare_var)  # 查看当前使用的全局变量的初始化情况

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
            self._get_stmt_const_var_init(rechecked_write_state_vars)  # 查看当前使用的全局变量的初始化情况
            self._get_stat_defs(stmt_info, rechecked_write_state_vars)  # 记录当前全局变量的修改位置

        self.stmts_var_info_maps[str(stmt_info.node_id)] = stmt_var_info

    def __stmt_call_send(self, stmt):

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
                # print("调用函数")
                pass

            else:  # 防止出现调用函数的情况
                self.transaction_stmts[str(stmt.node_id)] = {
                    "to": to,
                    "eth": eth,
                    "exp": stmt.expression.__str__()
                }
                print("=== 切片准则：{} at {}@{} ===".format(stmt.expression, self.name, stmt.node_id))
                print("发送以太币 {} 到 {}\n".format(eth, to))
                print("变量使用: {}".format(self.stmts_var_info_maps[str(stmt.node_id)]))

    def __if_loop_struct(self, stmt, stack):

        if stmt.type == NodeType.IF:
            stack.append(str(stmt.node_id))
            self.if_stmts.append(str(stmt.node_id))

        if stmt.type == NodeType.STARTLOOP:

            # begin_loop --> if_loop，寻找start_loop的父节点
            for suc_node_id in self.cfg.successors(str(stmt.node_id)):

                # NOTE 规避 node_id和function.nodes下标不一致的问题
                # 根据CFG_ID 找到function node的下标
                list_index = self.node_id_2_id[int(suc_node_id)]
                if self.function.nodes[list_index].type == NodeType.IFLOOP:
                    stack.append(str(suc_node_id))
                    self.if_stmts.append(str(suc_node_id))

        if stmt.type == NodeType.ENDIF or stmt.type == NodeType.ENDLOOP:
            if_start = stack.pop()
            if if_start not in self.if_paris:
                self.if_paris[if_start] = str(stmt.node_id)
            else:
                raise RuntimeError("IF END_IF 配对失败")

    def __loop_pair(self, stmt, cfg, remove_edges):

        if stmt.type == NodeType.IFLOOP:
            for pre_node_id in cfg.predecessors(str(stmt.node_id)):

                # IF_LOOP 的前驱节点中非 START_LOOP 的节点到IF_LOOP的边需要删除
                list_index = self.node_id_2_id[int(pre_node_id)]
                if self.function.nodes[list_index].type != NodeType.STARTLOOP:
                    remove_edges.append((pre_node_id, str(stmt.node_id)))

                    # 记录循环体的起止节点：循环执行的路径起止点
                    self.loop_stmts.append({"from": str(stmt.node_id), "to": pre_node_id})

    def __construct_simple_cfg(self, simple_cfg, remove_edges):

        # 删除循环边
        if len(remove_edges) != 0:
            simple_cfg.remove_edges_from(remove_edges)

        # 2: 给CFG中的所有叶子节点添加exit子节点作为函数退出的标识符
        leaf_nodes = []
        for cfg_node_id in simple_cfg.nodes:
            if simple_cfg.out_degree(cfg_node_id) == 0:  # 叶子节点列表
                leaf_nodes.append(cfg_node_id)
        # debug_get_graph_png(cfg, "cfg_noloop")

        simple_cfg.add_node("EXIT_POINT", label="EXIT_POINT")
        for leaf_node in leaf_nodes:
            simple_cfg.add_edge(leaf_node, "EXIT_POINT")

        self.simple_cfg = simple_cfg

    def _preprocess_function(self):

        simple_cfg = nx.DiGraph(self.cfg)

        # 局部信息
        stack = []
        remove_edges = []

        for id, stmt in enumerate(self.function.nodes):
            # 语句的变量使用情况
            self.__stmt_var_info(stmt)

            # msg.value语句
            self._get_msg_value_stmt(stmt)

            # 判断当前语句是否存在交易行为
            self.__stmt_call_send(stmt)

            # 匹配 (IF, END_IF) 和 (LOOP, END_LOOP)
            self.__if_loop_struct(stmt, stack)

            # 寻找(LOOP, END_LOOP), 并记录循环边到remove_edges
            self.__loop_pair(stmt, simple_cfg, remove_edges)

        # 简化原始cfg: 删除循环边, 添加exit节点
        self.__construct_simple_cfg(simple_cfg, remove_edges)

    def loop_body_extreact(self, criteria):
        """
        循环体执行
        for(循环条件){
            A ; criteria ;B ;C ;D
        }

        存在反向执行路径 <B, C, D, 循环条件, A, criteria>, 需要分析该路径的数据依赖关系，而B C D会对criteria造成影响
        """

        loop_reverse_paths = []
        for loop_structure in self.loop_stmts:

            src = loop_structure["from"]
            dst = loop_structure["to"]

            # start_loop --------- end_loop 执行轨迹
            cfg_paths = nx.all_simple_paths(self.simple_cfg, source=src, target=dst)
            for cfg_path in cfg_paths:

                for index, path_node in enumerate(cfg_path):
                    if path_node == str(criteria):
                        # a criteria b c d --> b c d criteria EXIT_POINT
                        loop_exe_path = cfg_path[index + 1:] + [path_node] + ["EXIT_POINT"]  # 将初始节点(切片准则)放在最后
                        loop_reverse_paths.append(loop_exe_path)
                        break

        return loop_reverse_paths

    #####################################################
    # 根据输入的全局变量名称，得到当前函数中修改该变量的语句信息集合 #
    ######################################################
    def get_state_var_def_stmts_info(self, state_var):

        state_var_related_stmts_infos = []

        if state_var in self.state_def_stmts:
            stmt_ids = self.state_def_stmts[state_var]

            for stmt_id in stmt_ids:
                var_info = self.stmts_var_info_maps[str(stmt_id)]
                current_node = self.function.nodes[stmt_id]
                state_var_related_stmts_infos.append({
                    "state_var": state_var,
                    "expression": current_node.expression.__str__(),
                    "type": current_node.type.__str__(),
                    "info": var_info,
                    "fun": self.function,
                    "func_name": self.name,
                    "node": current_node
                })

        return state_var_related_stmts_infos

    ######################################
    # 结构体赋值展开                        #
    ######################################
    def struct_assign_stmt_expand(self, external_stmt_info):

        structs_info = self.contract_info.structs_info
        state_var_declare_function_map = self.contract_info.state_var_declare_function_map

        node = external_stmt_info['node']
        for v in node.state_variables_read:
            if str(v) in state_var_declare_function_map \
                    and "full_expr" in state_var_declare_function_map[str(v)]:

                if str(v) not in self.const_var_init:
                    self.const_var_init[str(v)] = state_var_declare_function_map[str(v)]["full_expr"]

        struct_name = _new_struct(node, structs_info)  # 当前语句是否对结构体进行赋值
        if struct_name is not None:
            _, _, new_stmts = _new_struct_analyze(node, struct_name, structs_info)
            external_stmt_info["expand"] = new_stmts

    ######################################
    # 当前函数是否需要进一步分析              #
    ######################################
    def has_trans_stmts(self):
        return len(self.transaction_stmts)

    ######################################
    # 函数基本信息抽取                      #
    ######################################
    def function_info_analyze(self):
        self.cfg = self._get_function_cfg()
        self._get_node_id_2_cfg_id()
        self._preprocess_function()

    ######################################
    # 函数依赖图                           #
    ######################################
    def construct_dependency_graph(self):

        # 检查依赖
        if "ctrl_dep" not in self.semantic_edges:
            raise RuntimeError("ERROR: PDG缺少控制依赖")
        if "data_dep" not in self.semantic_edges:
            raise RuntimeError("ERROR: PDG缺少数据依赖")
        if "loop_data_dep" not in self.semantic_edges:
            raise RuntimeError("ERROR: PDG缺少循环体数据依赖")

        self.pdg = nx.MultiDiGraph(self.cfg)
        for semantic_type in self.semantic_edges:
            if semantic_type == "ctrl_dep" \
                    or semantic_type == "data_dep" \
                    or semantic_type == "loop_data_dep":
                self.pdg.add_edges_from(self.semantic_edges[semantic_type])

    #######################################
    # 获得所有的切片准则                     #
    #######################################
    def get_all_criterias(self):

        if self.append_criterias is None:
            raise RuntimeError("获得切片准则之前需要进行数据流分析")

        self.criterias = self.transaction_stmts
        self.criterias_msg = self.msg_value_stmt

    #######################################
    # 获得当前函数的图表示                    #
    #######################################
    def debug_png_for_graph(self, graph_type):

        if graph_type == "cfg":
            if self.cfg is not None:
                g = self.cfg
                # print("name = {}  {}".format(g.graph["name"], g.graph["contract_name"]))
                debug_get_graph_png(g, graph_type, cur_dir=True)
            else:
                raise RuntimeError("cfg 为空")

        if graph_type == "simple_cfg":
            if self.simple_cfg is not None:
                g = self.simple_cfg
                # print("name = {}  {}".format(g.graph["name"], g.graph["contract_name"]))
                debug_get_graph_png(g, graph_type, cur_dir=True)
            else:
                raise RuntimeError("simple_cfg 为空")

        if graph_type == "pdg":
            if self.pdg is not None:
                g = self.pdg
                # print("name = {}  {}".format(g.graph["name"], g.graph["contract_name"]))
                debug_get_graph_png(g, graph_type, cur_dir=True)
            else:
                raise RuntimeError("pdg 为空")

        if graph_type == "sliced_pdg":
            if len(self.sliced_pdg) > 0:
                for key in self.sliced_pdg:
                    g = self.sliced_pdg[key]
                    # print("name = {}  {}".format(g.graph["name"], g.graph["contract_name"]))
                    debug_get_graph_png(g, "{}_{}".format("spdg", key), cur_dir=True)
            else:
                raise RuntimeError("sliced_pdg 为空")

    #######################################
    # 显示某语句的变量使用情况                 #
    #######################################
    def debug_varinfo_for_stmt(self, cfg_id):

        stmt_id = self.id_2_node_id[cfg_id]
        expression = self.function.nodes[stmt_id]
        var_info = self.stmts_var_info_maps[str(stmt_id)]

        print("\n======DEBUG: VAR_INFO at {}======".format(cfg_id))
        print("语句：{}".format(expression))
        print("变量使用：{}".format(var_info))
        print("\n======DEBUG: VAR_INFO at {}======".format(cfg_id))
