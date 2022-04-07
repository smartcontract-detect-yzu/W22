import os
import subprocess
from typing import Dict, List
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot

from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from slither.core.declarations.function import Function
from slither.core.cfg.node import NodeType, Node


def _assign_to_zero(stmt_expression):
    # a = 0
    if " = " in stmt_expression:
        if "0" == stmt_expression.split(" = ")[1]:
            return True

    return False


def _get_real_parameters(expression: str):
    real_parameters = None

    temp1 = expression.split("(")[1].split(")")[0]
    if len(temp1) != 0:
        real_parameters = temp1.split(",")

    return real_parameters


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
        self.fid = function.id
        self.name: str = function.name.__str__()
        self.visibility = function.visibility

        # 函数调用链: 上级函数链 [[chain1], [chain2], [chain3]]
        self.callee_chains = None

        # 语义相关 初始信息
        self.input_params = {}  # 入参列表
        self.stmt_internal_call = {}  # 当前语句是否调用函数 <node_id, fid>
        self.if_stmts = []  # 条件语句列表
        self.vars_list = []  # 函数中使用的说有的变量列表
        self.stmts_var_info_maps = {}  # 各语句变量使用情况
        self.transaction_stmts = {}  # 存在交易行为的语句列表 transaction_stmts
        self.loop_stmts = []  # 循环语句列表
        self.if_paris = {}  # IF 与 END_IF
        self.node_id_2_idx = {}  # function.node.node_id -> node的idx function.nodes[idx]
        self.idx_2_node_id = {}  # function.node.node_id <- node的idx function.nodes[idx]
        self.state_def_stmts = {}  # 全局变量write(def)语句
        self.const_var_init = {}  # 涉及的全局变量初始化语句
        self.msg_value_stmt = {}  # 使用了msg.value的语句

        # 交易全局变量，需要通过数据流分析器获取
        self.transaction_states = None  # <交易语句, [交易涉及的全部全局变量]>

        # 函数内调用分析, cfg节点为调用外部函数
        self.intra_function_result_at_cfg = {}

        # 函数间：外部函数修改全局变量语句集合
        self.external_state_def_nodes_map = None

        # 切片准则
        self.criterias = None  # 交易语句
        self.criterias_msg = None  # msg.value
        self.append_criterias = None  # 交易相关全局变量，依赖数据流分析器 DataFlowAnalyzer
        self.external_criterias = None  # 过程间分析依赖的切片准则

        # 图表示
        self.cfg = None
        self.simple_cfg = None
        self.pdg = None

        # 切片后的图表示
        # key1: 交易语句ID，也就是criteria，无外部全局变量写和常数初始化
        # key2: tag@交易语句ID_with_外部写函数: 构建函数调用图时加入的
        self.sliced_pdg = {}

        # 不同的语义边
        self.semantic_edges = {}

        # 初始化
        self.function_info_analyze()

    def register_intra_fun(self, node_id, info):
        self.intra_function_result_at_cfg[node_id] = info

    def add_semantic_edges(self, semantic_type, edges: List):

        if semantic_type not in self.semantic_edges:
            self.semantic_edges[semantic_type] = edges
        else:
            for edge in edges:
                self.semantic_edges[semantic_type].append(edge)

    def get_callee_chain(self):
        return self.callee_chains

    def get_input_params(self):
        return self.input_params

    def get_sliced_pdg(self):
        return self.sliced_pdg

    def get_fid(self):
        return self.fid

    def get_transaction_stmts(self):
        return self.transaction_stmts

    def _get_msg_value_stmt(self, stmt):

        if "msg.value" in stmt.expression.__str__():
            self.msg_value_stmt[str(stmt.node_id)] = {
                "exp": stmt.expression.__str__()
            }

    def _get_function_cfg(self):

        cfg_dot_file = "{}_cfg.dot".format(self.function.name)
        self.function.cfg_to_dot(cfg_dot_file)

        cfg: nx.DiGraph = nx.drawing.nx_agraph.read_dot(cfg_dot_file)
        # os.remove(cfg_dot_file)
        cfg.graph["name"] = self.function.name
        cfg.graph["contract_name"] = self.contract_info.name

        for node in self.function.nodes:
            cfg_node = cfg.nodes[str(node.node_id)]
            cfg_node["expression"] = node.expression.__str__()
            if cfg_node["expression"] is None:
                cfg_node["expression"] = cfg_node["label"]
            cfg_node["type"] = node.type.__str__()
            cfg_node["fid"] = self.fid
            cfg_node["node_id"] = node.node_id

        return cfg

    def _get_node_id_2_cfg_id(self):

        for index, node in enumerate(self.function.nodes):

            # NOTE: 针对相同的语句 cfg_id == node.node_id != (index in function.nodes)
            # 如果需要直接通过function.nodes寻找node的信息，需要通过node_id_2_id转换
            self.node_id_2_idx[node.node_id] = index
            self.idx_2_node_id[index] = node.node_id
            if node.node_id != index:
                print("\033[0;31;40m\tNOTE: {} with ID: {} is at index:{}\033[0m".
                      format(node.expression.__str__(), node.node_id, index))
                pass

    def _get_stat_defs(self, stmt_info, write_state_vars):

        for stat_var in write_state_vars:

            if stat_var not in self.state_def_stmts:
                self.state_def_stmts[stat_var] = [stmt_info.node_id]
            else:
                self.state_def_stmts[stat_var].append(stmt_info.node_id)

    def _get_function_input_params(self):

        for idx, param in enumerate(self.function.parameters):
            print("入参特征： {} {} |  {}".format(param.type, param.name, param.expression))

            # 使用信息加入stmts_var_info_maps
            key = "{}_{}".format("input", idx)
            self.stmts_var_info_maps[key] = [{"list": [param.name], "type": "local", "op_type": "def"}]
            self.input_params[param.name] = {
                "type": param.type,
                "name": param.name,
                "id": idx,
                "key": key
            }

    def _add_input_params_to_cfg(self):
        """
        原始cfg: entry_point:"0" --> "second expression":"1"

        添加入参节点： entry_point:"0" --> input_param1:"input_1"  --> input_param2:"input_2" -->second expression:"1"
        """

        if len(self.input_params) == 0:
            return

        cfg: nx.DiGraph = self.cfg

        for param_name in self.input_params:
            input_param = self.input_params[param_name]
            cfg.add_node(input_param["key"],
                         label="INPUT {} {}".format(input_param["type"], input_param["name"]),
                         expression="{} {}".format(input_param["type"], input_param["name"]),
                         type="INPUT_PARAM")

        if len(list(cfg.neighbors("0"))) != 1:
            raise RuntimeError("ENTRY POINT 存在两个子节点？")
        else:
            second_node = list(cfg.neighbors("0"))[0]

        fist_id = current_id = None
        for param_name in self.input_params:
            input_param = self.input_params[param_name]
            if fist_id is None:
                fist_id = input_param["key"]
                last_id = input_param["key"]
                current_id = input_param["key"]
            else:
                last_id = current_id
                current_id = input_param["key"]

            if last_id != current_id:
                cfg.add_edge(last_id, current_id)

        cfg.add_edge("0", fist_id)
        cfg.add_edge(current_id, second_node)
        cfg.remove_edge("0", second_node)

    def _get_stmt_const_var_init(self, state_vars):

        state_declare = self.contract_info.state_var_declare_function_map
        for var in state_vars:
            if var in state_declare and "full_expr" in state_declare[var]:
                if var not in self.const_var_init:
                    self.const_var_init[var] = state_declare[var]["full_expr"]

    def __stmt_internal_call_info(self, stmt_info: Node):

        if len(stmt_info.internal_calls) != 0:

            called_infos = []
            for internal_call in stmt_info.internal_calls:
                if isinstance(internal_call, Function):
                    # print("====函数调用左边：{}".format([str(v) for v in stmt_info.variables_written]))
                    called_infos.append(internal_call.id)
                else:
                    # print("单纯内部调用{}".format(internal_call.name))
                    pass

            if len(called_infos) != 0:  # 当存在外部调用时

                expression = stmt_info.expression.__str__()
                real_params = _get_real_parameters(expression)
                assign_rets = [str(v) for v in stmt_info.variables_written]
                call_params_info = {
                    "real_params": real_params,
                    "assign_rets": assign_rets
                }

                print("\n外部调用信息：{}".format(stmt_info.expression.__str__()))
                print("实参：{}".format(real_params))
                print("返回值：{}".format(assign_rets))

                self.stmt_internal_call[stmt_info.node_id] = called_infos

                # 并将信息保存到cfg节点
                print("cfg id:{}  [{}] 内部调用：{} {}".format(str(stmt_info.node_id), self.name, internal_call.name,
                                                          internal_call.id))
                self.cfg.nodes[str(stmt_info.node_id)]["called"] = called_infos
                self.cfg.nodes[str(stmt_info.node_id)]["called_params"] = call_params_info

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

    def __get_all_vars_list(self):

        duplicate = {}

        for stmt_id in self.stmts_var_info_maps:
            stmt_var_infos = self.stmts_var_info_maps[stmt_id]
            for var_info in stmt_var_infos:
                if "list" in var_info:
                    for var in var_info["list"]:
                        if var not in duplicate:
                            duplicate[var] = 1
                            self.vars_list.append(var)

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
                # print("调用函数: {}".format(stmt.expression, stmt.cal))
                pass

            else:  # 防止出现调用函数的情况

                if to in self.const_var_init:  # 防止出现交易对象是常数的情况
                    ## print("交易对象是常数：{}".format(self.const_var_init[to]))
                    pass
                else:
                    self.transaction_stmts[str(stmt.node_id)] = {
                        "to": to,
                        "eth": eth,
                        "exp": stmt.expression.__str__()
                    }
                    print("\n=== 切片准则：{} at {}@{} ===".format(stmt.expression, self.name, stmt.node_id))
                    print("发送以太币 {} 到 {}\n".format(eth, to))
                    print("变量使用: {}".format(self.stmts_var_info_maps[str(stmt.node_id)]))
                    print("=============================================================\n")

    def __if_loop_struct(self, stmt, stack):

        if stmt.type == NodeType.IF:
            stack.append(str(stmt.node_id))
            self.if_stmts.append(str(stmt.node_id))

        if stmt.type == NodeType.STARTLOOP:

            # begin_loop --> if_loop，寻找start_loop的父节点
            for suc_node_id in self.cfg.successors(str(stmt.node_id)):

                # 根据CFG_ID 找到function node的下标, 并找到对应的节点
                target_node = self.get_node_info_by_node_id_from_function(int(suc_node_id))
                if target_node.type == NodeType.IFLOOP:
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
                target_node = self.get_node_info_by_node_id_from_function(int(pre_node_id))
                if target_node.type != NodeType.STARTLOOP:
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

    def _get_call_chain(self):
        self.callee_chains = self.contract_info.get_call_chain(self.function)
        # for call_chain in call_chains:
        #     for callee in call_chain:
        #         callee_fid = callee["fid"]
        #         if callee_fid is not None:
        #             callee_function = self.contract_info.get_function_by_fid(callee_fid)

    def _preprocess_function(self):

        simple_cfg = nx.DiGraph(self.cfg)

        # 局部信息
        stack = []
        remove_edges = []

        for idx, stmt in enumerate(self.function.nodes):
            # 语句的变量使用情况
            self.__stmt_var_info(stmt)

            # 语句是否进行接口调用
            self.__stmt_internal_call_info(stmt)

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

        # 向contract结构注册函数的切片
        self._register_slice_infos()

    def _register_slice_infos(self):

        slices_infos = []
        contract_name = self.contract_info.name
        function_name = self.name
        name_prefix = "{}_{}".format(contract_name, function_name)

        for criteria in self.transaction_stmts:
            name = "{}_{}".format(name_prefix, criteria)
            exp = self.function.nodes[self.node_id_2_idx[int(criteria)]].expression.__str__()
            if self.contract_info.duplicate_slice(name) is not True:
                slices_infos.append({"name": name, "exp": exp})

        self.contract_info.load_slices_infos(slices_infos)

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
    # 根据cfg id（等价于node_id） 获得function.nodes的信息#
    ######################################################
    def get_node_info_by_node_id_from_function(self, node_id):
        idx = self.node_id_2_idx[node_id]
        return self.function.nodes[idx]

    #####################################################
    # 根据输入的全局变量名称，得到当前函数中修改该变量的语句信息集合 #
    ######################################################
    def get_state_var_def_stmts_info(self, state_var):

        state_var_related_stmts_infos = []
        state_declare_info = self.contract_info.state_var_declare_function_map
        if state_var in self.state_def_stmts:
            stmt_ids = self.state_def_stmts[state_var]

            for stmt_id in stmt_ids:

                expr = self.function.nodes[stmt_id].expression.__str__()
                if _assign_to_zero(expr):
                    # 赋值为0
                    continue

                vars_infos = self.stmts_var_info_maps[str(stmt_id)]
                const_init = {}
                for vars_info in vars_infos:
                    if vars_info["type"] == 'state' and vars_info["op_type"] == 'use':
                        for v in vars_info["list"]:
                            if str(v) in state_declare_info and "full_expr" in state_declare_info[str(v)]:
                                const_init[str(v)] = state_declare_info[str(v)]["full_expr"]

                current_node = self.function.nodes[stmt_id]
                print("外部写全局变量语句：{}".format(current_node.expression.__str__()))
                state_var_related_stmts_infos.append({
                    "state_var": state_var,
                    "expression": current_node.expression.__str__(),
                    "type": current_node.type.__str__(),
                    "info": vars_infos,
                    "const_init": const_init,
                    "fun": self.function,
                    "func_name": self.name,
                    "node": current_node
                })

        return state_var_related_stmts_infos

    ######################################
    # 结构体赋值展开                        #
    ######################################
    def struct_assign_stmt_expand(self, external_stmt_info):
        """
        将结构体赋值语句展开:
        ==原始语句:
            queue.push(Deposit(msg.sender,uint128(msg.value),uint128(msg.value * MULTIPLIER / 100)))
        ==展开后:
            Deposit.depositor = msg.sender
            Deposit.deposit = msg.value
            Deposit.expect = msg.value * MULTIPLIER / 100
            queue.push(Deposit)
        """

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

        print("【START:函数预处理】函数预处理：{}".format(self.name))

        self.cfg = self._get_function_cfg()

        # 将函数入参作为语义补充到原始cfg中
        self._get_function_input_params()
        self._add_input_params_to_cfg()

        self.debug_png_for_graph("cfg")

        self._get_node_id_2_cfg_id()
        self._preprocess_function()
        self._get_call_chain()
        self.contract_info.function_info_map[self.fid] = self

        print("【END:函数预处理】函数预处理：{}".format(self.name))
    ######################################
    # 函数依赖图                           #
    ######################################
    def construct_dependency_graph(self):

        # 检查依赖
        if "ctrl_dep" not in self.semantic_edges:
            raise RuntimeError("ERROR: PDG缺少控制依赖")
        if "data_dep" not in self.semantic_edges:
            raise RuntimeError("ERROR: PDG缺少数据依赖")

        self.pdg = nx.MultiDiGraph(self.cfg)
        for semantic_type in self.semantic_edges:
            if semantic_type == "ctrl_dep" or semantic_type == "data_dep":
                self.pdg.add_edges_from(self.semantic_edges[semantic_type])

    #######################################
    # 获得所有的切片准则                     #
    #######################################
    def get_all_internal_criterias(self):

        if self.append_criterias is None:
            raise RuntimeError("获得切片准则之前需要进行数据流分析")

        self.criterias = self.transaction_stmts
        self.criterias_msg = self.msg_value_stmt

    def get_criterias_by_criteria_content(self, criteria_content, criteria_type):

        external_criterias = []

        if criteria_type == "external":
            for stmt_id in self.external_criterias:
                external_criteria = self.external_criterias[stmt_id]
                if external_criteria["external_call"] == criteria_content:
                    external_criterias.append(stmt_id)

        return external_criterias

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

        stmt_id = self.idx_2_node_id[cfg_id]
        expression = self.function.nodes[stmt_id]
        var_info = self.stmts_var_info_maps[str(stmt_id)]

        print("\n======DEBUG: VAR_INFO at {}======".format(cfg_id))
        print("语句：{}".format(expression))
        print("变量使用：{}".format(var_info))
        print("\n======DEBUG: VAR_INFO at {}======".format(cfg_id))

    #######################################
    # 判断指定的切片准则内容在函数中的位置           #
    #######################################
    def get_external_criteria_by_content(self, criteria_content):

        for idx, stmt in enumerate(self.function.nodes):

            if stmt.can_send_eth() and criteria_content in stmt.expression.__str__():

                if self.external_criterias is None:
                    self.external_criterias = {}

                self.external_criterias[str(stmt.node_id)] = {
                    "external_call": criteria_content,
                    "exp": stmt.expression.__str__()
                }
