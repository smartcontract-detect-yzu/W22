import os
import subprocess
from typing import Dict
import networkx as nx
import networkx.drawing.nx_pydot as nx_dot

from slither.core.declarations import Contract
from slither.core.declarations.function import Function
from slither.core.expressions import Identifier

EXAMPLE_PERFIX = "examples/ponzi/"


class ContractInfo:

    def __init__(self, contract: Contract):
        self.name = contract.name
        self.contract = contract

        self.function_info_map = {}

        # 当合约包含的切片内容
        self.dup_sliced_infos = {}  # 去重
        self.sliced_infos = []

        # 合约内部定义的结构体信息 <结构体名称, StructureContract>
        self.structs_info = {}

        # 直接调用 <>.send<> 和 <>.transfer<>接口的函数
        self.send_function_map = {}

        # 合约函数调用图
        self.funcid_2_graphid = {}  # <key:function.id value: call_graph.node_id>
        self.fid_2_function: Dict[str, Function] = {}
        self.call_graph = None

        # 全局变量 <--> 函数对应关系表
        self.state_var_declare_function_map = {}  # <全局变量名称, slither.function>
        self.state_var_read_function_map = {}  # <全局变量名称, slither.function>
        self.state_var_write_function_map = {}  # <全局变量名称, slither.function>

        # 初始化: 合约信息抽取
        self.contract_info_analyze()

    def _struct_info(self):

        # 结构体定义信息抽取
        for structure in self.contract.structures:
            self.structs_info[structure.name] = structure

    def _stat_vars_info_in_contract(self):

        # 全局变量定义
        for v in self.contract.state_variables:
            if v.expression is None:
                exp = str(v.type) + " " + str(Identifier(v))
                self.state_var_declare_function_map[str(Identifier(v))] = {"type": str(v.type), "exp": exp}

        for function in self.contract.functions:

            # print("[列表] 函数名：{}  fid:{}".format(function.name, function.id))
            self.fid_2_function[function.id] = function

            # 全局变量定义
            if function.is_constructor or function.is_constructor_variables:
                for node in function.nodes:
                    for v in node.state_variables_written:
                        full_exp = "{} {}".format(str(v.type), node.expression)
                        self.state_var_declare_function_map[str(v)] = {
                            "fun": function,
                            "expr": node.expression,
                            "full_expr": full_exp
                        }

            else:
                # 全局变量读
                for v in function.state_variables_read:
                    if str(v) not in self.state_var_read_function_map:
                        self.state_var_read_function_map[str(v)] = [function]
                    else:
                        self.state_var_read_function_map[str(v)].append(function)

                # 全局变量写
                for v in function.state_variables_written:

                    if not function.can_send_eth():
                        continue  # NOTE:对于参与交易的函数，下面会进行重点分析

                    if str(v) not in self.state_var_write_function_map:
                        self.state_var_write_function_map[str(v)] = [function]
                    else:
                        self.state_var_write_function_map[str(v)].append(function)

    def _functions_with_transaction_call(self):

        for function in self.contract.functions:
            for node in function.nodes:
                if ".transfer(" in str(node.expression) or ".send(" in str(node.expression):
                    if function.name not in self.send_function_map:
                        self.send_function_map[function.id] = {
                            "id": function.id,
                            "name": function.name,
                            "function": function,
                            "exp": node.expression,
                            "node": node
                        }

    def _construct_call_graph(self):

        # 函数调用图
        call_graph = nx.DiGraph()
        call_graph.graph["name"] = self.contract.name
        edges = []
        duplicate = {}

        node_id = 0
        for function in self.contract.functions:

            if function.id not in self.funcid_2_graphid:
                call_graph.add_node(node_id, label=function.name, fid=function.id)
                self.funcid_2_graphid[function.id] = node_id
                node_id += 1

            from_node = self.funcid_2_graphid[function.id]
            for internal_call in function.internal_calls:

                if isinstance(internal_call, Function):
                    if internal_call.id not in self.funcid_2_graphid:
                        call_graph.add_node(node_id, label=internal_call.name, fid=internal_call.id)
                        self.funcid_2_graphid[internal_call.id] = node_id
                        node_id += 1

                    to_node = self.funcid_2_graphid[internal_call.id]
                    if "{}-{}".format(from_node, to_node) not in duplicate:
                        duplicate["{}-{}".format(from_node, to_node)] = 1
                        edges.append((from_node, to_node))
        call_graph.add_edges_from(edges)

        entry_nodes = []
        for node in call_graph:
            if call_graph.in_degree(node) == 0:
                entry_nodes.append(node)

        # 添加虚拟entry_node, 方便寻找call_chain
        call_graph.add_node("entry_node", label="entry_node", fid="entry_node")
        for entry_node in entry_nodes:
            call_graph.add_edge("entry_node", entry_node)

        self.call_graph = call_graph

    def duplicate_slice(self, key):
        if key not in self.dup_sliced_infos:
            self.dup_sliced_infos[key] = 1
            return False
        else:
            return True

    def load_slices_infos(self, infos):

        self.sliced_infos += infos

    def contract_info_analyze(self):
        self._struct_info()  # 获得结构体信息
        self._stat_vars_info_in_contract()  # 获得结构体信息
        self._functions_with_transaction_call()

        self._construct_call_graph()  # 函数调用图
        # self.debug_get_call_graph()

    def debug_get_call_graph(self):

        if self.call_graph is not None:
            graph = self.call_graph
            dot_name = "{}_{}.dot".format(graph.graph["name"], "call_graph")
            cfg_name = "{}_{}.png".format(graph.graph["name"], "call_graph")
            nx_dot.write_dot(graph, dot_name)

            subprocess.check_call(["dot", "-Tpng", dot_name, "-o", cfg_name])
            os.remove(dot_name)
        else:
            raise RuntimeError("call_graph 为空")

    def debug_stat_var_info(self):

        print(u"===全局变量定义信息：")
        for var in self.state_var_declare_function_map:
            print("\t定义变量{}".format(str(var)))

            if "exp" in self.state_var_declare_function_map[var]:
                print("\t\t{}".format(self.state_var_declare_function_map[var]["exp"]))

            if "fun" in self.state_var_declare_function_map[var]:
                print("\t\t{}".format(self.state_var_declare_function_map[var]["full_expr"]))

        print("===全局变量读信息：")
        for var in self.state_var_read_function_map:

            print("读变量{}".format(str(var)))
            for func in self.state_var_read_function_map[var]:
                print("\t{}".format(func.name))

        print("===全局变量写信息：")
        for var in self.state_var_write_function_map:

            print("写变量{}".format(str(var)))
            for func in self.state_var_write_function_map[var]:
                print("\t{}".format(func.name))

    def get_call_chain(self, target_function: Function):

        ret_call_chain = []

        target_id = self.funcid_2_graphid[target_function.id]
        call_chains = nx.all_simple_paths(self.call_graph, source="entry_node", target=target_id)
        for call_chain in list(call_chains):

            chain_info = []
            # print("call chain:", call_chain)
            for node in call_chain:
                if node != "entry_node":
                    # print("cont :{}".format(self.call_graph.nodes[node]))
                    chain_info.append(self.call_graph.nodes[node])

            ret_call_chain.append(chain_info)
        return ret_call_chain

    def get_function_by_fid(self, fid: str):
        """
        如果被调用函数时modify
        则没有fid
        """
        if fid in self.fid_2_function:
            return self.fid_2_function[fid]
        else:
            return None

    def get_function_info_by_fid(self, fid: str):

        if fid in self.function_info_map:
            return self.function_info_map[fid]

        return None

    def get_contract_info(self):

        return self.sliced_infos
