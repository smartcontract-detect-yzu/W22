from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from ponzi_detector.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer


# 过程间分析器
class InterproceduralAnalyzer:

    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo):
        self.function_info = function_info
        self.contract_info = contract_info

        # 修改全局变量语句集合
        self.external_state_def_nodes_map = None

    def do_interprocedural_state_analyze(self):
        """
        Parameters:
        this_function -- 当前函数的信息
        transaction_states -- 当前函数依赖中交易行为依赖的全局变量
        state_var_write_function_map -- 当前智能合约中全局变量修改操作和函数的对应表
        state_var_declare_function_map -- 当前智能合约中全局变量声明操作和函数的对应表

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
        external_nodes_map = {}  # <key:交易语句, value:外部函数语句>

        for trans_criteria in transaction_states:  # 交易相关全局变量作为准则

            external_nodes_map[trans_criteria] = []
            for transaction_state in transaction_states[trans_criteria]:
                for trans_stat_var in transaction_state["vars"]:
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

                        for write_fun in write_funs:

                            if write_fun.full_name == this_function.full_name:
                                continue  # 过滤非当前函数

                            write_func_info = FunctionInfo(self.contract_info, write_fun)

                            # 记录下该函数中修改current_var的语句
                            def_var_infos = write_func_info.get_state_var_def_stmts_info(current_var)
                            for info in def_var_infos:

                                self.function_info.struct_assign_stmt_expand(info)

                                external_nodes_map[trans_criteria].append(info)

                                # 这些语句又使用了那些全局变量来修改current_var, 进行下一次的分析
                                for var_info in info["info"]:
                                    if var_info["type"] == "state" and var_info["op_type"] == "use":  # 只需要使用的全局变量
                                        for var in var_info["list"]:
                                            if var not in duplicate:
                                                # print("\t\t下一个变量{}".format(var))
                                                duplicate[var] = 1
                                                stack.append(var)

        self.external_state_def_nodes_map = external_nodes_map
        self.function_info.external_state_def_nodes_map = external_nodes_map
        return external_nodes_map