import json
import re

from slither import Slither

from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer
from slither.core.declarations import (
    Function,
    SolidityFunction,
    Contract,
    SolidityVariable, FunctionContract,
)
from slither.core.variables.variable import Variable
from slither.core.solidity_types.type import Type
from slither.core.solidity_types.user_defined_type import UserDefinedType

EXAMPLE_PERFIX = "../../example/"


def analyze():
    file_name = "0x06a566e7812413bc66215b48d6f26321ddf653a9.sol"
    file_analyzer = SolFileAnalyzer(file_name=file_name, work_path=EXAMPLE_PERFIX)

    file_analyzer.do_chdir()
    file_analyzer.do_file_analyze_prepare()  # 环境配置
    file_infos = file_analyzer.do_analyze_a_file_for_vul3()


def convert_safemath(stmt, safe_math_ops):
    print("开始分析......safemath")
    for ex_exp, op in zip(reversed(stmt.external_calls_as_expressions), safe_math_ops):
        print("expr:{} op:{}".format(ex_exp, op))

        p = re.compile(r'\.{}[(](.*?)[)]'.format(op))
        result = re.findall(p, str(ex_exp))
        print(result)


def test():
    slither = Slither(EXAMPLE_PERFIX + '0x06a566e7812413bc66215b48d6f26321ddf653a9.sol')
    target_functions = {}

    # 1.寻找需要分析的目标函数：visibility为external的函数，表面该函数可以被调用
    # 过滤规则: is_implemented && external/visibility && non-constructor && non-view
    for contract in slither.contracts:
        for function in contract.functions:
            if function.is_implemented \
                    and function.visibility in ["public", "external"] \
                    and not function.view \
                    and function.name != "constructor":
                target_functions[function.id] = function
                print("需要分析的函数  name:{} id:{} ".format(function.name, function.id, ))

    # 2. 解析当前函数使用的安全最佳实践语句
    # 2.1 modifier
    # 2.2 math/safeTrans
    """
        stmt.library_calls -- 调用库合约的接口
        stmt.high_level_calls -- 调用外部函数（函数间调用关系）
        stmt.internal_calls -- 
        stmt.low_level_calls -- 
        stmt.solidity_calls -- 
    """
    for f_id in target_functions:
        function = target_functions[f_id]
        # print("\n==========开始分析函数：{}============".format(function.name))
        # for modifier in function.modifiers:
        #     if modifier.name in ["nonReentrant", "onlyRole", "onlyOwner"]:
        #         print("函数：{} 包含最佳实践 {}".format(function.name, modifier.name))

        for stmt in function.nodes:
            # print("【AST-ID:{}】".format(stmt.node_ast_id), stmt.expression)

            # for in_call in stmt.internal_calls:
            #     if isinstance(in_call, Function):
            #         print("----【in_call:Function】", in_call.name, in_call.id)
            #
            #     if isinstance(in_call, SolidityFunction):
            #         print("----【in_call:SolidityFunction】", in_call.name, in_call.name)
            #
            # for lib_call in stmt.library_calls:
            #     if isinstance(lib_call, Function):
            #         print("****【lib_call:Function】", lib_call.name)
            #     if isinstance(lib_call, Contract):
            #         print("****【lib_call:Contract】", lib_call.name)

            # safe_math_ops = []
            # for h_call in stmt.high_level_calls:
            #     if h_call[0].name == "SafeMath":
            #         safe_math_ops.append(h_call[1].name)
            #         if isinstance(h_call[1], Function):
            #             print("^^^^【h_call:Function】", h_call[0].name, " ", h_call[1].name)

            if 2351 == stmt.node_ast_id:
                print("============================")
                for h_call in stmt.high_level_calls:
                    if isinstance(h_call[1], Function):
                        print("^^^^【h_call:Function】", h_call[0].name, " ", h_call[1].name)
                    if isinstance(h_call[1], Contract):
                        print("^^^^【h_call:Contract】", h_call[0].name, " ", h_call[1].name)
                for ext_call in stmt.external_calls_as_expressions:
                    print("^^^^【ext_call】", ext_call)
                print("============================")
            # if len(safe_math_ops) != 0:
            #     convert_safemath(stmt, safe_math_ops)

            # for external_call in stmt.external_calls_as_expressions:
            #     print("@@@@【external_call】", external_call)


if __name__ == '__main__':
    test()

    # can_send_eth = {}
    # slither = Slither(EXAMPLE_PERFIX + '0x06a566e7812413bc66215b48d6f26321ddf653a9.sol')
    # for c in slither.contracts:
    #     for f in c.functions:
    #         if f.can_send_eth():
    #             can_send_eth[f.id] = f.name
    #         else:
    #             for lc in f.low_level_calls:
    #                 if lc[1] == "call":
    #                     can_send_eth[f.id] = f.name
    # 
    # for c in slither.contracts:
    #     for f in c.functions:
    #         for in_call in f.internal_calls:
    #             if isinstance(in_call, Function):
    #                 if in_call.id in can_send_eth:
    #                     can_send_eth[f.id] = f.name
    # print(json.dumps(can_send_eth))
    # 
    # for c in slither.contracts:
    #     for f in c.functions:
    #         for md in f.modifiers:
    #             if md.name == "nonReentrant":
    #                 print("func with nonReentrant:{}".format(f.name))
    #                 for stmt in f.nodes:
    #                     for lb_call in stmt.library_calls:
    #                         print("-- 库调用 合约: {} {}".format(lb_call[0].name, lb_call[0].id))
    #                         print("-- 库调用 函数: {} {}".format(lb_call[1].name, lb_call[1].id))
    # 
    #                 for md in f.modifiers:
    #                     if md.name != "nonReentrant":
    #                         print("modifier: {} fn:{} id:{}".format(md.name, md.full_name, md.id))
    # 
    #                         for node in md.nodes:
    #                             print("{} @ {} {} {}".format(node._ast_id, node.node_id, node.expression, node.type))
    # 
    # tag = 0
    # if tag == 1:
    #     slither = Slither(EXAMPLE_PERFIX + '0x06a566e7812413bc66215b48d6f26321ddf653a9.sol')
    #     # slither = Slither(EXAMPLE_PERFIX + '0x2103021FDd7cDA7A747007Dbb71F248630De5aA5/AIDeGods.sol')
    #     for c in slither.contracts:
    #         print("c: {} ck:{}".format(c.name, c.contract_kind))
    # 
    #     for contract in slither.contracts:
    # 
    #         if tag == 1 and contract.name == "SafeERC20":
    #             print("==========================")
    #             for l_call in contract.all_library_calls:
    #                 print(
    #                     "****** 外部调用：contract:{} function:{} f_id:{}".format(l_call[0].name, l_call[1].name,
    #                                                                          l_call[1].id))
    #             print("+++++++++++++++++++++++++++")
    # 
    #             m_d = contract.available_modifiers_as_dict()
    #             for m_n in m_d:
    #                 print("****** 保护 modifier:{} {}".format(m_n, m_d[m_n]))
    # 
    #             for f_i in contract.functions_inherited:
    #                 print(f_i.name)
    # 
    #             print("---------------------------")
    # 
    #             for list_libs in contract.using_for.values():
    #                 for lib_candidate_type in list_libs:
    #                     if isinstance(lib_candidate_type, UserDefinedType):
    #                         lib_candidate = lib_candidate_type.type
    #                         if isinstance(lib_candidate, Contract):
    #                             print("adasda:{} {}".format(lib_candidate.name, lib_candidate.id))
    #             print("hhhhhhhhhhhhhhhhhhhhhhhhh")
    # 
    #             for f in contract.functions:
    # 
    #                 print("开始进行函数解析：{}".format(f.name))
    #                 for f_l_c in f.all_library_calls():
    #                     print("--当前函数的外部调用 c:{} f:{}".format(f_l_c[0].name, f_l_c[1].name))
    # 
    #                 for in_call_f in f.internal_calls:
    #                     if isinstance(in_call_f, Function):
    #                         print("--incall:{} {}".format(in_call_f.name, in_call_f.id))
    #                     if isinstance(in_call_f, SolidityFunction):
    #                         print("--incall:{} {}".format(in_call_f.name, in_call_f.full_name))
    # 
    #                 for expr in f.nodes:
    #                     print("----- 当前语句能否发射：{}".format(expr.can_send_eth()))
    #                     for e_lb_call in expr.library_calls:
    #                         print("----********当前语句依赖库 c:{} f:{} s:{}".format(e_lb_call[0].name, e_lb_call[1].name,
    #                                                                           e_lb_call[1].can_send_eth()))

    # for function in contract.functions:
    #     if function.is_implemented:
    #         if tag == 1 and contract.name == "SafeERC20" and function.name == "safeTransfer":
    #             print("11111111")
    #             print(function.name, function.id)
    #             for expr in function.external_calls_as_expressions:
    #                 print("expr  ", expr)
    #             for in_call in function.internal_calls:
    #                 print("in_call  ", in_call)
    #             print("222222222")
    #         if tag == 1 and contract.name == "Gauge" and function.name == "_deposit":
    #             print(function.name, function.id)
    #             function.cfg_to_dot("_deposit.dot")
    #             for in_call in function.internal_calls:
    #                 print("--in_call  ", in_call)
    #             for node in function.nodes:
    #                 print(node.expression, node.node_id)
    #                 for call in node.internal_calls:
    #                     print("--internal--", call)
    #                 for e_call_e in node.external_calls_as_expressions:
    #                     print("--external--", e_call_e)
    #                 for l_call in node.library_calls:
    #                     if isinstance(l_call, Function):
    #                         print("--l_c_f--", l_call.name)
    #                     elif isinstance(l_call, Contract):
    #                         print("--l_c_c--", l_call.name)
    #
    #             print("3123123123123123123123123")
    #
    #         if function.can_send_eth():
    #             print("eeeeeeeeeeeeeeeetttttttttttttttttttt", function.name)
    #         else:
    #             for lc in function.low_level_calls:
    #                 print("eeeeeeeeeeeeeeeetttttttttttttttttttt", lc[0], lc[1])
    #                 if lc[1] == "call":
    #                     print(function.name)
