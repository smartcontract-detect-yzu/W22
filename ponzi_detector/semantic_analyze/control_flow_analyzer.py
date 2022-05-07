# 控制流分析器
import networkx as nx

from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from slither.core.cfg.node import NodeType


# 控制流分析器
class ControlFlowAnalyzer:

    def __init__(self, contract_info: ContractInfo, function_info: FunctionInfo):

        self.function_info = function_info
        self.contract_info = contract_info

        self.predom_relations = None

        self.control_dep_relations = None
        self.control_dep_edges = None

    def _get_predom_relations(self):
        """
        利用前向支配关系生成控制流依赖
        1.生成前向支配关系
        2.控制流依赖关系计算
        """

        simple_cfg = self.function_info.simple_cfg

        # 前向支配关系生成，reverse_cfg的入口为 "EXIT_POINT"
        reverse_cfg = simple_cfg.reverse()
        predom_relations = nx.algorithms.immediate_dominators(reverse_cfg, "EXIT_POINT")
        del predom_relations["EXIT_POINT"]  # 删除EXIT_POINT，因为这是虚拟节点
        self.predom_relations = predom_relations
        return predom_relations

    def _get_control_dependency_relations_by_prdom(self):

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

        cfg = self.function_info.simple_cfg
        if_stmts = self.function_info.if_stmts
        predom_relations = self.predom_relations

        control_dep_relations_dup = {}

        control_dep_relations = []
        for x in if_stmts:
            predom_node = predom_relations[x]
            cfg_paths = nx.all_simple_paths(cfg, source=x, target="EXIT_POINT")

            for cfg_path in list(cfg_paths):

                for y in cfg_path[1:-1]:

                    # 获得当前节点的function.node信息
                    node_info = self.function_info.get_node_info_by_node_id_from_function(int(y))

                    # 虚拟节点暂时不进行控制依赖分析
                    if node_info.type != NodeType.ENDIF \
                            and node_info.type != NodeType.ENDLOOP \
                            and node_info.type != NodeType.STARTLOOP:

                        if y != predom_node:  # does’t contain the immediate forward dominator
                            key = "{}-{}".format(y, x)  # y控制依赖于x
                            if key not in control_dep_relations_dup:
                                control_dep_relations_dup[key] = 1
                                length = nx.shortest_path_length(cfg, x, y)
                                control_dep_relations.append({'from': y, "to": x, 'color': 'red', 'distance': length})

        # NOTE: 就近原则，永远依赖于较近的那个
        control_dep_relations_by_distance = {}
        for cdg_edge in control_dep_relations:
            from_node = cdg_edge["from"]
            to_node = cdg_edge["to"]
            distance = cdg_edge["distance"]

            # NOTE: 就近原则，永远依赖于较近的那个
            if from_node not in control_dep_relations_by_distance:
                control_dep_relations_by_distance[from_node] = {"to": to_node, "distance": distance}
            else:
                old_distance = control_dep_relations_by_distance[from_node]["distance"]
                if old_distance > distance:
                    control_dep_relations_by_distance[from_node] = {"to": to_node, "distance": distance}

        self.control_dep_relations = control_dep_relations_by_distance
        return control_dep_relations_by_distance

    def _control_dependency_fliter(self, from_node, to_node):
        """

        情况1：
        过滤掉终止的控制流依赖
        if(a) -- A -- end_if -- B
        此时 B 并不控制依赖于 if(a)


        情况2：
        if(a) -- A -- end_if -- B
              -- c --    d   --
        此时 B 控制依赖于 if(a)
        """

        if_paris = self.function_info.if_paris
        simple_cfg = self.function_info.simple_cfg

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

    def do_control_dependency_analyze(self):
        """
        利用前向支配关系生成控制流依赖
        1.生成前向支配关系
        2.控制流依赖关系计算
        """

        # 根据simple cfg获得前向支配关系
        self._get_predom_relations()

        # 根据cfg前向支配关系获得控制依赖关系
        self._get_control_dependency_relations_by_prdom()

        cdg_edges = []
        for from_node in self.control_dep_relations:
            to_node = self.control_dep_relations[from_node]["to"]

            # NOTE: 控制依赖边方向和控制流边方向相反（if(A) {B}: B -依赖于->A）
            if not self._control_dependency_fliter(from_node, to_node):
                cdg_edges.append((from_node, to_node, {'color': "red", "type": "ctrl_dependency"}))
            else:
                # print("过滤控制流边{}-{}".format(from_node, to_node))
                pass

        print("控制依赖边数量：{}".format(len(cdg_edges)))
        self.control_dep_edges = cdg_edges  # 结果保存在当前控制流分析器中
        self.function_info.semantic_edges["ctrl_dep"] = self.control_dep_edges  # 结果保存在当前函数信息中

    ###############################
    #  生成当前函数的控制依赖边        #
    ###############################
    def get_control_dependency_edges(self):

        if self.control_dep_edges is None:
            raise RuntimeError("控制依赖关系未分析")

        return self.control_dep_edges
