import argparse
import os
import shutil
import subprocess
from ponzi_detector.info_analyze.contract_analyze import ContractInfo
from ponzi_detector.info_analyze.function_analyze import FunctionInfo
from ponzi_detector.semantic_analyze.control_flow_analyzer import ControlFlowAnalyzer
from ponzi_detector.semantic_analyze.data_flow_analyzer import DataFlowAnalyzer
from ponzi_detector.semantic_analyze.code_graph_constructor import CodeGraphConstructor
from slither import Slither

versions = ['0', '0.1.7', '0.2.2', '0.3.6', '0.4.26', '0.5.17', '0.6.12', '0.7.6', '0.8.6']


def _select_solc_version(version_info):
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


# 文件分析器
class SolFileAnalyzer:
    def __init__(self, file_name: str, path: str):

        self.file_name = file_name
        self.work_path = path

        self.solc_version = None

    def parse_solc_version(self):

        with open(self.file_name, 'r', encoding='utf-8') as contract_code:

            mini = 100
            version_resault = None

            for line in contract_code:
                target_id = line.find("pragma solidity")
                if target_id != -1:
                    new_line = line[target_id:]
                    version_info = new_line.split("pragma solidity")[1]
                    v = _select_solc_version(version_info)

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

            if version_resault is None:
                version_resault = "0.4.26"

            self.solc_version = version_resault
            return version_resault
