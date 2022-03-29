import argparse
import json
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
    def __init__(self, file_name: str, work_path: str):

        self.file_name = file_name
        self.work_path = work_path
        self.pwd = None

        self.opcode_file = file_name.split(".sol")[0] + ".evm"

        self.contract_opcode_files = {}

        self.contract_asm_files = {}

        self.opcode_cnt = {}
        self.opcode_total_cnt = 0
        self.opcode_frequency = {}

        self.solc_version = None

    def do_file_analyze_prepare(self):

        self._parse_solc_version()  # 解析编译器版本 必须在工作目录下

        self._select_compiler()  # 更改编译器版本

    def do_chdir(self):
        self.pwd = os.getcwd()
        os.chdir(self.work_path)  # 切换工作目录

    def revert_chdir(self):
        os.chdir(self.pwd)  # 切换工作目录

    def get_opcode_and_asm_file(self):

        if os.path.exists("asm_done.txt"):
            return

        with open(self.opcode_file, 'w+') as f:
            subprocess.check_call(["solc", "--bin-runtime", self.file_name], stdout=f)

        with open(self.opcode_file, 'r') as f:

            key = "{}:".format(self.file_name)

            code_flag = 0
            for line in f.readlines():

                if code_flag == 1:
                    code_cnt -= 1

                    if code_cnt == 0:
                        code_flag = 0

                        with open("{}.bin".format(contract_name), 'w+') as contract_op_file:
                            contract_op_file.write(line)

                        # 去除.old_asm文件第一行,写入.asm文件
                        with open("{}.old_asm".format(contract_name), 'w+') as asm_file:
                            subprocess.call(["evm", "disasm", "{}.bin".format(contract_name)], stdout=asm_file)
                        with open("{}.old_asm".format(contract_name), 'r') as fin:
                            data = fin.read().splitlines(True)
                        with open("{}.asm".format(contract_name), 'w') as fout:
                            fout.writelines(data[1:])
                        os.remove("{}.old_asm".format(contract_name))

                        self.contract_opcode_files[contract_name] = "{}.bin".format(contract_name)
                        self.contract_asm_files[contract_name] = "{}.asm".format(contract_name)

                elif key in line:
                    file_contract = line.split("=======")[1][1:-1]
                    contract_name = file_contract.split(":")[1]
                    print(contract_name)
                    code_flag = 1
                    code_cnt = 2  # 空两行

        with open("asm_done.txt", "w+") as f:
            f.write("1")

        return

    def _select_compiler(self):
        print("========={} V: {}".format(self.file_name, self.solc_version))
        subprocess.check_call(["solc-select", "use", self.solc_version])

    def _parse_solc_version(self):
        version_resault = None

        with open(self.file_name, 'r', encoding='utf-8') as contract_code:

            mini = 100
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

                    print("version_resault:{}".format(version_resault))
                    self.solc_version = version_resault
                    return version_resault

                    # version_info = version_info.replace('\r', '').replace('\n', '').replace('\t', '')
                    # print("info:%s  ---  version:%s" % (version_info, version_resault))

            if version_resault is None:
                version_resault = "0.4.26"

            self.solc_version = version_resault
            return version_resault

    def get_opcode_frequency_feature(self):

        if os.path.exists("frequency.json"):
            return

        for contract_name in self.contract_asm_files:
            file_name = self.contract_asm_files[contract_name]
            with open(file_name, "r") as f:
                for line in f.readlines():

                    if "not defined" in line:
                        continue

                    else:
                        opcode = line.split(":")[1].split("\n")[0][1:].split(" ")[0]
                        self.opcode_total_cnt += 1
                        if opcode not in self.opcode_cnt:
                            self.opcode_cnt[opcode] = 1
                        else:
                            self.opcode_cnt[opcode] += 1

        info = {"total": self.opcode_total_cnt, "opcode_cnt": self.opcode_cnt}
        with open("frequency.json", "w+") as f:
            f.write(json.dumps(info))

        return





