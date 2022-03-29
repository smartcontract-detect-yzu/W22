import os
import shutil

from ponzi_detector.info_analyze.file_analyze import SolFileAnalyzer

DATASET_NAMES = [
    "sad_chain",
    "sad_tree",
    "xblock_dissecting",
    "buypool",
    "deposit",
    "etherscan"
]


class DataSet:

    def __init__(self, name="all"):
        self.name = name

    def get_work_dirs(self):

        dataset_prefixs = []
        analyze_prefixs = []

        if self.name != "all":

            dataset_prefixs.append("examples/ponzi_src/{}/".format(self.name))
            analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(self.name))
            cnt = 1
        else:

            for name in DATASET_NAMES:
                dataset_prefixs.append("examples/ponzi_src/{}/".format(name))
                analyze_prefixs.append("examples/ponzi_src/analyze/{}/".format(name))
            cnt = len(DATASET_NAMES)

        return dataset_prefixs, analyze_prefixs, cnt

    def get_asm_and_opcode_for_dataset(self, pass_tag=0):

        """
        为数据集中所有bin文件生成opcode和asm文件
        """

        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            dataset_prefix = dataset_prefix_list[i]
            analyze_prefix = analyze_prefix_list[i]

            g = os.walk(dataset_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".sol"):
                        src_file = os.path.join(path, file_name)

                        address = file_name.split(".sol")[0]
                        analyze_dir = analyze_prefix + address

                        if not os.path.exists(analyze_dir):
                            os.mkdir(analyze_dir)

                        if not os.path.exists(analyze_dir + "/" + file_name):
                            shutil.copy(src_file, analyze_dir)

                        done_file = analyze_dir + "/asm_done.txt"
                        pass_file = analyze_dir + "/pass.txt"

                        if os.path.exists(pass_file):
                            continue

                        if os.path.exists(done_file) and pass_tag:
                            print("========={}===========".format(file_name))
                            continue
                        else:

                            if os.path.exists(done_file):
                                os.remove(done_file)

                            solfile_analyzer = SolFileAnalyzer(file_name, analyze_dir)
                            solfile_analyzer.do_chdir()
                            solfile_analyzer.do_file_analyze_prepare()
                            solfile_analyzer.get_opcode_and_asm_file()
                            solfile_analyzer.get_opcode_frequency_feature()
                            solfile_analyzer.revert_chdir()

    def clean_up(self):

        dataset_prefix_list, analyze_prefix_list, cnt = self.get_work_dirs()
        for i in range(cnt):

            analyze_prefix = analyze_prefix_list[i]
            g = os.walk(analyze_prefix)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name.endswith(".asm") or file_name.endswith(".bin") or file_name.endswith(".evm"):
                        src_file = os.path.join(path, file_name)
                        os.remove(src_file)
