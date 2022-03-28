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

        cnt = 1
        dataset_prefix = []
        analyze_prefix = []

        if self.name != "all":

            dataset_prefix.append("examples/ponzi_src/{}/".format(self.name))
            analyze_prefix.append("examples/ponzi_src/analyze/{}/".format(self.name))
            cnt = 1
        else:
            
            for name in DATASET_NAMES:
                dataset_prefix.append("examples/ponzi_src/{}/".format(name))
                analyze_prefix.append("examples/ponzi_src/analyze/{}/".format(name))
            cnt = len(DATASET_NAMES)

        return dataset_prefix, analyze_prefix, cnt
