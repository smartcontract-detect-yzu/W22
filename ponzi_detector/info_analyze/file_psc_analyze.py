import os.path
import re
import shutil
import torch as th
from nltk.tokenize import word_tokenize


def _deleteSingleComment(_content):
    pattern = r"//(.)*(\n)?"
    return re.sub(pattern, "", _content)


def _deleteMultiComment(_content):
    pattern = r"/\*((.)|((\r)?\n))*?\*/"
    return re.sub(pattern, "", _content, re.S)


def reSubT(_content):
    pattern = r"\t"
    return re.sub(pattern, "", _content)


def reSubN(_content):
    pattern = r"\n"
    return re.sub(pattern, "", _content)


def reSubS(_content):
    pattern = r"(\s){1,}"
    return re.sub(pattern, " ", _content)


class FilePscAnalyzer:

    def __init__(self, file_name, label):
        self.file_name = file_name
        self.label = label
        self.file_content = self.getContent()
        self.no_comment_content = None
        self.sequence_content = None
        self.tokens = None

    def getContent(self):
        with open(self.file_name, "r", encoding="utf-8") as f:
            return f.read()

    # 删除注释
    def do_delete_comment(self):

        nowContent = self.file_content

        # 1. delete the single-line comment
        nowContent = _deleteSingleComment(nowContent)

        # 2. delete the multi-line comment
        nowContent = _deleteMultiComment(nowContent)

        self.no_comment_content = nowContent

        return nowContent

    # 变成一行
    def do_change_to_sequence(self):
        nowContent = self.no_comment_content

        # 1. delete \t
        nowContent = reSubT(nowContent)

        # 2. delete \n
        nowContent = reSubN(nowContent)

        # 3. delete \s
        nowContent = reSubS(nowContent)

        # 4. 保存
        self.sequence_content = nowContent

        return nowContent

    def get_psc_from_sol(self):

        self.do_delete_comment()
        self.do_change_to_sequence()

        psc_name = str(self.file_name).split(".sol")[0] + ".psc"

        if os.path.exists(psc_name):
            os.remove(psc_name)

        with open(psc_name, "w+") as psc_file:
            psc_file.write(self.sequence_content)

        with open("psc_done.txt", "w+") as f:
            f.write("done")

    def do_get_psc_for_file(self, FASTTEXT_MODEL):

        self.do_delete_comment()
        self.do_change_to_sequence()

        psc_name = str(self.file_name).split(".sol")[0] + ".psc"
        with open(psc_name, "w+") as psc_file:
            psc_file.write(self.sequence_content)

        dataset_dir_prefix = "../../../../psc_dataset/"
        if self.label == 1:
            dst = dataset_dir_prefix + "p"
        else:
            dst = dataset_dir_prefix + "np"
        shutil.copy(psc_name, dst)

        with open("psc_done.txt", "w+") as f:
            f.write("done")

    def do_tokenized_sequence(self, FASTTEXT_MODEL):

        self.tokens = list(word_tokenize(self.sequence_content))
        for token in self.tokens:
            token_data = FASTTEXT_MODEL.wv.__getitem__(token)
            print(token_data.shape)

    def do_print_sequence(self):
        print(self.sequence_content)



