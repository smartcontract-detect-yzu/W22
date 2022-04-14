# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import os.path
import urllib
import time

import random
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler
import urllib.request;
from bs4 import BeautifulSoup
import requests

OPCODE_MAP = {
    "STOP": 1,
    "ADD": 2,
    "SUB": 3,
    "MUL": 4,
    "DIV": 5,
    "SDIV": 6,
    "MOD": 7,
    "SMOD": 8,
    "EXP": 9,
    "NOT": 10,
    "LT": 11,
    "GT": 12,
    "SLT": 13,
    "SGT": 14,
    "EQ": 15,
    "ISZERO": 16,
    "AND": 17,
    "OR": 18,
    "XOR": 19,
    "BYTE": 20,
    "SHL": 21,
    "SHR": 22,
    "SAR": 23,
    "ADDMOD": 24,
    "MULMOD": 25,
    "SIGNEXTEND": 26,
    "KECCAK256": 27,
    "ADDRESS": 28,
    "BALANCE": 29,
    "ORIGIN": 30,
    "CALLER": 31,
    "CALLVALUE": 32,
    "CALLDATALOAD": 33,
    "CALLDATASIZE": 34,
    "CALLDATACOPY": 35,
    "CODESIZE": 36,
    "CODECOPY": 37,
    "GASPRICE": 38,
    "EXTCODESIZE": 39,
    "EXTCODECOPY": 40,
    "RETURNDATASIZE": 41,
    "RETURNDATACOPY": 42,
    "EXTCODEHASH": 43,
    "BLOCKHASH": 44,
    "COINBASE": 45,
    "TIMESTAMP": 46,
    "NUMBER": 47,
    "DIFFICULTY": 48,
    "GASLIMIT": 49,
    "CHAINID": 50,
    "SELFBALANCE": 51,
    "POP": 52,
    "MLOAD": 53,
    "MSTORE": 54,
    "MSTORE8": 55,
    "SLOAD": 56,
    "SSTORE": 57,
    "JUMP": 58,
    "JUMPI": 59,
    "PC": 60,
    "MSIZE": 61,
    "GAS": 62,
    "JUMPDEST": 63,
    "LOG0": 64,
    "LOG1": 65,
    "LOG2": 66,
    "LOG3": 67,
    "LOG4": 68,
    "CREATE": 69,
    "CALL": 70,
    "CALLCODE": 71,
    "STATICCALL": 72,
    "RETURN": 73,
    "DELEGATECALL": 74,
    "CREATE2": 75,
    "REVERT": 76,
    "INVALID": 77,
    "SELFDESTRUCT": 78
}

USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
    "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
]


# 使用代理访问网站 - GET方式实现
def crawl_url_by_get(url, proxy=None, enable_proxy=False):
    # make a string with 'GET'
    method = 'GET'

    # create a handler
    proxy_handler = ProxyHandler(proxy)
    null_proxy_handler = ProxyHandler({})

    # create an openerdirector instance according to enable_proxy
    if enable_proxy:
        cookies = urllib.request.HTTPCookieProcessor()
        opener = urllib.request.build_opener(cookies, proxy_handler, urllib.request.HTTPHandler)
        # print 'using proxy to crawl pages', url
    else:
        opener = urllib.request.build_opener(null_proxy_handler)
        # print 'without using proxy to crawl pages', url

    # install opener
    urllib.request.install_opener(opener)

    # buidl a request
    request = urllib.request.Request(url)

    # Umcomment the below line for ramdom choose the user_agent
    # user_agent = random.choice(USER_AGENT_LSIT)
    user_agent = USER_AGENT_LIST[1]
    request.add_header('User-Agent', user_agent)
    request.get_method = lambda: method

    try:
        connection = opener.open(request, timeout=5)
        if connection.code == 200:
            html = connection.read()
            return html
        else:
            return None
    except HTTPError as ex:
        # print e.code, e.reason
        print('spider_url_by_get（） -------- ' + str(ex))
        connection = ex
        return None
    except URLError as ex:
        # print e.reason
        # print e.code, e.reason
        print('spider_url_by_get（） -------- ' + str(ex))
        # remove_proxy(proxy)
        return None
    except Exception as ex:
        print('spider_url_by_get（） -------- ' + str(ex))
        # remove_proxy(proxy)
        return None


# 代码图表示构建器
class Tools:
    def __init__(self, address=None):

        self.api_key = 'api_key.json'
        self.to_download_list = "./examples/download/download_list.txt"
        self.to_download_dir = "./examples/download/"

        self.target_address = address

        self.last_error_download = None
        self.download_cnt = 0

    def _download(self):
        file = open(self.to_download_list, 'r')
        count = 0
        while True:
            count = count + 1
            address = file.readline()
            if address == '':
                break  # 退出条件

            address = address[0:address.find(" ")]
            target_file = self.to_download_dir + address + '.sol'
            if os.path.exists(target_file):
                continue

            print("开始下载:{}".format(address))
            url = "https://cn.etherscan.com/address/" + address + "#code"
            html = crawl_url_by_get(url, proxy=None, enable_proxy=True)
            soup = BeautifulSoup(html, 'lxml')
            data_contract = soup.select('#editor')
            contract = data_contract.__str__()
            contract = contract[contract.find("/"):-7]
            contract = contract.replace("&gt;", ">");
            contract = contract.replace("&lt;", "<");
            contract = contract.replace("&#39;", "'");
            contract = contract.replace("&amp;&amp;", "&&");
            contract = contract.replace("&amp;", "&");
            contract = contract.replace("&nbsp;", " ");

            with open(target_file, 'w', encoding='utf-8') as f:
                try:
                    f.write(contract)
                    print("write success")
                except UnicodeError as u:
                    print("write fail")
                    continue

        return True

    def download_from_etherscan_by_address(self):

        address = self.target_address
        print("开始下载:{}".format(address))
        url = "https://cn.etherscan.com/address/" + address + "#code"
        html = crawl_url_by_get(url, proxy=None, enable_proxy=True)
        soup = BeautifulSoup(html, 'lxml')
        data_contract = soup.select('#editor')
        contract = data_contract.__str__()
        contract = contract[contract.find("/"):-7]
        contract = contract.replace("&gt;", ">");
        contract = contract.replace("&lt;", "<");
        contract = contract.replace("&#39;", "'");
        contract = contract.replace("&amp;&amp;", "&&");
        contract = contract.replace("&amp;", "&");
        contract = contract.replace("&nbsp;", " ");

        target_file = self.to_download_dir + address + '.sol'
        with open(target_file, 'w+', encoding='utf-8') as f:
            f.write(contract)
            print("write success")

    def download_from_etherscan_by_list(self):
        while True:
            try:
                if self._download():
                    break
            except TypeError:
                print("报错，等待再次拉起")
                time.sleep(5)
