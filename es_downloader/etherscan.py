"""Etherscan API wrapper."""
import logging
from typing import Any, Dict, List, Optional

import requests

from .utils import clean_params

logger = logging.getLogger(__name__)


class EtherscanAPIError(Exception):
    def __init__(self, response, message=""):
        super().__init__(message)
        self.response = response
        self.message = message

    def __str__(self):
        return f"{self.response.status_code} {self.response.content.decode()}"


class Etherscan:
    """Etherscan API wrapper.

    https://etherscan.io/
    """

    # Mainnet endpoint
    BASE_URL = "https://api.etherscan.io/api"
    CN_BASE_URL = "https://api-cn.etherscan.com/api"

    def __init__(
            self, key: str, plan: str = "free", fail_silently: bool = False
    ):
        self.key = key
        self.plan = plan.lower()
        self.fail_silently = fail_silently

    def _get(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get requests to the specified path on Etherscan API."""
        if params is None:
            params = {}

        params.update({"apikey": self.key})

        r = requests.get(url=self.CN_BASE_URL, params=clean_params(params))

        if r.status_code == 200:
            return r.json()

        self._fail(r, params)

    def _fail(self, r, params):
        details = r.content.decode()
        try:
            details = r.json()
        except Exception:
            pass

        if not self.fail_silently:
            logger.warning(
                f"Etherscan API error {r.status_code} with {params} {details}"
            )
            raise EtherscanAPIError(response=r)

        logger.info(
            f"Etherscan API silent error {r.status_code} with {params} "
            f"{details}"
        )

    def get_gas_oracle(self):
        """Get the current Safe, Proposed and Fast gas prices."""
        return self._get(params={"module": "gastracker", "action": "gasoracle"})

    def get_balance_single_address(self, address: str, tag: str):
        """Get the Ether balance of a given address.

        address : the string representing the address to check for balance
        tag : the string pre-defined block parameter,
        either **earliest**, **pending** or **latest**
        """
        return self._get(
            params={
                "module": "account",
                "action": "balance",
                "address": address,
                "tag": tag,
            }
        )

    def get_balance_multiple_addresses(self, address: List[str], tag: str):
        """Get the balance of the accounts from a list of addresses.

        address : the strings representing the addresses to check for balance,
        separated by, up to 20 addresses per call
        tag : the integer pre-defined block parameter,
        either **earliest**, **pending** or **latest**
        """
        return self._get(
            params={
                "module": "account",
                "action": "balancemulti",
                "address": address,
                "tag": tag,
            }
        )

    def get_normal_transactions_by_address(
            self,
            address: str,
            startblock: int,
            endblock: int,
            page: int,
            offset: int,
            sort: str,
    ):
        """Get the list of transactions performed by an address+optional pagination.

        address : the string representing the addresses to check for balance
        startblock :the integer block number to start searching for transactions
        endblock : the integer block number to stop searching for transactions
        page : the integer page number, if pagination is enabled
        offset : the number of transactions displayed per page
        sort : the sorting preference, use **asc** to sort by ascending
        and **desc** to sort by descendin Tip: Specify a smaller startblock
        and endblock range for faster search results.
        """
        return self._get(
            params={
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offsetoffset": offset,
                "sort": sort,
            }
        )

    def get_internal_transactions_by_address(
            self,
            address: str,
            startblock: int,
            endblock: int,
            page: int,
            offset: int,
            sort: str,
    ):
        """Get list internal transactions performed by address+optional pagination.

        address : the string representing the addresses to check for balance
        startblock :the integer block number to start searching for transactions
        endblock : the integer block number to stop searching for transactions
        page : the integer page number, if pagination is enabled
        offset : the number of transactions displayed per page
        sort : the sorting preference, use **asc** to sort by ascending and
        **desc** to sort by descending
        """
        return self._get(
            params={
                "module": "account",
                "action": "txlistinternal",
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offsetoffset": offset,
                "sort": sort,
            }
        )

    def get_internal_transactions_by_hash(self, txhash: str):
        """Get the list of internal transactions performed within a transaction.

        txhash : the string representing the transaction hash to check
        for internal transactions
        """
        return self._get(
            params={
                "module": "account",
                "action": "txlistinternal",
                "txhash": txhash,
            }
        )

    def get_internal_transactions_by_block_range(
            self, startblock: int, endblock: int, page: int, offset: int, sort: str
    ):
        """Get list internal transactions performed in block range.

        startblock :the integer block number to start searching for transactions
        endblock : the integer block number to stop searching for transactions
        page : the integer page number, if pagination is enabled
        offset : the number of transactions displayed per page
        sort : the sorting preference, use **asc** to sort by ascending and
        **desc** to sort by descending
        """
        return self._get(
            params={
                "module": "account",
                "action": "txlistinternal",
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offsetoffset": offset,
                "sort": sort,
            }
        )

    def get_erc20_token_transferred_by_address(
            self,
            address: str,
            contractaddress: str,
            page: int,
            offset: int,
            startblock: int,
            endblock: int,
            sort: str,
    ):
        """Get list ERC-20 tokens transferred by address+filter by token contract.

        address : the string representing the address to check for balance
        contractaddress : the string representing the token contract address
        to check for balance
        page : the integer page number, if pagination is enabled
        offset : the number of transactions displayed per page
        startblock :the integer block number to start searching for transactions
        endblock : the integer block number to stop searching for transactions
        sort : the sorting preference, use **asc** to sort by ascending and
        **desc** to sort by descending
        """
        return self._get(
            params={
                "module": "account",
                "action": "tokentx",
                "contractaddress": contractaddress,
                "address": address,
                "page": page,
                "offset": offset,
                "startblock": startblock,
                "endblock": endblock,
                "sort": sort,
            }
        )

    def get_erc721_token_transferred_by_address(
            self,
            address: str,
            contractaddress: str,
            page: int,
            offset: int,
            startblock: int,
            endblock: int,
            sort: str,
    ):
        """Get list of ERC-721 (NFT) tokens transferred by an address.

        address : the string representing the address to check for balance
        contractaddress : the string representing the token contract address
        to check for balance
        page : the integer page number, if pagination is enabled
        offset : the number of transactions displayed per page
        startblock :the integer block number to start searching for transactions
        endblock : the integer block number to stop searching for transactions
        sort : the sorting preference, use **asc** to sort by ascending and
        **desc** to sort by descending
        """
        return self._get(
            params={
                "module": "account",
                "action": "tokennfttx",
                "contractaddress": contractaddress,
                "address": address,
                "page": page,
                "offset": offset,
                "startblock": startblock,
                "endblock": endblock,
                "sort": sort,
            }
        )

    def get_blocks_mined_by_address(
            self, address: str, blocktype: str, page: int, offset: int
    ):
        """Get the list of blocks mined by an address.

        address : the string representing the address to check for balance
        blocktype : the string pre-defined block type,
        either blocks for canonical blocks or uncles for uncle blocks only
        page : the integer page number, if pagination is enabled
        offset : the number of transactions displayed per page
        """
        return self._get(
            params={
                "module": "account",
                "action": "getminedblocks",
                "address": address,
                "blocktype": blocktype,
                "page": page,
                "offset": offset,
            }
        )

    def get_abi_verified_smart_contract(self, address: str):
        """Get Contract Application Binary Interface(ABI) of smart contract.

        address : the contract address that has a verified source code
        """
        return self._get(
            params={
                "module": "contract",
                "action": "getabi",
                "address": address,
            }
        )

    def get_source_code_smart_contract(self, address: str):
        """Get the Solidity source code of a verified smart contract.

        address : the contract address that has a verified source code
        """
        return self._get(
            params={
                "module": "contract",
                "action": "getsourcecode",
                "address": address,
            }
        )

    def get_contract_execution_status(self, txhash: str):
        """Get the status code of a contract execution.

        txhash : the string representing the transaction hash
        to check the execution status
        """
        return self._get(
            params={
                "module": "transaction",
                "action": "getstatus",
                "txhash": txhash,
            }
        )

    def get_transaction_execution_status(self, txhash: str):
        """Get the status code of a transaction execution.

        txhash : the string representing the transaction hash
        to check the execution status
        """
        return self._get(
            params={
                "module": "transaction",
                "action": "gettxreceiptstatus",
                "txhash": txhash,
            }
        )

    def get_block_uncleblock_reward_by_blockno(self, blockno: int):
        """Get the block reward and 'Uncle' block rewards.

        Args:
            blockno : the integer block number to check block rewards for eg.
        """
        return self._get(
            params={
                "module": "block",
                "action": "getblockreward",
                "blockno": blockno,
            }
        )

    def get_estimate_mined_countdown_by_blockno(self, blockno: int):
        """Get estimated time remaining until a certain block is mined.

        Args:
            blockno : the integer block number to estimate time remaining
                to be mined.
        """
        return self._get(
            params={
                "module": "block",
                "action": "getblockcountdown",
                "blockno": blockno,
            }
        )

    def get_block_number_by_tymestamp(self, timestamp: int):
        """Get the block number that was mined at a certain timestamp.

        timestamp : the integer representing the Unix timestamp in seconds.
        closest : the closest available block to the provided timestamp,
        either before or after
        """
        return self._get(
            params={
                "module": "block",
                "action": "getblocknobytime",
                "timestamp": timestamp,
            }
        )

    def get_no_most_recent_block(self):
        """Get the number of most recent block."""
        return self._get(
            params={"module": "proxy", "action": "eth_blockNumber"}
        )

    def get_block_by_number(self, tag: str, boolean: bool):
        """Get information about a block by block number.

        tag : the block number, in hex eg. 0xC36B3C
        boolean : the boolean value to show full transaction objects.
        when true, returns full transaction objects and their information,
        when false only returns a list of transactions.
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getblockbynumber",
                "tag": tag,
                "boolean": boolean,
            }
        )

    def get_uncle_by_block_number(self, tag: str, index: str):
        """Get information about a uncle by block number.

        tag : the block number, in hex eg. 0xC36B3C
        index : the position of the uncle's index in the block, in hex eg. 0x5
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getUncleByBlockNumberAndIndex",
                "tag": tag,
                "index": index,
            }
        )

    def get_number_transaction_in_block(self, tag: str):
        """Get the number of transactions in a block.

        tag : the block number, in hex eg. 0x10FB78
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getBlockTransactionCountByNumber",
                "tag": tag,
            }
        )

    def get_transaction_by_hash(self, txhash: str):
        """Get the information about a transaction requested by transaction hash.

        txhash : the string representing the hash of the transaction
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getTransactionByHash",
                "txhash": txhash,
            }
        )

    def get_transaction_by_blocknumber_and_index(self, tag: str, index: str):
        """Get info about transaction by block number&transaction index position.

        tag : the block number, in hex eg. 0x10FB78
        index : the position of the uncle's index in the block, in hex eg. 0x0
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getTransactionByBlockNumberAndIndex",
                "tag": tag,
                "index": index,
            }
        )

    def get_count_transactions_by_address(self, address: str, tag: str):
        """Get the number of transactions performed by an address.

        address : the string representing the address to get transaction count
        tag : the string pre-defined block parameter, either earliest,
        pending or latest
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getTransactionCount",
                "address": address,
                "tag": tag,
            }
        )

    def get_receipt_by_transaction_hash(self, txhash: str):
        """Get the receipt of a transaction by transaction hash.

        txhash : the string representing the hash of the transaction
        """
        return self._get(
            params={
                "module": "proxy",
                "action": "eth_getTransactionReceipt",
                "txhash": txhash,
            }
        )

    def get_gas_price(self):
        """Get the current price per gas in wei."""
        return self._get(params={"module": "proxy", "action": "eth_gasPrice"})

    def get_erc20_in_circulation(self, contractaddress: str):
        """Get the current amount of an ERC-20 token in circulation.

        contractaddress : the contract address of the ERC-20 token
        """
        return self._get(
            params={
                "module": "stats",
                "action": "tokensupply",
                "contractaddress": contractaddress,
            }
        )

    def get_erc20_balance_of_address(self, contractaddress: str, address: str):
        """Get the current balance of an ERC-20 token of an address.

        contractaddress : the contract address of the ERC-20 token
        address : the string representing the address to check for token balance
        """
        return self._get(
            params={
                "module": "stats",
                "action": "tokenbalance",
                "contractaddress": contractaddress,
                "address": address,
            }
        )

    def get_ether_supply(self):
        """Get the current amount of Ether in circulation."""
        return self._get(params={"module": "stats", "action": "ethsupply"})

    def get_eth2_supply(self):
        """Get ETH in circulation+ETH2 Staking reward+EIP1559 burnt fee stat."""
        return self._get(params={"module": "stats", "action": "ethsupply2"})

    def get_ether_last_price(self):
        """Get the latest price of 1 ETH."""
        return self._get(params={"module": "stats", "action": "ethprice"})

    def get_nodes_size(
            self,
            startdate: str,
            enddate: str,
            clienttype: str,
            syncmode: str,
            sort: str,
    ):
        """Get the size of the Ethereum blockchain in bytes over a date range.

        Args:
            startdate : the starting date
            enddate : the ending date
            clienttype : the Ethereum to use, either geth or parity
            syncmode : the  to run, either default or archive
            sort : the sorting preference, asc or desc
        """
        return self._get(
            params={
                "module": "stats",
                "action": "chainsize",
                "startdate": startdate,
                "enddate": enddate,
                "clienttype": clienttype,
                "syncmode": syncmode,
                "sort": sort,
            }
        )

    def get_total_nodes_count(self):
        """Get the total number of discoverable Ethereum nodes."""
        return self._get(params={"module": "stats", "action": "nodecount"})
