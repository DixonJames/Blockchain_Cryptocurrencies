import hashlib
import uuid

import math
from Item import Item
from block import Block
from client import Client
from network import Network
from transaction import Transaction


class BlockChain:
    """
    the blockchain containing everything
    """

    def __init__(self, previous_chain=None, difficulty=5, block_length=99):
        self.id = uuid.uuid4()
        self.difficulty = difficulty
        self.block_length = block_length

        self.hash_dict = dict()

        if previous_chain is None:
            self.blocks = []
            self.createGenesis()
        else:
            self.blocks = previous_chain




    def createGenesis(self):
        """
        creates the genesis block
        adss a transaction issuing 100 value to user 0 as the genesis transaction
        :return:
        """
        # create a base user
        user = Client("Satoshi", network=Network())
        user_addr = user.key_pair.public_key_str

        # create fictitious mystery items
        start_input = Item(value=100, recipient=None)
        start_output = Item(value=100, recipient=user_addr)

        # create and sign transaction
        genesis_transaction = Transaction(sender=user,
                                          receivers=user_addr,
                                          inputs=[start_input],
                                          outputs=[start_output])
        # add block to the chain
        self.addGenesisBlock(transactions=[genesis_transaction],
                             nonce=0)

    def addGenesisBlock(self, transactions: [Transaction], nonce: int):
        """
        adds a block to the chain
        :param transactions: transaction list to be included in the block
        :param nonce: proof found by miner
        :return:
        """
        # change the previous ash depending on if genesis block or not

        previous_hash = None

        # create the block
        block = Block(transactions=transactions,
                      index=len(self.blocks) + 1,
                      previous_hash=previous_hash,
                      nonce=nonce,
                      max_transaction=self.block_length,
                      chain=self,
                      genesis=True)

        self.hash_dict.update({f"{block.genHash()}": 0})

        # gp though items in genesis transaction and add the block and transaction hashes
        for transaction in block.transactions:
            for item in transaction.outputs:
                item.block_hash = block.genHash()
                item.transaction_hash = transaction.genHash()

        # append the block
        self.blocks.append(block)

    def addBlock(self, block: Block):
        """
        adds a block to the chain only if the chain is valid with it
        :param block:
        :return: a boolean as to weather or not the block is accepted
        """
        self.blocks.append(block)
        if (not self.verify_chain()):
            self.blocks = self.blocks[:-1]
            return False

        self.hash_dict.update({f"{block.genHash()}": len(self.blocks) - 1})
        return True

    def hashProofs(self, current_proof, previous_proof):
        """
        takes two proofs for neighboring blocks on the blockchain
        and creates a hash of them
        :param current_proof:
        :param previous_proof:
        :return:
        """
        proof_string = str(current_proof ** 2 - previous_proof ** 2)
        sha = hashlib.sha256()
        sha.update(proof_string.encode('utf-8'))
        return sha.hexdigest()

    def verify_chain(self):
        """
        goes through the current chain of blocks and checks:
        1. current stated block hash is the same as the now computed hash
        2. the stated previous hash is the same as the hash of the previous block in the chain
        :return: a boolean as to weather or not the chain is accepted
        """
        for i in range(len(self.blocks) - 1, 0, -1):

            # block hash checking
            current: Block = self.blocks[i]
            previous: Block = self.blocks[i - 1]

            if current.hash != current.genHash():
                return False
            if current.previous_block_hash != previous.hash:
                return False

            # nonce checking
            # hash of previous proof and current proof is valid hash inside hash difficulty region
            # checks validity of the current nonce mined

            hash_proofs = current.genHash()
            if hash_proofs[:self.difficulty] != "0" * self.difficulty:
                return False

        return True

    def findTransaction(self, t_hash=None, t_ID=None, t_datetime=None):
        if t_ID is None and t_hash is None:
            print(f"cant find individual transction without {t_hash} or {t_ID}")
            return None

        self.blocks.sort(key=lambda x: x.time_stamp)

        # look at t_datetime, do a binary search though the block's transaction time ranges
        if t_datetime is not None:
            target_time = t_datetime
            low_block = self.blocks[0]
            high_block = self.blocks[-1]

            mid_index = int(len(self.blocks) / 2)

            while not (self.blocks[mid_index].first_transaction_time <= target_time <= self.blocks[
                mid_index].last_transaction_time):

                if self.blocks[mid_index].first_transaction_time > target_time:
                    high_block = self.blocks[mid_index]
                    mid_index = self.blocks.index(low_block) + math.ceil(
                        (self.blocks.index(high_block) - self.blocks.index(low_block)) / 2)

                elif self.blocks[mid_index].last_transaction_time < target_time:
                    low_block = self.blocks[mid_index]
                    mid_index = self.blocks.index(low_block) + math.ceil(
                        (self.blocks.index(high_block) - self.blocks.index(low_block)) / 2)



            containing_block = self.blocks[mid_index]

            found_transaction = containing_block.findTransaction(t_hash=t_hash, t_ID=t_ID, t_datetime=t_datetime)
            return found_transaction


        else:
            for b in self.blocks:
                t = b.findTransaction(t_hash=t_hash, t_ID=t_ID, t_datetime=t_datetime)
                if t is not None:
                    return t
        return None
