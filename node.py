import hashlib
import base64
from block import Block
from blockchain import BlockChain
from network import Network
from transaction import Transaction
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_der_public_key
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes

import time


class Node:
    """
    base class of node

    this one only records its correct value of the blockchain
    from the transactions from the network
    """

    def __init__(self, chain: BlockChain, network: Network):
        self.chain = chain
        self.network = network

        self.unprocessed_transaction_memory = []
        self.processed_transactions_memory = []

        # list of transactions received in valid blocks that have not been received individualy (shouldn't happen here)
        self.processed_transactions_currently_unreceived = []

        self.blocks_mined = []
        self.other_blocks_accepted = []
        self.other_blocks_rejected = []

    def verifySteakholders(self, transaction):
        """
        verifies ownership of the transaction
        :param transaction:
        :return:
        """

        indicator_signature = base64.b64decode(transaction.signature)
        sender_public_key = load_der_public_key(base64.b64decode(transaction.sender), default_backend())
        try:
            sender_public_key.verify(
                indicator_signature,
                str(transaction.details()).encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except Exception:
            return False
        for client in self.network.users:
            if client.key_pair.public_key_str == transaction.sender:
                return client


    def receive_transactions(self):
        """
        get unprocessed transactions from the network
        """
        all_transactions = self.network.broadcast_transactions
        unprocessed = [t for t in all_transactions if (t not in self.processed_transactions_memory) and (
                t not in self.processed_transactions_currently_unreceived)]

        self.unprocessed_transaction_memory.extend(unprocessed)
        self.unprocessed_transaction_memory.sort(key=lambda x: x.time)

    def processTransactions(self, transactions: [Transaction]):
        """
        update the transaciton sets
        :param transactions:
        :return:
        """
        to_process = set(transactions)
        processed = set(self.processed_transactions_memory)
        unprocessed = set(self.unprocessed_transaction_memory)

        unreceived_transactions = [i for i in to_process if i not in unprocessed]

        processed.update(to_process)
        self.processed_transactions_memory = list(processed)

        unprocessed.difference_update(to_process)
        self.unprocessed_transaction_memory = list(unprocessed)

        self.processed_transactions_currently_unreceived.extend(
            unreceived_transactions)

    def addBlock(self, block: Block):
        block_transactions = block.transactions

        # check that block has no repeited processed transactions
        if len(set(self.processed_transactions_memory).intersection(set(block_transactions))) != 0:
            return False

        # attempt to add the block to the chain
        agree = self.chain.addBlock(block=block)
        if agree:
            print("block accepted")
            self.other_blocks_accepted.append(block)

            # update the transactions sets
            self.processTransactions(block_transactions)

        else:
            print("block rejected")
            self.other_blocks_rejected.append(block)
            return False

        return True


class Miner(Node):
    def __init__(self, chain: BlockChain, network: Network, attempt_time=0):
        super().__init__(chain=chain, network=network)

        self.mine_flag = True
        self.start_hash = "1" * self.chain.difficulty
        self.success_string = "0" * self.chain.difficulty

        self.attempt_time = attempt_time

    def hashStr(self, s):
        sha = hashlib.sha256()
        sha.update(s.encode('utf-8'))
        return sha.hexdigest()

    def getTransactionsToProcess(self):
        """
        :return: list of maximum processes a block can handle
        """
        # finds to process
        to_process = self.unprocessed_transaction_memory[:self.chain.block_length]

        return to_process

    def getPreviousHash(self):
        """
        :return: the previous proof braodcast to the nework
        """
        if len(self.network.broadcast_blocks) != 0:
            return self.network.broadcast_blocks[-1].hash
        return self.chain.blocks[0].hash

    def mine(self, attempts: int = None, track=False):
        """
        mines continuously untill attemts reached
        :param track:
        :param attempts: number of rounds to attempt to mine for
        :return:
        """

        while self.mine_flag:
            previous_hash = self.getPreviousHash()
            if track:
                # 4)a) 4)b)
                hashes_computed, time_taken = self.POW(previous_hash=previous_hash, track_hashes=track,
                                                       time_limmit=float(3600))
                return time_taken, hashes_computed
            else:
                self.POW(previous_hash=previous_hash, track_hashes=track)

            if attempts is not None:
                attempts -= 1
                if attempts == 0:
                    self.mine_flag = False
        self.mine_flag = True

    def POW(self, previous_hash, track_hashes=False, time_limmit=None):
        """
        performs proof of work.
        attempts to mine the next block
        if successful
            creates, broadcasts and appends block to chain
        otherwise:
            validates the successful broadcast block
        :param time_limmit: stops analysis after this periould of time
        :param track_hashes: bool to turn the method to just return the number of hashes nessaiiryt o find valid proof
        :param previous_hash: the nonce found in the previous block
        :return:
        """
        # 4)a)
        start_time = time.perf_counter()

        current_blocks_mined = self.network.blocks_mined

        current_hash = self.start_hash
        candidate_block = Block(transactions=self.getTransactionsToProcess(),
                                index=len(self.chain.blocks) + 1,
                                previous_hash=previous_hash,
                                max_transaction=self.chain.block_length,
                                chain=self.chain)

        proof = 0
        hash_numer = 0

        if time_limmit is not None:
            # for time limmit
            while (current_hash[:self.chain.difficulty] != self.success_string) and (
                    self.network.blocks_mined == current_blocks_mined) and (
                    time.perf_counter() - start_time < time_limmit):
                hash_numer += 1
                proof += 1
                current_hash = candidate_block.genHash(trial_nonce=proof)

            if time.perf_counter() - start_time > time_limmit:
                # 4)a) 4)b)
                return hash_numer, -1

        else:
            # for non time limmit
            while (current_hash[:self.chain.difficulty] != self.success_string) and (
                    self.network.blocks_mined == current_blocks_mined):
                hash_numer += 1
                proof += 1
                current_hash = candidate_block.genHash(trial_nonce=proof)



        if current_hash[:self.chain.difficulty] == self.success_string and not (
                self.network.blocks_mined != current_blocks_mined):

            if track_hashes:
                # exits at this part when we are doing analysis on the mining
                # regular execution does not enter here
                # 4)a) 4)b)
                return hash_numer, time.perf_counter() - start_time

            # we have a winning proof
            # need to create our new block and broadcast it
            self.network.blocks_mined += 1

            new_block = candidate_block
            candidate_block.changeNonce(nonce=proof)
            candidate_block.setTransactionBlockHash()
            candidate_block.verifyAllTransactions()

            # broadcast the new block along with the new valid proof
            self.network.broadcast_blocks.append(new_block)

            # add the new block to its own chain
            accepted = self.addBlock(new_block)
            if not accepted:
                print("ERROR - self mined block not accepted")
            else:
                print(current_hash)

            self.blocks_mined.append(new_block)
            #time.sleep(1)

        else:
            # we have failed to mine a valid proof in time have lost the race
            # need to give up on our own eforts and validate the new block

            # wait for new block to be uploaded
            time.sleep(0.1)
            # get sucsessful block minned form network
            unvalidated_new_block = self.network.broadcast_blocks[-1]

            # validates the block and adds to own chain if so
            agree = self.chain.addBlock(unvalidated_new_block)
            if agree:
                print("block accepted")
                self.other_blocks_accepted.append(unvalidated_new_block)
            else:
                print("block rejected")
                self.other_blocks_rejected.append(unvalidated_new_block)
        pass
