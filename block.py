import base64
import hashlib
import uuid
from datetime import datetime
import math

from merkleTree import MerkleTree
from transaction import Transaction
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_der_public_key
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes


class Block:
    """
    a block on the chain
    """

    def __init__(self, transactions, chain, index=None, previous_hash=None, nonce=None, max_transaction=100, version=1,
                 difficulty=1, genesis=False):
        self.time_stamp = datetime.utcnow()
        self.first_transaction_time = datetime.utcnow()
        self.last_transaction_time = datetime.utcnow()

        self.input_transactions = transactions

        self.chain = chain

        self.genesis = genesis
        self.block_height = None
        self.transactions = []
        self.version = version
        self.id = uuid.uuid4()
        self.index = index
        self.previous_block_hash = previous_hash
        self.nonce = nonce
        self.difficulty = difficulty
        self.max_transaction = max_transaction
        self.transaction_count = 0

        self.hash_dict = dict()

        # compute the merkle tree
        self.merkle_tree = MerkleTree(transactions=transactions)

        self.setTransactionBlockHash()

        # add transactions
        self._addTransactions(transactions=self.input_transactions)

        # self.verifyAllTransactions()

        # generate the hash
        self.hash = self.genHash()

    def setTransactionBlockHash(self):
        for t in self.input_transactions:
            for i in t.outputs:
                i.block_hash = self.genHash()

    def changeNonce(self, nonce):
        self.nonce = nonce
        self.hash = self.genHash()

    def _addTransactions(self, transactions: [Transaction]):
        """
        adds transactions to the block if enough room
        called at creation of the block
        :param transactions: list of the transactions to add to the block
        :return:
        """
        for t in transactions:
            added = False
            if self.genesis:
                self.transactions.append(t)
                added = True
            else:
                if len(self.transactions) < self.max_transaction and not self.genesis:
                    self.transactions.append(t)
                    added = True

            if added:
                self.transaction_count += 1
                self.hash_dict.update({f"{t.genHash()}": len(self.transactions) - 1})
                if t.time < self.first_transaction_time:
                    self.first_transaction_time = t.time
                if t.time > self.last_transaction_time:
                    self.last_transaction_time = t.time

    def verifyAllTransactions(self):
        valid = []
        for t in self.input_transactions:
            if not self.genesis:
                if self.verify_transaction(t):
                    valid.append(t)
            else:
                valid.append(t)

        if len(self.input_transactions) != len(valid):
            self.transactions = valid
            self.merkle_tree = MerkleTree(transactions=self.transactions)
        else:
            self.transactions = valid
            self.merkle_tree = MerkleTree(transactions=self.transactions)

    def verify_transaction_details(self, transaction: Transaction):
        """
        from transaction gets the public key of the transaction
        verifies that the transaction data matches what was signed by the sender

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
        return True

    def verify_transaction(self, transaction: Transaction):
        # check signature on transaction matches up withe transaction details
        # ensures that details written by sender
        detail_check = self.verify_transaction_details(transaction=transaction)

        # check that inputs go to same as signature of indicator
        input_items = transaction.inputs
        for i in input_items:

            # value of -1 means that the value has no input and is simply created
            if i.value != -1:
                # find block in chain from items block hash
                # transaction in block from items transaction hash
                transaction_block_i = self.chain.hash_dict[i.block_hash]
                transaction_index = self.chain.blocks[transaction_block_i].hash_dict[i.transaction_hash]

                t = self.chain.blocks[transaction_block_i].transactions[transaction_index]

                # go though outputs
                found = False
                for output in t.outputs:
                    if output.recipient == transaction.sender:
                        found = True

                if not (found):
                    # didn't find the item in the transaction
                    return False

        return True

    def genHash(self, trial_nonce=None):
        """hash the block via header. first computes merkle root"""
        if trial_nonce is None:
            header = str(
                [self.previous_block_hash, self.merkle_tree.root, self.nonce])
        else:
            header = str(
                [self.previous_block_hash, self.merkle_tree.root, trial_nonce])
        sha = hashlib.sha256()
        sha.update(header.encode('utf-8'))
        return sha.hexdigest()

    def details(self):
        return {"time_stamp": self.time_stamp,
                "block_height": self.block_height,
                "transactions": self.transactions,
                "version": self.version,
                "id": self.id,
                "previous_block_hash": self.previous_block_hash,
                "nonce": self.nonce,
                "max_transaction": self.max_transaction,
                "transaction_count": self.transaction_count,
                "Merkle_root": self.merkle_tree.root}

    def findTransaction(self, t_hash=None, t_ID=None, t_datetime=None):
        if t_ID is None and t_hash is None:
            print(f"cant find individual transction without {t_hash} or {t_ID}")
            return None

        self.transactions.sort(key=lambda x: x.time)
        found_transaction = None

        # look at t_datetime, do a binary search though the block's min transaction times
        if t_datetime is not None:
            target_time = t_datetime
            low_time = self.transactions[0]
            high_time = self.transactions[-1]

            mid_index = int(len(self.transactions) / 2)

            while not (self.transactions[mid_index].time == target_time or
                       low_time.time == target_time or
                       high_time.time == target_time):

                if self.transactions[mid_index].time < target_time:
                    low_time = self.transactions[mid_index]
                elif self.transactions[mid_index].time >= target_time:
                    high_time = self.transactions[mid_index]

                mid_index = self.transactions.index(low_time) + math.ceil((self.transactions.index(high_time) - self.transactions.index(low_time)) / 2)

            if self.transactions[mid_index].time == target_time:
                found_transaction = self.transactions[mid_index]
            elif low_time.time == target_time:
                found_transaction = low_time
            elif high_time.time == target_time:
                found_transaction = high_time

            if found_transaction.genHash() == t_hash and str(found_transaction.id) == str(t_ID):
                return found_transaction


        else:
            for t in self.transactions:
                if t.genHash() == t_hash or t.id == t_ID:
                    return found_transaction

        for t in self.transactions:
            if t.genHash() == t_hash or t.id == t_ID:
                return found_transaction

        return None
