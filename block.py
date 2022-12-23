import base64
import hashlib
import uuid
from datetime import datetime

from merkleTree import MerkleTree
from transaction import Transaction
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes


class Block:
    """
    a block on the chain
    """

    def __init__(self, transactions, chain, index=None, previous_hash=None, nonce=None, max_transaction=100, version=1,
                 difficulty=1):
        self.time_stamp = datetime.utcnow()

        self.chain = chain

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

        # add transactions
        self._addTransactions(transactions=transactions)

        for t in self.transactions:
            for i in t.outputs:
                i.block_hash = self.genHash()

        # generate the hash
        self.hash = None

    def _addTransactions(self, transactions: [Transaction]):
        """
        adds transactions to the block if enough room
        called at creation of the block
        :param transactions: list of the transactions to add to the block
        :return:
        """
        for t in transactions:
            if len(self.transactions) < self.max_transaction and self.verify_transaction(t):
                self.transactions.append(t)
            self.transaction_count += 1
            self.hash_dict.update({f"{t.genHash()}": len(self.transactions) - 1})

    def verify_transaction(self, transaction: Transaction):
        # check signature on transaction matches up withe transaction details
        indicator_signature = transaction.signature
        try:
            signature = base64.b64decode(indicator_signature)
            self.public_key.verify(
                signature,
                transaction.details(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except Exception:
            return False

        # check that inputs go to same as signature of indicator
        input_items = transaction.inputs
        for i in input_items:
            # find block in chain from items block hash
            # transaction in block from items transaction hash
            item_creation_transaction = self.chain.hash_table[i.block_hash].hash_table[i.transaction_hash]

            original_transaction_recipient = None
            # go through the creation transaction and find the original recipient for the item
            for output_i in range(len(item_creation_transaction.outputs)):
                if item_creation_transaction.outputs[output_i].id == item_creation_transaction.id:
                    original_transaction_recipient = item_creation_transaction.reciepients[output_i]

            if original_transaction_recipient is None:
                # didn't find the item in the transaction
                return False

            # compare the current sender and the previous recipient public keys
            if transaction.sender != original_transaction_recipient:
                # trying to spend money sent to someone else
                return False

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
