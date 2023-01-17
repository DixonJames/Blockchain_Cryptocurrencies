import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify, send_file
import random
import energyusage

import math
import os
import qrcode
import time
import base64
import hashlib
import uuid
import numpy as np
import pickle

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption, PublicFormat, \
    load_der_public_key
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.backends import default_backend

from datetime import datetime


class Item:
    def __init__(self, value, recipient=None, description=None):
        self.id = uuid.uuid4()

        if value is not None:
            self.value = value
        else:
            # for the null object that
            self.value = -1

        if self.value == -1:
            self.description = "currency creation item"
        else:
            self.description = description

        self.transaction_hash = None
        self.block_hash = None
        self.recipient = recipient

    def details(self):
        return {"id": self.id,
                "value": self.value,
                "description": self.description
                }

    def genHash(self):
        """
        hashes item data
        :return:
        """
        header = str(self.details())
        sha = hashlib.sha256()
        sha.update(header.encode('utf-8'))
        return sha.hexdigest()


class Client:
    """
    the class for a user on the BC
    """

    def __init__(self, name, network):
        self.key_pair = Keypair()
        self.qr_code_dir = os.path.join(os.getcwd(), "QR_ " + self.key_pair.keyStrings()[1][10:20] + ".png")

        self.identifier = name
        self.qr_code = self.genQRCOde()
        self.network = network

    def genQRCOde(self):
        """
        creates the Qr code from the public key
        :return:
        """
        qr = qrcode.QRCode(version=1,
                           box_size=10,
                           border=5)
        pub_key_string = self.key_pair.keyStrings()[1]
        qr.add_data(pub_key_string)
        qr.make(fit=True)

        img = qr.make_image(fill_color='black',
                            back_color='white')

        img.save(self.qr_code_dir, format="PNG")

        return img

    def details(self):
        """
        returns dict of user details
        :return:
        """
        return {"private key": self.key_pair.keyStrings()[0],
                "wallet address": self.key_pair.keyStrings()[1],
                "QR code": f"{self.qr_code_dir}"}

    def sendTransaction(self, receivers, inputs: [int], outputs: [Item]):
        reciver_i = 0
        for output in outputs:
            output.recipient = receivers[reciver_i]
            reciver_i += 1

        transaction = Transaction(sender=self, receivers=receivers, inputs=inputs, outputs=outputs)

        if transaction.verify():
            self.network.broadcastTransaction(transaction)
            return transaction.genHash(), transaction
        else:
            print(f"invalid transaction details:{transaction.details()}")


class Network:
    """
    the network in which all nodes are on
    """

    def __init__(self):
        self.users = []
        self.broadcast_transactions = []

        self.broadcast_blocks = []
        self.blocks_mined = 0

    def broadcastTransaction(self, transaction):
        self.broadcast_transactions.append(transaction)

    def addUser(self, user: Client):
        self.users.append(user)


class Transaction:
    """
    a transaction on a block in the blockchain
    """

    def __init__(self, sender: Client, receivers: [Client], inputs: [Item], outputs: [Item]):
        """
        indexes of outputs must match up to the indexes of receivers


        param sender: a Client object
        :param receivers: list of wallet addresses
        :param inputs:
        :param outputs: list of items. same length as receivers
        """

        self.id = uuid.uuid4()
        self.sender = sender.key_pair.public_key_str
        self.receivers = receivers

        self.outputs = outputs
        self.time = datetime.utcnow()

        # inputs should already have hashes of their creation transactions
        self.inputs = inputs

        # every output item contains the hash from whence it came
        for i in self.outputs:
            i.transaction_hash = self.genHash()

        self.signature = sender.key_pair.sign(str(self.details()))

    def details(self):
        """
        creates a dict of the attributes of the transaction
        :return:
        """
        return {"id": str(self.id),
                "sender": self.sender,
                "receiver": self.receivers,
                "inputs": str([i.genHash() for i in self.inputs]),
                "outputs": str([i.genHash() for i in self.outputs]),
                "in value": sum([i.value for i in self.inputs]),
                "out value": sum([i.value for i in self.outputs])}

    def verify(self):
        in_sum = sum([i.value for i in self.inputs])
        out_sum = sum([i.value for i in self.outputs])
        if in_sum != out_sum and not (in_sum == -1):
            return False

        if len(self.outputs) != len(self.receivers):
            return False

        return True

    def genHash(self):
        """
        hashes transaction data
        :return:
        """

        header = str(self.details())
        sha = hashlib.sha256()
        sha.update(header.encode('utf-8'))
        return sha.hexdigest()


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
                    mgf=padding.MGF1(SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                SHA256()
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

                mid_index = self.transactions.index(low_time) + math.ceil \
                    ((self.transactions.index(high_time) - self.transactions.index(low_time)) / 2)

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
        """
        finds a transaction on the chain via a binary search method
        :param t_hash:
        :param t_ID:
        :param t_datetime:
        :return:
        """
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


class Keypair:
    def __init__(self):
        self.priv_key, self.public_key = self.createKeyPair()

        self.verify_padding = padding.PSS(
            mgf=padding.MGF1(SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        )

        self.priv_key_str, self.public_key_str = self.keyStrings()

    def createKeyPair(self):
        """
        creates a new random key-pairing
        :return:
        """
        priv_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = priv_key.public_key()

        return priv_key, public_key

    def keyStrings(self, ):
        """
        creates the keypair keystrings

        :return:
        """
        private = str(self.priv_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=NoEncryption()
        ), 'utf-8').split('-----')[2].replace('\n', '')

        public = str(self.public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        ), 'utf-8').split('-----')[2].replace('\n', '')

        return private, public

    def sign(self, data):
        signature = self.priv_key.sign(
            str(data).encode('utf-8'),
            self.verify_padding,
            SHA256()
        )

        return base64.b64encode(signature)

    def verify(self, data, signature):
        try:
            signature = base64.b64decode(signature)
            self.public_key.verify(
                signature,
                data,
                self.verify_padding,
                SHA256()
            )
            return True
        except Exception:
            return False


class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.findRoot()

    def findRoot(self):
        tree = [hash(t.genHash()) for t in self.transactions]

        while len(tree) != 1:
            next_layer = []
            if len(tree) % 2 != 0:
                tree.append(tree[-1])
            for i in range(int(len(tree) / 2)):
                h_val = hash(str(hash(tree[i * 2])) + str(hash(tree[i * 2 + 1])))
                next_layer.append(h_val)
            tree = next_layer

        header = str(tree[0])
        sha = hashlib.sha256()
        sha.update(header.encode('utf-8'))
        return sha.hexdigest()


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
                    mgf=padding.MGF1(SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                SHA256()
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
            # add the new block to its own chain
            accepted = self.addBlock(new_block)
            if accepted:
                candidate_block.verifyAllTransactions()

            # broadcast the new block along with the new valid proof
            self.network.broadcast_blocks.append(new_block)

            if not accepted:
                print("ERROR - self mined block not accepted")
            else:
                print(current_hash)

            self.blocks_mined.append(new_block)
            # time.sleep(1)

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


class AppWebpage:
    """
    receive results
    """
    app = Flask(__name__)
    app.debug = True

    def __init__(self, host_ip, port, network):
        self.host_ip = host_ip
        self.port = port
        self.network = network

        bc = BlockChain(previous_chain=None, difficulty=1, block_length=99)
        self.node = Miner(chain=bc, network=self.network)
        self.transactions = None

        self.apply_rules()

    def apply_rules(self):
        """
        sets rules for the server
        binding methods to certain calls
        :return:
        """

        self.app.add_url_rule("/create-user", "create_user", self.create_user)
        self.app.add_url_rule("/list-user", "list_user", self.list_users)

        self.app.add_url_rule("/3a", "3a", self.q3a)
        self.app.add_url_rule("/3b", "3b", self.q3b)

        self.app.add_url_rule("/4", "4", self.q4)
        self.app.add_url_rule("/4_stats", "4 stats", self.q4_stats)

        self.app.add_url_rule("/5t", "5t", self.q5t)
        self.app.add_url_rule("/5a", "5a", self.q5a)
        self.app.add_url_rule("/5b", "5b", self.q5b)

        self.app.add_url_rule("/get-qr", "qr", self.sendQRCode)
        self.app.add_url_rule("/", "home", self.home)

    def home(self):
        options = {"options": {"add user": "/create-user?name=",
                               "list users": "/list-user",
                               "Question 3a": "/3a",
                               "Question 3b": "/3b",
                               "Question 4 mine": "/4?difficulty=<select 1-10>",
                               "Question 4 stats": "/4_stats?difficulty=<select 1-10>",
                               "Question 5_create_transactions": "/5t",
                               "Question 5_time_transactions_trace_verify": "/5a?args=<transaction hash>@<transaction id>@<str transaction time>",
                               "Question 5_trace_all_stats": "/5b", },
                   "advice": "i recommend noting down returned information for later input use",
                   "how to use": "add the extension in options into the url for desired questions. \n make sure to add input option where prompted with <> in url"}
        return jsonify(options)

    def run_debug(self):
        """
        starts the flask server in bu-bug mode
        :return:
        """

        self.app.run(host=self.host_ip, port=self.port, debug=True, use_reloader=False)

    def run(self):
        """
        run server without debugging
        """
        port = int(os.environ.get("PORT", self.port))
        self.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    def q3a(self):
        return jsonify(q_3a())

    def q3b(self):
        return jsonify(q_3b())

    def q4(self):
        difficulty = request.args.get("difficulty")
        details = fullTransactionAdditionMine(difficulty=int(difficulty))

        return jsonify(details)

    def q4_stats(self):
        difficulty = request.args.get("difficulty")
        results = q4(redundancy=1, leading_zeros=int(difficulty), energy=False)

        time_list = []
        hashes_lst = []
        energy_lst = []

        for d in results:
            d_time = []
            d_hash = []
            d_energy = []
            for i in d:
                if (i[0] is not None) and not (np.isnan(i[0])) and (i[0] != -1):
                    d_time.append(i[0])
                    d_hash.append(i[1])
                    d_energy.append(i[2])

            if len(d_time) != 0:
                time_list.append(d_time)
                hashes_lst.append(d_hash)
                energy_lst.append(d_energy)

        # 4a)
        t_mean = np.mean(time_list, axis=1)
        t_variance = np.var(time_list, axis=1)
        t_std = np.std(time_list, axis=1)

        # 4b)
        h_mean = np.mean(hashes_lst, axis=1)
        h_variance = np.var(hashes_lst, axis=1)
        h_std = np.std(hashes_lst, axis=1)

        e_mean = np.mean(energy_lst, axis=1)
        e_variance = np.var(energy_lst, axis=1)
        e_std = np.std(energy_lst, axis=1)

        def dificulty_display(lst):
            return {f"difficulty {j + 1}": v for v, j in zip(list(lst), [i for i in range(len(list(lst)))])},

        details = {"time mean": dificulty_display(t_mean),
                   "hash number mean": dificulty_display(h_mean),
                   "energy mean": dificulty_display(e_mean),
                   "time variance": dificulty_display(t_variance),
                   "hash number variance": dificulty_display(h_variance),
                   "energy variance": dificulty_display(e_variance),
                   "time std": dificulty_display(t_std),
                   "hash number std": dificulty_display(h_std),
                   "energy std": dificulty_display(e_std)
                   }

        return jsonify(details)

    def q5t(self):
        """
        creates transactions and returns the details
        :return:
        """

        transaction_details, transactions = q5CreateTransacitons(self.node)
        self.transactions = transactions
        rv = {"number of transactions": len(self.transactions),
              "number of blocks": len(self.node.chain.blocks)}

        return jsonify([rv, {"transactions": transaction_details}])

    def q5a(self):
        """
        track one transaction
        :return:
        """
        args = str(request.args.get("args")).split("@")

        hash = args[0]
        id = args[1]
        time = args[2]

        traced = q5TraceTransaction(self.node, t_h=hash, t_id=id, t_t=time)
        traced["sender"] = traced["sender"].details()
        return jsonify(traced)

    def q5b(self):
        """
        track all transactions and fidn the mean time
        :return:
        """
        if self.transactions is None:
            return jsonify({"error": "please create some transactions first"})
        trace_results = q5TraceAllTransactions(node=self.node, transactions=self.transactions)

        return jsonify({"number of transactions": len(trace_results["transaction"]),
                        "number of blocks": len(self.node.chain.blocks),
                        "average time to track": trace_results["average_time"]})

    def create_user(self, ):
        error = None

        name = request.args.get("name")
        new_user = Client(name=name, network=self.network)
        self.network.users.append(new_user)

        details = new_user.details()

        page = f"""<!DOCTYPE html>
                            <html>
                            <body>

                            <h1>New User: {name}</h1>

                            <h2>Details</h2>

                            <p>
                                <h3>private key</h3>
                                {details["private key"]}
                                <h3>wallet address</h3>
                                {details["wallet address"]}
                            </p>

                            <h2>Wallet Address QR code</h2>
                            <img src="/get-qr?id={details['QR code']}">



                            </body>
                            </html>"""

        return render_template_string(page)

    def list_users(self):
        users = []
        for u in self.network.users:
            users.append(u.details())
        return jsonify(users)

    def sendQRCode(self, ):
        name = request.args.get("id")

        return send_file(name, mimetype='image/gif')


def startFlaskApp():
    """
    starts the webapp
    :return:
    """
    wp = AppWebpage(host_ip="0.0.0.0", port=5000, network=network)
    wp.run_debug()


def q_2a():
    names = ["farmer", "manufacturer", "customer", "retailer"]

    for name in names:
        new_user = Client(name=name, network=network)
        network.users.append(new_user)

        details = new_user.details()
        print(details)


def q_3a():
    bc = BlockChain(previous_chain=None, difficulty=5, block_length=99)
    genesis = bc.blocks[-1]

    return {"time_stamp": genesis.time_stamp,
            "transaction_count": genesis.transaction_count,
            "previous_block_hash": genesis.previous_block_hash,
            "nonce": genesis.nonce,
            "genesis_hash": genesis.genHash(),
            "genesis_ID": genesis.id,
            "genesis_transaction": genesis.transactions[0].details(),
            }


def q_3b():
    network = Network()
    bc = BlockChain(previous_chain=None, difficulty=5, block_length=99)

    node_a = Miner(chain=bc, network=network)

    client_a = Client(name="farmer", network=network)
    client_b = Client(name="factory", network=network)
    client_c = Client(name="retailer", network=network)
    client_d = Client(name="customer", network=network)

    null_item = Item(value=None)
    raw_material = Item(10, description="raw material")
    manufactured_goods = Item(10, description="manufactured material")
    product = Item(10, description="product")

    client_a.sendTransaction(receivers=[client_b.key_pair.public_key_str],
                             inputs=[null_item],
                             outputs=[raw_material])

    client_b.sendTransaction(receivers=[client_c.key_pair.public_key_str],
                             inputs=[raw_material],
                             outputs=[manufactured_goods])

    client_c.sendTransaction(receivers=[client_d.key_pair.public_key_str],
                             inputs=[manufactured_goods],
                             outputs=[product])

    client_d.sendTransaction(receivers=[client_c.key_pair.public_key_str],
                             inputs=[null_item],
                             outputs=[Item(10, description="payment")])

    client_c.sendTransaction(receivers=[client_b.key_pair.public_key_str],
                             inputs=[null_item],
                             outputs=[Item(10, description="payment")])

    client_b.sendTransaction(receivers=[client_a.key_pair.public_key_str],
                             inputs=[null_item],
                             outputs=[Item(10, description="payment")])

    node_a.receive_transactions()

    node_a.mine(attempts=1)

    transaction_details = {f"t_{i.genHash()}": i.details() for i in node_a.chain.blocks[-1].transactions}

    results = {"valid nonce": node_a.chain.blocks[-1].nonce,
               "block hash": node_a.chain.blocks[-1].genHash(),
               "previous block hash": node_a.chain.blocks[-1].previous_block_hash,
               "transactions": transaction_details}

    return results


def q4(redundancy=5, leading_zeros=10, energy=True):
    """

    :param redundancy: number of times experiment looped for each difficulty level
    :param leading_zeros: difficulty level of hash to be found
    :param energy: weather to track energy usage [REQUIRES TO BE RUN AS SUPERUSER]
    :return:
    """
    results = [[(np.nan, np.nan, np.nan) for i in range(redundancy)] for j in range(leading_zeros)]

    for lz in range(1, leading_zeros + 1):
        print(f"Leading 0s: {lz}")

        # analysis of time, hashes comuted, and energy usage

        continue_calc = True
        for i in range(redundancy):
            print(f"{i}/{redundancy}")

            network = Network()
            bc = BlockChain(previous_chain=None, difficulty=lz, block_length=99)

            # setup an irrelevant transaction

            client_a = Client(name="Alice", network=network)
            client_b = Client(name="Bob", network=network)
            client_a.balance = 100
            item_to_send = Item(random.randint(1, 100))
            client_a.sendTransaction(receivers=[client_b.key_pair.public_key_str],
                                     inputs=[item_to_send],
                                     outputs=[item_to_send])

            if continue_calc:

                # randomise the nonces so that have different proof starting points
                bc.blocks[0].nonce = random.randint(1, 100)
                node_a = Miner(chain=bc, network=network)
                node_a.receive_transactions()

                # 4)a) 4)b) get hashes computed and the time taken
                time_taken, hashes_computed = node_a.mine(attempts=1, track=True)
                if time_taken == -1:
                    print("It’s very difficult to find nonce")
                print(time_taken)

                # 4)c) find the energy that is used
                try:
                    if energy:
                        if time_taken != -1:
                            energy_res = energyusage.evaluate(node_a.mine, 1, True, energyOutput=True,
                                                              printToScreen=False)
                        else:
                            energy_res = [0, 0, 0]
                    else:
                        energy_res = [0, 0, 0]
                except:
                    energy_res = [0, 0, 0]

                if time_taken == -1:
                    continue_calc = False

                results[lz - 1][i] = (time_taken, hashes_computed, energy_res[1])

    return results


def q4Display(results, graph=True, energy=True):
    time_list = []
    hashes_lst = []
    energy_lst = []

    for d in results:
        d_time = []
        d_hash = []
        d_energy = []
        for i in d:
            if (i[0] is not None) and not (np.isnan(i[0])) and (i[0] != -1):
                d_time.append(i[0])
                d_hash.append(i[1])
                d_energy.append(i[2])

        if len(d_time) != 0:
            time_list.append(d_time)
            hashes_lst.append(d_hash)
            energy_lst.append(d_energy)

    # 4a)
    t_mean = np.mean(time_list, axis=1)
    t_variance = np.var(time_list, axis=1)
    t_std = np.std(time_list, axis=1)

    # 4b)
    h_mean = np.mean(hashes_lst, axis=1)
    h_variance = np.var(hashes_lst, axis=1)
    h_std = np.std(hashes_lst, axis=1)

    if energy:
        # 4c)
        e_mean = np.mean(energy_lst, axis=1)
        e_variance = np.var(energy_lst, axis=1)
        e_std = np.std(energy_lst, axis=1)
    else:
        e_mean = 0
        e_variance = 0
        e_std = 0
        energy_lst = []

    details = {"time mean": t_mean,
               "hash number mean": h_mean,
               "energy mean": e_mean,
               "time variance": t_variance,
               "hash number variance": h_variance,
               "energy variance": e_variance,
               "time std": t_std,
               "hash number std": h_std,
               "energy std": e_std
               }
    res = {"times list": time_list, "hashes list": hashes_lst, "energy list": energy_lst, "details": details}

    details = {"time mean": t_mean,
               "hash number mean": h_mean,
               "energy mean": e_mean,
               "time variance": t_variance,
               "hash number variance": h_variance,
               "energy variance": e_variance,
               "time std": t_std,
               "hash number std": h_std,
               "energy std": e_std
               }
    print(details)

    if graph:
        # Create a figure with 6 subplots
        fig, axs = plt.subplots(2, 3)
        # plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.0)
        fig.tight_layout(h_pad=3, w_pad=2)

        # 4a)

        axs[0, 0].errorbar(range(len(t_mean)), t_mean, yerr=t_std, fmt="o", capsize=5)
        axs[0, 0].semilogy(t_mean)
        axs[0, 0].set_title("time mean")
        axs[0, 0].set_xlabel("length of difficulty")
        axs[0, 0].set_ylabel("seconds")

        axs[1, 0].plot(t_variance)
        axs[1, 0].set_title("time variance")
        axs[1, 0].set_xlabel("length of difficulty")
        axs[1, 0].set_ylabel("seconds")

        # 4b)

        axs[0, 1].errorbar(range(len(h_mean)), h_mean, yerr=h_std, fmt="o", capsize=5)
        axs[0, 1].semilogy(h_mean)
        axs[0, 1].set_title("hash number mean")
        axs[0, 1].set_xlabel("length of difficulty")
        axs[0, 1].set_ylabel("Hashes computed")

        axs[1, 1].semilogy(h_variance)
        axs[1, 1].set_title("hash number variance")
        axs[1, 1].set_xlabel("length of difficulty")
        axs[1, 1].set_ylabel("Hashes computed")

        # 4c)

        axs[0, 2].errorbar(range(len(e_mean)), e_mean, yerr=e_std, fmt="o", capsize=5)
        axs[0, 2].semilogy(e_mean)
        axs[0, 2].set_title("energy mean")
        axs[0, 2].set_xlabel("length of difficulty")
        axs[0, 2].set_ylabel("KWh")

        axs[1, 2].semilogy(e_variance)
        axs[1, 2].set_title("energy variance")
        axs[1, 2].set_xlabel("length of difficulty")
        axs[1, 2].set_ylabel("KWh")

        # add grids
        axs[0, 0].grid(color='green', linestyle='--', linewidth=0.5)
        axs[1, 0].grid(color='green', linestyle='--', linewidth=0.5)
        axs[0, 1].grid(color='green', linestyle='--', linewidth=0.5)
        axs[1, 1].grid(color='green', linestyle='--', linewidth=0.5)
        axs[0, 2].grid(color='green', linestyle='--', linewidth=0.5)
        axs[1, 2].grid(color='green', linestyle='--', linewidth=0.5)

        # Show the plot
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
            "#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3",
        ])

        plt.rcParams['figure.figsize'] = [100, 100]
        plt.savefig(fname="graph.pdf", format="png")
        plt.show(block=True)


def fullTransactionAdditionMine(difficulty=1):
    """
    functions that shows vlidation of transactions in an added block
    :return:
    """
    network = Network()
    bc = BlockChain(previous_chain=None, difficulty=difficulty, block_length=99)

    node_a = Miner(chain=bc, network=network)

    client_a = Client(name="Alice", network=network)
    client_b = Client(name="Bob", network=network)

    item_to_send = Item(value=10)
    null_item = Item(value=None)

    client_a.sendTransaction(receivers=[client_b.key_pair.public_key_str],
                             inputs=[null_item],
                             outputs=[item_to_send])

    node_a.receive_transactions()

    node_a.mine(attempts=1)

    transaction_details = {f"t_{i.genHash()}": i.details() for i in node_a.chain.blocks[-1].transactions}

    results = {"valid nonce": node_a.chain.blocks[-1].nonce,
               "block hash": node_a.chain.blocks[-1].genHash(),
               "transactions": transaction_details}

    return results


def q5CreateTransacitons(node):
    """
    functions that shows vlidation of transactions in an added block
    :return: average time taken to find transaction
    """
    transactions = []

    bc = BlockChain(previous_chain=None, difficulty=1, block_length=99)

    node_a = node

    client_a = Client(name="Alice", network=network)
    client_b = Client(name="Bob", network=network)
    client_c = Client(name="charlie", network=network)
    network.users.extend([client_c, client_b, client_a])

    item_to_send = Item(value=10)
    null_item = Item(value=None)

    # add a bunch of linked transactions

    i1 = Item(value=10)
    i2 = Item(value=10)
    i3 = Item(value=10)

    t_h, t1 = client_a.sendTransaction(receivers=[client_a.key_pair.public_key_str],
                                       inputs=[Item(value=None)],
                                       outputs=[i1])
    t_h, t2 = client_b.sendTransaction(receivers=[client_b.key_pair.public_key_str],
                                       inputs=[Item(value=None)],
                                       outputs=[i2])
    t_h, t3 = client_c.sendTransaction(receivers=[client_c.key_pair.public_key_str],
                                       inputs=[Item(value=None)],
                                       outputs=[i3])

    transactions.extend([t1, t2, t3])

    node_a.receive_transactions()
    node_a.mine(attempts=1)

    for i in range(100):
        o1 = Item(value=5)
        o2 = Item(value=5)
        o3 = Item(value=5)
        o4 = Item(value=5)
        o5 = Item(value=5)
        o6 = Item(value=5)

        # all send 5 to both neighbors
        t_h, t4 = client_a.sendTransaction(
            receivers=[client_b.key_pair.public_key_str, client_c.key_pair.public_key_str],
            inputs=[i1],
            outputs=[o1, o2])

        t_h, t5 = client_b.sendTransaction(
            receivers=[client_a.key_pair.public_key_str, client_c.key_pair.public_key_str],
            inputs=[i2],
            outputs=[o3, o4])

        t_h, t6 = client_c.sendTransaction(
            receivers=[client_b.key_pair.public_key_str, client_a.key_pair.public_key_str],
            inputs=[i3],
            outputs=[o5, o6])

        node_a.receive_transactions()

        node_a.mine(attempts=1)

        i1 = Item(value=10)
        i2 = Item(value=10)
        i3 = Item(value=10)

        # all merge the 2*5 they have received from neighbors
        t_h, t7 = client_a.sendTransaction(receivers=[client_a.key_pair.public_key_str],
                                           inputs=[o3, o6],
                                           outputs=[i1])

        t_h, t8 = client_b.sendTransaction(receivers=[client_b.key_pair.public_key_str],
                                           inputs=[o1, o5],
                                           outputs=[i2])

        t_h, t9 = client_c.sendTransaction(receivers=[client_c.key_pair.public_key_str],
                                           inputs=[o2, o4],
                                           outputs=[i3])

        transactions.extend([t4, t5, t6, t7, t8, t9])

        node_a.receive_transactions()

        node_a.mine(attempts=1)

    t_details = dict()
    for t in transactions:
        t_details.update({f"{t.id}": {"id": t.id,
                                      "hash": t.genHash(),
                                      "time": str(t.time)}})

    return t_details, transactions


def q5TraceTransaction(node, t_h, t_t, t_id):
    """

    :param node: node that contains a copy of the blockchain containing the transaction
    :param t_h: transaction hash
    :param t_t: transaciton time
    :param t_id: transaction id
    :return:
    """

    st = time.perf_counter()
    f_t = node.chain.findTransaction(t_hash=t_h, t_datetime=datetime.strptime(str(t_t), "%Y-%m-%d %H:%M:%S.%f"),
                                     t_ID=t_id)

    # verify steakholders and find sender
    sender = node.verifySteakholders(f_t)

    result = {"transaction": None,
              "sender": None,
              "time": None}

    if False != sender:
        result["transaction"] = f_t.details()
        result["sender"] = sender
        result["time"] = (time.perf_counter() - st)

    return result


def q5TraceAllTransactions(node, transactions):
    results = {"transaction": [],
               "sender": [],
               "average_time": []}
    for t in transactions:
        st = time.perf_counter()
        f_t = node.chain.findTransaction(t_hash=t.genHash(), t_datetime=t.time, t_ID=t.id)

        # verify steakholders and find sender
        sender = node.verifySteakholders(f_t)

        if False != sender:
            results["transaction"].append(f_t)
            results["sender"].append(sender)
            results["average_time"].append(time.perf_counter() - st)
    results["average_time"] = sum(results["average_time"]) / len(results["average_time"])
    return results


if __name__ == '__main__':
    network = Network()
    bc = BlockChain(previous_chain=None, difficulty=1, block_length=99)
    node = Miner(chain=bc, network=network)

    # !!!!Flask app!!!!
    # uncomment below to run the questions in a more displayable format. this is up to you.
    # although if something does not work in either way or another please do run it in the other one!

    # startFlaskApp()

    # q2
    """
    # creates the 4 users and puts their QR codes in to the workign DIR
    # i would recommend instead running this in the flask app as it displays the QR code there.
    q_2a()
    """

    # q3
    """
    # here dispalys a simple output of the minned genesis/ and/or regular blocks
    print(q_3a())
    print()
    """

    """
    print(q_3b())
    print()
    """

    # q4
    """
    
    calculate_energy = True
    # to get a better output or results run -Question 4 stats- in the flask app
    results = q4(redundancy=5, leading_zeros=10, energy=calculate_energy)
    
    #will output the graph
    q4Display(results, graph=True, energy=calculate_energy)
    """

    # q5

    """# running in the flask app will give a better visualization of output
    
    #first create the transactions
    transaction_details, transactions = q5CreateTransacitons(node)
    
    # pick a transaction to track
    to_trace = transaction_details[list(transaction_details.keys())[0]]
    
    # trace the transaction
    traced = q5TraceTransaction(node, t_h=to_trace["hash"], t_id=to_trace["id"], t_t=to_trace["time"])
    print(traced)
    
    # trace all transactions
    trace_results = q5TraceAllTransactions(node=node, transactions=transactions)
    print(trace_results)"""
