"""
Part of submission for COMP4137 coursework 2022/3.
Code is property of ldzc78.
Distribution of code is only allowed under circumstances for marking as part of COMP4137 coursework.
"""
import matplotlib.pyplot as plt
# for keypair
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
# for viewing QR code
from flask import Flask, render_template_string, request, jsonify, send_file
import qrcode

import os
import random
import uuid
import numpy as np
import energyusage
import hashlib
import base64
import time
from datetime import datetime

app = Flask(__name__)


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

        # self.current_user = None

        self.apply_rules()

    def apply_rules(self):
        """
        sets rules for the server
        binding methods to certain calls
        :return:
        """

        self.app.add_url_rule("/create-user", "create_user", self.create_user)
        self.app.add_url_rule("/get-qr", "qr", self.sendQRCode)
        self.app.add_url_rule("/", "home", self.home)

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

    def home(self):
        options = {"add user": "/create-user?name="}
        return jsonify(options)

    def create_user(self, ):
        error = None

        name = request.args.get("name")
        new_user = Client(name)
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

    def sendQRCode(self, ):
        name = request.args.get("id")

        return send_file(name, mimetype='image/gif')


class Keypair:
    def __init__(self):
        self.priv_key, self.public_key = self.createKeyPair()

        self.verify_padding = padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
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
        :param keypair:
        :return:
        """
        private = str(self.priv_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ), 'utf-8').split('-----')[2].replace('\n', '')

        public = str(self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ), 'utf-8').split('-----')[2].replace('\n', '')

        return private, public

    def sign(self, data):
        signature = self.priv_key.sign(
            str(data).encode('utf-8'),
            self.verify_padding,
            hashes.SHA256()
        )

        return base64.b64encode(signature)

    def verify(self, data, signature):
        try:
            signature = base64.b64decode(signature)
            self.public_key.verify(
                signature,
                data,
                self.verify_padding,
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.findRoot()

    def findRoot(self):
        tree = self.transactions

        while len(tree) != 1:
            next_layer = []
            if len(tree) % 2 != 0:
                tree.append(tree[-1])
            for i in range(int(len(tree) / 2)):
                h_val = hash(str(hash(tree[i * 2])) + str(hash(tree[i * 2 + 1])))
                next_layer.append(h_val)
            tree = next_layer

        return tree[0]


class Item:
    def __init__(self, value, recipient):
        self.id = uuid.uuid4()
        self.value = value

        self.description = None

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
        self.identifier = name
        self.key_pair = Keypair()
        self.qr_code = self.genQRCOde()
        self.network = network

        self.balance = 0

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

        img.save(os.path.join(os.getcwd(), pub_key_string[:10] + ".png"), format="PNG")

        return img

    def details(self):
        """
        returns dict of user details
        :return:
        """
        return {"private key": self.key_pair.keyStrings()[0],
                "wallet address": self.key_pair.keyStrings()[1],
                "QR code": f"{self.key_pair.keyStrings()[1][:10]}.png"}

    def sendTransaction(self, receiver, inputs: [Item], outputs: [Item]):
        transaction = Transaction(sender=self, receivers=receiver, inputs=inputs, outputs=outputs)

        if transaction.verify():
            self.network.broadcastTransaction(transaction)
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


        :param sender: a Client object
        :param receivers: a wallet address
        :param inputs:
        :param outputs:
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
        if in_sum != out_sum:
            return False

        if len(self.outputs) != self.receivers:
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
        inicaitor_signature = transaction.signature
        try:
            signature = base64.b64decode(inicaitor_signature)
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

        # check that inputs go to same as signature of iniciator
        input_items = transaction.inputs
        for i in input_items:
            # find block in chain from items block hash
            # transaction in block from items transaction hash
            item_creation_transaction = self.chain.hash_table[i.block_hash].hash_table[i.transaction_hash]

            original_transaction_recipient = None
            # go though the creation transaction and find the original recipient for the item
            for output_i in range(len(item_creation_transaction.outputs)):
                if item_creation_transaction.outputs[output_i].id == item_creation_transaction.id:
                    original_transaction_recipient = item_creation_transaction.reciepients[output_i]

            if original_transaction_recipient is None:
                # didint find the item in the transaction
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
        adss a transaction issusing 100 value to user 0 as the genesis transaciton
        :return:
        """
        # create a base user
        user = Client("Satoshi", network=Network())
        user_addr = user.key_pair.public_key_str

        # create ficitious mystry items
        start_input = Item(value=100)
        start_output = Item(value=100)

        # create and sighn transaction
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
        :param genesis: bool to change the previous hash depending on if genesis block or not
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
                      chain=self)

        self.hash_dict.update({f"{block.genHash()}": 0})

        # append the block
        self.blocks.append(block)

    def addBlock(self, block: Block):
        """
        adds a block to the chain only if the chain is valid with it
        :param block:
        :return: a boolean as to weather or not the block is accepted
        """
        self.blocks.append(block)
        if not self.verify_chain():
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

            hash_proofs = self.hashProofs(current_proof=current.nonce, previous_proof=previous.nonce)
            if hash_proofs[:self.difficulty] != "0" * self.difficulty:
                return False

        return True


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

        self.processed_transactions_currently_unreceived = self.processed_transactions_currently_unreceived.extend(
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

    def getPreviousProof(self):
        """
        :return: the previous proof braodcast to the nework
        """
        if len(self.network.broadcast_blocks) != 0:
            return self.network.broadcast_blocks[-1].nonce
        return self.chain.blocks[0].nonce

    def mine(self, attempts: int = None, track=False):
        """
        mines continuously untill attemts reached
        :param track:
        :param attempts: number of rounds to attempt to mine for
        :return:
        """

        while self.mine_flag:
            previous_proof = self.getPreviousProof()
            if track:
                # 4)a) 4)b)
                hashes_computed, time_taken = self.POW(previous_proof=previous_proof, track_hashes=track,
                                                       time_limmit=float(3600))
                return time_taken, hashes_computed
            else:
                self.POW(previous_proof=previous_proof, track_hashes=track)

            if attempts is not None:
                attempts -= 1
                if attempts == 0:
                    self.mine_flag = False

    def POW(self, previous_proof, track_hashes=False, time_limmit=None):
        """
        performs proof of work.
        attempts to mine the next block
        if successful
            creates, broadcasts and appends block to chain
        otherwise:
            validates the successful broadcast block
        :param time_limmit: stops analysis after this periould of time
        :param track_hashes: bool to turn the method to just return the number of hashes nessaiiryt o find valid proof
        :param previous_proof: the nonce found in the previous block
        :return:
        """
        # 4)a)
        start_time = time.perf_counter()

        current_blocks_mined = self.network.blocks_mined

        current_hash = self.start_hash
        candidate_block = Block(transactions=self.getTransactionsToProcess(),
                                index=len(self.chain.blocks) + 1,
                                previous_hash=previous_proof,
                                max_transaction=self.chain.block_length,
                                chain=self.chain)

        proof = 0
        hash_numer = 0
        while (current_hash[:self.chain.difficulty] != self.success_string) and (
                self.network.blocks_mined == current_blocks_mined) and (
                time.perf_counter() - start_time < time_limmit):
            hash_numer += 1
            proof += 1
            current_hash = candidate_block.genHash(trial_nonce=proof)

        if time.perf_counter() - start_time > time_limmit:
            # 4)a) 4)b)
            return hash_numer, -1

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

            # broadcast the new block along with the new valid proof
            self.network.broadcast_blocks.append(new_block)

            # add the new block to its own chain
            accepted = self.addBlock(new_block)
            if not accepted:
                print("ERROR - self mined block not accepted")
            else:
                print(current_hash)

            self.blocks_mined.append(new_block)
            time.sleep(1)

        else:
            # we have failed to mine a valid proof in time have lost the race
            # need to give up on our own eforts and validate the new block

            # wait for new block to be uploaded
            time.sleep(1)
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


def start_interactive_app():
    """
    starts the webapp
    :return:
    """
    wp = AppWebpage(host_ip="0.0.0.0", port=5000)
    wp.run_debug()


def q_3a():
    bc = BlockChain(previous_chain=None, difficulty=1, block_length=99)
    genesis = bc.blocks[-1]

    return {"time_stamp": genesis.time_stamp,
            "transaction_count": genesis.transaction_count,
            "previous_block_hash": genesis.previous_block_hash,
            "nonce": genesis.nonce,
            "genesis_hash": genesis.genHash(),
            "genesis_transaction": genesis.transactions[0].details(),
            }


def q_3b():
    network = Network()
    bc = BlockChain(previous_chain=None, difficulty=1, block_length=99)

    node_a = Miner(chain=bc, network=network)

    client_a = Client(name="Alice", network=network)
    client_b = Client(name="Bob", network=network)

    client_a.balance = 100
    item_to_send = Item(10)

    client_a.sendTransaction(receiver=client_b.key_pair.public_key_str,
                             inputs=[item_to_send],
                             outputs=[item_to_send])

    # add some more transactions for submit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    node_a.receive_transactions()

    node_a.mine(attempts=1)

    transaction_details = {f"t_{i.genHash()}": i.details() for i in node_a.chain.blocks[-1].transactions}

    results = {"valid nonce": node_a.chain.blocks[-1].nonce,
               "block hash": node_a.chain.blocks[-1].genHash(),
               "transactions": transaction_details}

    return results


def q4(redundancey=5, leading_zeros=10):
    results = [[None for i in range(redundancey)] for j in range(leading_zeros)]

    for lz in range(1, leading_zeros):
        network = Network()
        bc = BlockChain(previous_chain=None, difficulty=lz, block_length=99)

        # setup an irrelevant transaction

        client_a = Client(name="Alice", network=network)
        client_b = Client(name="Bob", network=network)
        client_a.balance = 100
        item_to_send = Item(random.randint(1, 100))
        client_a.sendTransaction(receiver=client_b.key_pair.public_key_str,
                                 inputs=[item_to_send],
                                 outputs=[item_to_send])

        # analysis of time, hashes comuted, and energy usage

        continue_calc = True
        for i in range(redundancey):
            if continue_calc:

                # randomise the nonces so that have different proof starting points
                bc.blocks[0].nonce = random.randint(1, 100)
                node_a = Miner(chain=bc, network=network)
                node_a.receive_transactions()

                # 4)a) 4)b) get hashes computed and the time taken
                time_taken, hashes_computed = node_a.mine(attempts=1, track=True)

                # 4)c) find the energy that is used
                try:
                    # energy_res = energyusage.evaluate(node_a.mine, 1, True, energyOutput=True, printToScreen=False)
                    energy_res = [0, lz, 0]
                except:
                    energy_res = [0, 0, 0]

                if time_taken == -1:
                    continue_calc = False

                results[lz - 1][i] = (time_taken, hashes_computed, energy_res[1])

        # calcualte averages and plots
    times = [([i[0] for i in d]) for d in results if d[0] is not None]
    hashes = [([i[1] for i in d]) for d in results if d[0] is not None]
    energy = [([i[2] for i in d]) for d in results if d[0] is not None]

    return times, hashes, energy


def q4_display(times, hashes, energy, graph=True):
    # 4a)
    t_mean = np.mean(times, axis=1)
    t_variance = np.var(times, axis=1)
    t_std = np.std(times, axis=1)

    # 4b)
    h_mean = np.mean(hashes, axis=1)
    h_variance = np.var(hashes, axis=1)
    h_std = np.std(hashes, axis=1)

    # 4c)
    e_mean = np.mean(energy, axis=1)
    e_variance = np.var(energy, axis=1)
    e_std = np.std(energy, axis=1)

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

        # 4a)

        axs[0, 0].errorbar(range(len(t_mean)), t_mean, yerr=t_std, fmt="o", capsize=5)
        axs[0, 0].plot(t_mean)
        axs[0, 0].set_title("time mean")
        axs[0, 0].set_xlabel("leading Zeros")
        axs[0, 0].set_ylabel("seconds")

        axs[1, 0].plot(t_variance)
        axs[1, 0].set_title("time variance")
        axs[0, 0].set_xlabel("leading Zeros")
        axs[0, 0].set_ylabel("seconds")

        # 4b)

        axs[0, 1].errorbar(range(len(h_mean)), h_mean, yerr=h_std, fmt="o", capsize=5)
        axs[0, 1].plot(h_mean)
        axs[0, 1].set_title("hash number mean")
        axs[0, 0].set_xlabel("leading Zeros")
        axs[0, 0].set_ylabel("Hashes computed")

        axs[1, 1].plot(h_variance)
        axs[1, 1].set_title("hash number variance")
        axs[0, 0].set_xlabel("leading Zeros")
        axs[0, 0].set_ylabel("Hashes computed")

        # 4c)

        axs[0, 2].errorbar(range(len(e_mean)), e_mean, yerr=e_std, fmt="o", capsize=5)
        axs[0, 2].plot(e_mean)
        axs[0, 2].set_title("energy mean")
        axs[0, 0].set_xlabel("leading Zeros")
        axs[0, 0].set_ylabel("KWh")

        axs[1, 2].plot(e_variance)
        axs[1, 2].set_title("energy variance")
        axs[0, 0].set_xlabel("leading Zeros")
        axs[0, 0].set_ylabel("KWh")

        # Show the plot
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
            "#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3",
            "#03A9F4", "#00BCD4", "#009688", "#4CAF50", "#8BC34A", "#CDDC39",
            "#FFEB3B", "#FFC107", "#FF9800", "#FF5722", "#795548", "#9E9E9E",
            "#607D8B",
        ])

        plt.show()


def q5():
    network = Network()
    bc = BlockChain(previous_chain=None, difficulty=1, block_length=99)

    node_a = Miner(chain=bc, network=network)

    client_a = Client(name="Alice", network=network)
    client_b = Client(name="Bob", network=network)

    client_a.balance = 100
    item_to_send = Item(10)

    client_a.sendTransaction(receiver=client_b.key_pair.public_key_str,
                             inputs=[item_to_send],
                             outputs=[item_to_send])

    # add some more transactions for submit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    node_a.receive_transactions()

    node_a.mine(attempts=1)

    transaction_details = {f"t_{i.genHash()}": i.details() for i in node_a.chain.blocks[-1].transactions}

    results = {"valid nonce": node_a.chain.blocks[-1].nonce,
               "block hash": node_a.chain.blocks[-1].genHash(),
               "transactions": transaction_details}

    return results


if __name__ == '__main__':
    # q5

    # q4
    """times, hashes, energy = q4(redundancey=5, leading_zeros=4)
    q4_display(times, hashes, energy)"""

    """print("q_3a")
    print(q_3a())
    print()"""

    """print("q_3b")
    print(q_3b())
    print()"""
