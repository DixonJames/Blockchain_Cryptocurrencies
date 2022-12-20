from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import qrcode
import os
import json
import uuid
import hashlib
import base64
import time
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)

total_users = []


class Network:
    """
    the network in wich all nodes are on
    """

    def __init__(self):
        self.users = []
        self.broadcast_transactions = []

        self.broadcast_blocks = []
        self.blocks_minned = 0

    def boadcastTransaction(self, transaction):
        self.broadcast_transactions.append(transaction)


class AppWebpage:
    """
    receive results
    """
    app = Flask(__name__)
    app.debug = True

    def __init__(self, host_ip, port):
        self.host_ip = host_ip
        self.port = port

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
        starts the flask server in bubug mode
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
        network.users.append(new_user)

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
        creates a new random keypairing
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
        creates the keypairs keystrings
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

    def sighn(self, data):
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
    def __init__(self, value):
        self.value = value

    def string(self):
        return {"value": self.value}


class Client:
    """
    the class for a user on the BC
    """

    def __init__(self, name, network):
        self.identifier = name
        self.key_pair = Keypair()
        self.qr_code = self.genQRCOde()
        self.network = network

        self.balence = 0

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
        transaction = Transaction(sender=self, receiver=receiver, inputs=inputs, outputs=outputs)

        if transaction.check_valid():
            self.network.boadcastTransaction(transaction)
        else:
            print(f"invalid transaction details:{transaction.details()}")


class Transaction:
    """
    a transaction on a block in the blockchain
    """

    def __init__(self, sender: Client, receiver, inputs: [Item], outputs: [Item]):
        """

        :param sender: a Client object
        :param receiver: a wallet address
        :param inputs:
        :param outputs:
        """
        self.id = uuid.uuid4()
        self.sender = sender.key_pair.public_key_str
        self.receiver = receiver
        self.inputs = inputs
        self.outputs = outputs
        self.time = datetime.utcnow()

        self.signature = sender.key_pair.sighn(str(self.details()))

    def details(self):
        """
        creates a dict of the attributes of the transaction
        :return:
        """
        return {"id": str(self.id),
                "sender": self.sender,
                "receiver": self.receiver,
                "inputs": str([i.string() for i in self.inputs]),
                "outputs": str([i.string() for i in self.outputs]),
                "in value": sum([i.value for i in self.inputs]),
                "out value": sum([i.value for i in self.outputs])}

    def check_valid(self):
        in_sum = sum([i.value for i in self.inputs])
        out_sum = sum([i.value for i in self.outputs])
        if in_sum != out_sum:
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

    def __init__(self, transactions, index=None, previous_hash=None, nonce=None, max_transaction=100, version=1,
                 dificulty=1):
        self.time_stamp = datetime.utcnow()

        self.block_height = None
        self.transactions = []
        self.version = version
        self.id = uuid.uuid4()
        self.index = index
        self.previous_block_hash = previous_hash
        self.nonce = nonce
        self.dificulty = dificulty
        self.max_transaction = max_transaction
        self.transaction_count = 0

        # compute the merkle tree
        self.merkle_tree = MerkleTree(transactions=transactions)

        # add transactions
        self._addTransactions(transactions=transactions)

        # generate the hash
        self.hash = self.genHash()

    def _addTransactions(self, transactions):
        """
        adds transactions to the block if anough room
        called at creation of the block
        :param transactions: list of the transactions to add to the block
        :return:
        """
        for t in transactions:
            if len(self.transactions) < self.max_transaction:
                self.transactions.append(t)
            self.transaction_count += 1
        self.hash = self.genHash()

    def genHash(self):
        """hash the block via header. first comutes merkle root"""
        header = str(
            [self.previous_block_hash, self.merkle_tree.root, self.nonce])
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

        if previous_chain is None:
            self.blocks = []
            self.createGenasis()
        else:
            self.blocks = previous_chain

    def createGenasis(self):
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
                                          receiver=user_addr,
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
                      max_transaction=self.block_length)

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

        # list of transactions received in valid blocks that have not been received individualy (shouldnt happen here)
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
        block_transacitons = block.transactions

        # check that block has no repeited processed transactions
        if len(set(self.processed_transactions_memory).intersection(set(block_transacitons))) != 0:
            return False

        # attempt to add the block to the chain
        agree = self.chain.addBlock(block=block)
        if agree:
            print("block accepted")
            self.other_blocks_accepted.append(Block)

            # update the transactions sets
            self.processTransactions(block_transacitons)

        else:
            print("block rejected")
            self.other_blocks_rejected.append(block)
            return False

        return True


class Miner(Node):
    def __init__(self, chain: BlockChain, network: Network):
        super().__init__(chain=chain, network=network)

        self.mine_flag = True
        self.start_hash = "1" * self.chain.difficulty
        self.success_string = "0" * self.chain.difficulty

    def hashStr(self, s):
        sha = hashlib.sha256()
        sha.update(s.encode('utf-8'))
        return sha.hexdigest()

    def getTransactionsToPorcess(self):
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
        return 0

    def mine(self, attempts: int = None):
        """
        mines continuously untill attemts reached
        :param attempts: number of rounds to attempt to mine for
        :return:
        """
        while self.mine_flag:
            previous_proof = self.getPreviousProof()
            self.POW(previous_proof=previous_proof)

            if attempts is not None:
                attempts -= 1
                if attempts == 0:
                    self.mine_flag = False

    def POW(self, previous_proof):
        """
        performs proof of work.
        attempts to mine the next block
        if successful
            creates, broadcasts and appends block to chain
        otherwise:
            validates the successful broadcast block
        :param previous_proof: the nonce found in the previous block
        :return:
        """
        current_blocks_mined = self.network.blocks_minned

        current_hash = self.start_hash
        proof = 0

        while (current_hash[:self.chain.difficulty] != self.success_string) and (
                self.network.blocks_minned == current_blocks_mined):
            proof += 1
            current_hash = self.chain.hashProofs(current_proof=proof, previous_proof=previous_proof)

        if current_hash[:self.chain.difficulty] == self.success_string and not (
                self.network.blocks_minned != current_blocks_mined):
            # we have a winning proof
            # need to create our new block and broadcast it
            self.network.blocks_minned += 1

            new_block = Block(transactions=self.getTransactionsToPorcess(),
                              index=len(self.chain.blocks) + 1,
                              previous_hash=self.chain.blocks[-1].hash,
                              nonce=proof,
                              max_transaction=self.chain.block_length)

            # broadcast the new block along with the new valid proof
            self.network.broadcast_blocks.append(new_block)

            # add the new block to its own chain
            accepted = self.addBlock(new_block)
            if not accepted:
                print("ERROR - self mined block not accepted")

            self.blocks_mined.append(new_block)
            time.sleep(1)

        else:
            # we have failed to mine a valid proof in time have have lost the race
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

    client_a.balence = 100
    item_to_send = Item(10)

    client_a.sendTransaction(receiver=client_b.key_pair.public_key_str,
                             inputs=[item_to_send],
                             outputs=[item_to_send])

    # add some more transactions for submit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    node_a.receive_transactions()

    node_a.mine(attempts=1)

    transaction_details = {f"t_{i.genHash()}": i.details() for i in node_a.chain.blocks[-1].transactions}

    results = {"valid nonce":node_a.chain.blocks[-1].nonce,
               "block hash":node_a.chain.blocks[-1].genHash(),
               "transactions":transaction_details}

    return results

if __name__ == '__main__':
    print("q_3a")
    print(q_3a())
    print()

    print("q_3b")
    print(q_3b())
    print()
