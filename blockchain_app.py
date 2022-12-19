from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import qrcode
import os
import uuid
import hashlib
import base64
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
        self.sender = sender
        self.receiver = receiver
        self.inputs = inputs
        self.outputs = outputs
        self.time = datetime.utcnow()

        self.signature_sender = sender.key_pair.sighn(str(self.details()))

    def details(self):
        """
        creates a dict of the attributes of the transaction
        :return:
        """
        return {"id": self.id,
                "sender": self.sender.key_pair.public_key_str,
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
        self.addBlock(transactions=[genesis_transaction],
                      nonce=0,
                      genesis=True)

    def addBlock(self, transactions: [Transaction], nonce: int, genesis=False):
        """
        adds a block to the chain
        :param transactions: transaction list to be included in the block
        :param nonce: proof found by miner
        :param genesis: bool to change the previous hash depending on if genesis block or not
        :return:
        """
        # change the previous ash depending on if genesis block or not
        if not genesis:
            previous_hash = self.blocks[-1].genHash()
        else:
            previous_hash = 0

        # create the block
        block = Block(transactions=transactions,
                      index=len(self.blocks) + 1,
                      previous_hash=previous_hash,
                      nonce=nonce,
                      max_transaction=self.block_length)

        # append the block
        self.blocks.append(block)

    def verify_chain(self):
        for i in range(len(self.blocks) - 1, 0, -1):
            current: Block = self.blocks[i]
            previous: Block = self.blocks[i - 1]

            if current.hash != current.genHash():
                return False
            if current.previous_block_hash != previous.hash:
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
        self.candidate_block = None

    def receive_transactions(self):
        """
        get unprocessed transactions from the network
        """
        all_transactions = self.network.broadcast_transactions
        unprocessed = [t for t in all_transactions if t not in self.processed_transactions_memory]

        self.unprocessed_transaction_memory.append(unprocessed)
        self.unprocessed_transaction_memory.sort(key=lambda x: x.time)


class Miner(Node):
    def __init__(self, chain):
        super().__init__(chain=chain)

        hash = "1"
        for i in range(self.chain.difficulty):
            hash += "1"
        self.start_hash = hash

        self.sucsess_string = ""
        for i in range(self.chain.difficulty):
            self.sucsess_string += "0"

    def hashStr(self, s):
        sha = hashlib.sha256()
        sha.update(s.encode('utf-8'))
        return sha.hexdigest()

    def POW(self, previous_proof):
        current_blocks_mined = self.network.blocks_minned

        current_hash = self.start_hash
        proof = 0

        while (current_hash[:self.chain.difficulty] != self.sucsess_string) and (self.network.blocks_minned == current_blocks_mined):
            proof+=1
            current_hash = self.hashStr(str(proof**2 - previous_proof**2))


        if current_hash[:self.chain.difficulty] != self.sucsess_string:
            #we have a winning proof
            new_block =


    def MineBlock(self):
        current_blocks_mined = self.network.blocks_minned
        while self.network.blocks_minned == current_blocks_mined:
            self.POW(self.network.)

def start_interactive_app():
    """
    starts the webapp
    :return:
    """
    wp = AppWebpage(host_ip="0.0.0.0", port=5000)
    wp.run_debug()


if __name__ == '__main__':
    network = Network()

    node_a = Node(chain=BlockChain(), network=network)

    client_a = Client(name="Alice", network=network)
    client_b = Client(name="Bob", network=network)

    client_a.balence = 100
    item_to_send = Item(10)

    client_a.sendTransaction(receiver=client_b.key_pair.public_key,
                             inputs=[item_to_send],
                             outputs=[item_to_send])

    node_a.receive_transactions()
