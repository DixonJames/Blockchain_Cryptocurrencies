from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import qrcode
import os
import json
import hashlib
from datetime import datetime
from flask import Flask, render_template, render_template_string,send_from_directory, request, jsonify, g, abort, redirect, url_for, flash, \
    send_file
import flask

app = Flask(__name__)


def createKeyPair():
    """
    creates a new random keypairing
    :return:
    """
    priv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return priv_key


def keyStrings(keypair):
    """
    creates the keypairs keystrings
    :param keypair:
    :return:
    """
    private = str(keypair.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ), 'utf-8').split('-----')[2].replace('\n', '')
    public = str(keypair.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ), 'utf-8').split('-----')[2].replace('\n', '')


    return private, public


class User:
    """
    the calss for a user on the BC
    """

    def __init__(self, name):
        self.identifier = name
        self.secret = createKeyPair()
        self.public = self.secret.public_key()
        self.qr_code = self.genQRCOde()

    def genQRCOde(self):
        """
        creates the Qr code from the public key
        :return:
        """
        qr = qrcode.QRCode(version=1,
                           box_size=10,
                           border=5)
        pub_key_string = keyStrings(self.secret)[1]
        qr.add_data(pub_key_string)
        qr.make(fit=True)

        img = qr.make_image(fill_color='black',
                            back_color='white')

        img.save(os.path.join(os.getcwd(), pub_key_string[:10]+".png"), format="PNG")

        return img

    def details(self):
        """
        returns dict of user details
        :return:
        """
        return {"private key": keyStrings(self.secret)[0],
                "wallet address": keyStrings(self.secret)[1],
                "QR code": f"{keyStrings(self.secret)[1][:10]}.png"}


class Blockchain:
    """
    the blockchain containing everything
    """

    def __init__(self):
        self.users = []
        self.blocks = []

        self.createGenasis()

        self.local_coppy = False

    def createGenasis(self):
        """
        creates the genesis block
        adss a transaction issusing 100 value to user 0 as the genesis transaciton
        :return:
        """
        # create a base user

        self.addUser(User("Satoshi"))

        genesis = Block(0, None, 0)
        genesis_transaction = Transaction(self.users[0].public, self.users[0].public, 100)
        genesis.addTransaction(genesis_transaction)

        self.addBlock(genesis)

    def addUser(self, user: User):
        self.users.append(user)

    def addBlock(self, block):
        self.blocks.append(block)


class Block:
    """
    a block on the chain
    """

    def __init__(self, index, previous_hash, nonce, max_transaction=100):
        self.transactions = []
        self.index = index
        self.previous_hash = previous_hash
        self.nonce = nonce

        self.time = datetime.utcnow()

        self.max_transaction = max_transaction

    def addTransaction(self, transaction):
        if len(self.transactions) < 100:
            self.transactions.append(transaction)

    def __hash__(self):
        hashlib.sha256([str([hash(t) for t in self.transactions])])


class Transaction:
    """
    a transaction on the BC
    """

    def __init__(self, sender, receiver, value):
        self.sender = sender
        self.receiver = receiver
        self.value = value

        self.time = datetime.utcnow()

    def details(self):
        return {"sender": self.sender,
                "receiver": self.receiver,
                "value": self.value,
                "time": self.time}

    def __hash__(self):
        hashlib.sha256(str(self.details()))


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
        new_user = User(name)
        BC.users.append(new_user)

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




if __name__ == '__main__':
    BC = Blockchain()
    wp = AppWebpage(host_ip="0.0.0.0", port=5000)
    wp.run_debug()
