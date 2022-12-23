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

from keypair import Keypair
from merkleTree import MerkleTree
from Item import Item
from client import Client
from transaction import Transaction
from block import Block
from blockchain import BlockChain
from node import Miner, Node
from network import Network


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
