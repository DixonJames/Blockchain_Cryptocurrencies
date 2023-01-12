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
import json, pickle

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
                               "Question 4 mine": "/4?difficulty=[select 1-10]",
                               "Question 4 stats": "/4_stats?difficulty=[select 1-10]",
                               "Question 5_create_transactions": "/5t",
                               "Question 5_time_transactions_trace_verify": "/5a?args=<transaction hash>@<transaction id>@<str transaction time>",
                               "Question 5_trace_all_stats": "/5b", },
                   "advice": "i recommend noting down returned information for later input use"}
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
        details = full_transaction_addition_mine(difficulty=int(difficulty))

        return jsonify(details)

    def q4_stats(self):
        difficulty = request.args.get("difficulty")
        times, hashes, energy, details = q4(redundancy=1, leading_zeros=int(difficulty), energy_lst=False)

        return jsonify(details)

    def q5t(self):
        """
        creates transactions and returns the details
        :return:
        """

        transaction_details, transactions = q5_create_transacitons(self.node)
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

        traced = q5_trace_transaction(self.node, t_h=hash, t_id=id, t_t=time)
        traced["sender"] = traced["sender"].details()
        return jsonify(traced)

    def q5b(self):
        """
        track all transactions and fidn the mean time
        :return:
        """
        if self.transactions is None:
            return jsonify({"error": "please create some transactions first"})
        trace_results = q5_trace_all_transactions(node=self.node, transactions=self.transactions)

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


def start_interactive_app():
    """
    starts the webapp
    :return:
    """
    wp = AppWebpage(host_ip="0.0.0.0", port=5000)
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
    results = [[None for i in range(redundancy)] for j in range(leading_zeros)]

    for lz in range(1, leading_zeros+1):
        print(f"zeros: {lz}")


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
                print(time_taken)

                # 4)c) find the energy that is used
                try:
                    if energy:
                        energy_res = energyusage.evaluate(node_a.mine, 1, True, energyOutput=True, printToScreen=False)

                    else:
                        energy_res = [0, 0, 0]
                except:
                    energy_res = [0, 0, 0]

                if time_taken == -1:
                    continue_calc = False

                results[lz - 1][i] = (time_taken, hashes_computed, energy_res[1])

    # calcualte averages and plots
    time_list = [([i[0] for i in d]) for d in results if d[0] is not None]
    hashes_lst = [([i[1] for i in d]) for d in results if d[0] is not None]

    if energy:
        energy_lst = [([i[2] for i in d]) for d in results if d[0] is not None]

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

    with open('filename.pickle', 'wb') as file:
        pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)

    return time_list, hashes_lst, energy_lst, details


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


def full_transaction_addition_mine(difficulty=1):
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


def q5_create_transacitons(node):
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


def q5_trace_transaction(node, t_h, t_t, t_id):
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


def q5_trace_all_transactions(node, transactions):
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

    # q2
    # q_2a()

    # q3
    """
    print(q_3a())
    print()"""

    """
    print(q_3b())
    print()"""

    # q5

    """    transaction_details, transactions = q5_create_transacitons(node)

    to_trace = transaction_details[list(transaction_details.keys())[0]]
    traced = q5_trace_transaction(node, t_h=to_trace["hash"], t_id=to_trace["id"], t_t=to_trace["time"])

    trace_results = q5_trace_all_transactions(node=node, transactions=transactions)"""

    # q4
    #times, hashes, energy = q4(redundancy=2, leading_zeros=5, energy=True)
    with open('filename.pickle', 'rb') as file:
        res = pickle.load(file)


    q4_display(res["times list"], res["hashes list"], res["energy list"])

    #app_page = AppWebpage(host_ip="0.0.0.0", port="5555", network=network)
    #app_page.run_debug()
