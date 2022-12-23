from client import Client


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

