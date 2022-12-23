
import hashlib
import uuid

from client import Client
from Item import Item
from datetime import datetime


class Transaction:
    """
    a transaction on a block in the blockchain
    """

    def __init__(self, sender: Client, receivers: [Client], inputs: [Item], outputs: [Item]):
        """
        indexes of outputs must match up to the indexes of receivers


        param sender: a Client object
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
