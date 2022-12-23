import os

import qrcode

from Item import Item
from transaction import Transaction
from keypair import Keypair


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
