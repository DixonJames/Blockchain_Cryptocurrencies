import os

import qrcode

from Item import Item

from keypair import Keypair
from transaction import Transaction


class Client:
    """
    the class for a user on the BC
    """

    def __init__(self, name, network):
        self.key_pair = Keypair()
        self.qr_code_dir = os.path.join(os.getcwd(), "QR_"+ self.key_pair.keyStrings()[1][10:20] + ".png")

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
