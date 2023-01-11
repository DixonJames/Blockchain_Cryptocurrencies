from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

import base64


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
