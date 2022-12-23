import hashlib
import uuid


class Item:
    def __init__(self, value, recipient):
        self.id = uuid.uuid4()
        self.value = value

        self.description = None

        self.transaction_hash = None
        self.block_hash = None
        self.recipient = recipient

    def details(self):
        return {"id": self.id,
                "value": self.value,
                "description": self.description
                }

    def genHash(self):
        """
        hashes item data
        :return:
        """
        header = str(self.details())
        sha = hashlib.sha256()
        sha.update(header.encode('utf-8'))
        return sha.hexdigest()
