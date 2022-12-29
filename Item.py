import hashlib
import uuid


class Item:
    def __init__(self, value, recipient=None, description=None):
        self.id = uuid.uuid4()

        if value is not None:
            self.value = value
        else:
            #for the null object that
            self.value = -1

        if self.value == -1:
            self.description = "currency creation item"
        else:
            self.description = description

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
