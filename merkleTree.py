import hashlib

class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.findRoot()

    def findRoot(self):
        tree = [hash(t.genHash()) for t in self.transactions]

        while len(tree) != 1:
            next_layer = []
            if len(tree) % 2 != 0:
                tree.append(tree[-1])
            for i in range(int(len(tree) / 2)):
                h_val = hash(str(hash(tree[i * 2])) + str(hash(tree[i * 2 + 1])))
                next_layer.append(h_val)
            tree = next_layer

        header = str(tree[0])
        sha = hashlib.sha256()
        sha.update(header.encode('utf-8'))
        return sha.hexdigest()
