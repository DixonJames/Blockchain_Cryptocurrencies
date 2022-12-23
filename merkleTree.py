
class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self.findRoot()

    def findRoot(self):
        tree = self.transactions

        while len(tree) != 1:
            next_layer = []
            if len(tree) % 2 != 0:
                tree.append(tree[-1])
            for i in range(int(len(tree) / 2)):
                h_val = hash(str(hash(tree[i * 2])) + str(hash(tree[i * 2 + 1])))
                next_layer.append(h_val)
            tree = next_layer

        return tree[0]
