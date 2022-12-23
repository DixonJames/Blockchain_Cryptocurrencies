import unittest
from blockchain_app import Keypair

class Testing(unittest.TestCase):
    def test_same(self):
        for i in range(10):
            KP = Keypair()
            data = b"test data"
            signed_data = KP.sign(data)

            verified = KP.verify(data, signed_data)
            print(verified)
            self.assertEqual(verified, True)

    def test_different(self):
        for i in range(10):
            KPa = Keypair()
            KPb = Keypair()
            data = b"test data"
            signed_data = KPa.sign(data)

            verified = KPb.verify(data, signed_data)
            print(verified)
            self.assertEqual(verified, False)

if __name__ == '__main__':

    unittest.main()
