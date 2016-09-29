import unittest
import bst
from collections import OrderedDict

class TestBST(unittest.TestCase):
    def setUp(self):
        self.testBST = bst.BST()

    def test_node(self):
        a = bst.BSTNode(10, "Silver")
        self.assertEqual(a.value, "Silver")
        self.assertEqual(a.key, 10)
        self.assertEqual(a.is_root(), True)
        self.assertEqual(a.is_lchild(), False)
        self.assertEqual(a.is_rchild(), False)
        self.assertEqual(a.has_both_children(), False)
        self.assertEqual(a.has_child(), False)

    def test_size(self):
        self.assertEqual(self.testBST.size, 0)
    
    def test_len(self):
        self.assertEqual(len(self.testBST), 0)

    def test_insertroot(self):
        self.assertEqual(len(self.testBST), 0)
        self.testBST[10] = "Silver"
        self.assertEqual(self.testBST.root.key_value, (10, "Silver"))
        self.assertEqual(len(self.testBST), 1)

    def test_deleteroot(self):
        self.assertEqual(len(self.testBST), 0)
        self.testBST[10] = "Silver"
        del self.testBST[10]
        self.assertEqual(self.testBST.root, None)
        self.assertEqual(len(self.testBST), 0)

    def test_getroot(self):
        self.assertEqual(len(self.testBST), 0)
        self.testBST[10] = "Silver"
        self.assertEqual(len(self.testBST), 1)
        node1 = self.testBST[10]
        node2 = self.testBST.get(10)
        self.assertEqual(node1, "Silver")
        self.assertEqual(node2.key_value, (10, "Silver"))

    def test_insertnodes(self):
        n = 0
        self.assertEqual(len(self.testBST), n)
        d = [(10,"silver"), (5,"Hello"), (7,"World"), (3,"Min"), (15,"When"), (20,"Will")]
        for item in d:
            key, value = item[0], item[1]
            self.testBST[key] = value
            n += 1
            self.assertEqual(len(self.testBST), n)
            self.assertEqual(self.testBST[key], value)
    
    def test_removenodes(self):
        n = 0
        self.assertEqual(len(self.testBST), n)
        d = [(10,"silver"), (5,"Hello"), (7,"World"), (3,"Min"), (15,"When"), (20,"Will")]
        for item in d:
            key, value = item[0], item[1]
            self.testBST[key] = value
            n += 1
        del self.testBST[10]
        self.assertEqual(self.testBST.root.key_value, (15, "When"))
        self.assertEqual(self.testBST.size, n - 1)
            
        
        
        
 
if __name__ == "__main__":
    unittest.main()
     