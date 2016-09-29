class BSTNode(object):
    def __init__(self, key, value, lchild=None, rchild=None, parent=None):
        self.key = key
        self.value = value
        self.parent = parent
        self.lchild = lchild
        self.rchild = rchild

    @property
    def key_value(self):
        return (self.key, self.value)

    def is_root(self):
        return not self.parent

    def is_lchild(self):
        if self.parent:
            return self.parent.lchild == self
        return False

    def is_rchild(self):
        if self.parent:
            return self.parent.rchild == self
        return False

    def has_child(self):
        return bool(self.lchild or self.rchild)

    def has_both_children(self):
        return bool(self.lchild and self.rchild)

    def is_leaf(self):
        return not self.has_child()

    def update(self, key, value, lchild, rchild):
        self.key = key
        self.value = value
        self.lchild = lchild
        self.rchild = rchild
        if self.lchild:
            self.lchild.parent = self
        if self.rchild:
            self.rchild.parent = self

    def __str__(self):
        return str(self.value)


class BST(object):
    def __init__(self):
        self.root = None
        self.size = 0

    def __len__(self):
        return self.size

    def insert(self, key, value):
        if not self.root:
            self.root = BSTNode(key, value)
            self.size += 1
        else:
            self._insert(key, value, self.root)

    def _insert(self, key, value, node):
        if node.key == key:
            node.value = value
        elif node.key > key:
            if not node.lchild:
                node.lchild = BSTNode(key, value, parent=node)
                self.size += 1
            else:
                self._insert(key, value, node.lchild)
        elif node.key < key:
            if not node.rchild:
                node.rchild = BSTNode(key, value, parent=node)
                self.size += 1
            else:
                self._insert(key, value, node.rchild)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def get(self, key):
        if self.root.key == key:
            return self.root
        else:
            return self._get(key, self.root)

    def _get(self, key, node):
        if node.key == key:
            return node
        elif node.key > key and node.lchild:
            return self._get(key, node.lchild)
        elif node.key < key and node.rchild:
            return self._get(key, node.rchild)
        else:
            raise KeyError (": " + str(key))

    def __getitem__(self, key):
        return self.get(key).value

    def __contains__(self, key):
        try:
            if self.get(key):
                return True
        except KeyError:
            return False

    def delete(self, key):
        if self.size >= 1:
            removenode = self.get(key)
            if removenode:
                self._remove(removenode)
                self.size -= 1

    def __delitem__(self, key):
        self.delete(key)

    def _getsuccessor(self, node):
        if node.lchild:
            self._getsuccessor(node.lchild)
        elif node.rchild:
            if node.is_lchild():
                node.parent.lchild = node.rchild
            else:
                node.parent.rchild = node.rchild
            node.rchild.parent = node.parent
            return node
        else:
            return node

    def _remove(self, node):
        if node.is_leaf():
            if node.is_root():
                self.root = None
            elif node.is_lchild():
                node.parent.lchild = None
            elif node.is_rchild():
                node.parent.rchild = None
        elif node.has_both_children():
            successor = self._getsuccessor(node.rchild)
            node.key = successor.key
            node.value = successor.value
        else:
            if node.lchild:
                child = node.lchild
            else:
                child = node.rchild
            if node.is_lchild():
                node.parent.lchild = child
            else:
                node.parent.rchild = child
