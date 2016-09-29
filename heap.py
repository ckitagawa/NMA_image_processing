from math import floor, log

MIN_HEAP = True
MAX_HEAP = False

class Heap(object):
    def __init__(self, heap_type=MAX_HEAP):
        """
        Initializes a pythonic implementation of a min/max heap.
        
        Keyword Arguements:
        heap_type -- the type of heap default is MAX_HEAP
        """
        self.arr = []
        self.h_type = heap_type

    def __len__(self):
        return len(self.arr)

    def __str__(self):
        """Output the heap in a human readable format"""
        tree_rep = ""
        levels = round(log(len(self.arr), 2)) + 1
        last = 0

        #TODO: add spacing to appear as a tree
        for i in range(levels):
            elem_in_level = 2 ** i
            space = "\t"
            tree_rep += space.join(map(str, self.arr[last:last+elem_in_level]))
            last += elem_in_level
            if last < len(self):
                tree_rep += "\n"
        return tree_rep

    def is_empty(self):
        if not self.arr:
            return True
        return False

    def _parent(self, i):
        if floor((i - 1) / 2) >= 0:
            return floor((i - 1) / 2)
        return None

    def _left(self, i):
        if 2 * i + 1 < len(self):
            return 2 * i + 1
        return None

    def _right(self, i):
        if 2 * i + 2 < len(self):
            return 2 * i + 2
        return None

    def peek(self):
        if not self.is_empty():
            return self.arr[0]
        return None

    def sift_up(self, i):
        p = self._parent(i)
        if p or p == 0:
            if self.h_type == MIN_HEAP:
                if self.arr[p] > self.arr[i]:
                    self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
                    self.sift_up(p)
            else:
                if self.arr[p] < self.arr[i]:
                    self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
                    self.sift_up(p)

    def sift_down(self, i):
        l = self._left(i)
        r = self._right(i)

        if self.h_type == MIN_HEAP:
            if l and self.arr[l] < self.arr[i]:
                m = l
            else:
                m = i
            if r and self.arr[r] < self.arr[m]:
                m = r
        else:
            if l and self.arr[l] > self.arr[i]:
                m = l
            else:
                m = i
            if r and self.arr[r] > self.arr[m]:
                m = r
        if m != i:
            self.arr[i], self.arr[m] = self.arr[m], self.arr[i]
            self.sift_down(m)

    def create_heap(self, array):
        assert type(array) == list
        self.arr = array
        for i in range(floor(len(self) / 2), -1, -1):
            self.sift_down(i)

    def push(self, value):
        self.arr.append(value)
        self.sift_up(len(self) - 1)
        return None

    def delete(self, i):
        if i < len(self):
            self.arr[i] = self.arr[-1]
            del self.arr[-1]
            self.sift_down(i)
            return True
        return False

    def delete_head(self):
        if not self.is_empty():
            self.delete(0)
            return True
        return False

    def replace(self, value):
        if not self.is_empty():
            prev_head = self.arr[0]
            self.arr[0] = value
            self.sift_down(0)
            return prev_head
        return None

    def pop(self):
        if not self.is_empty():
            head = self.arr[0]
            self.delete_head()
            return head
        return None

if __name__ == "__main__":
    h = Heap(heap_type=MIN_HEAP)
    A = [5,2,3,4,23,45,34,2,5,23,4,1,6,7,33,9,0,21]
    h.create_heap(A)
#    h.push(1)
#    print(h.arr)
#    h.push(3)
#    print(h.arr)
#    h.push(2)
#    print(h.arr)
#    print(h.peek())
    print(h)
    h.delete(1)
    print(h)
    h.replace(90)
    print(h)
