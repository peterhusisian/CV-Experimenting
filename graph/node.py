

class Node:

    def __init__(self, data, next):
        self.data = data
        self.next = next

    def has_next(self):
        return self.next is not None
