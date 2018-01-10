from graph.node import Node

class Queue:
    '''
    def __init__(self):
        self.header = None

    def push(self, data):
        self.header = Node(data, self.header)


    def pop(self):
        if self.header is not None:
            old_header = self.header
            self.header = old_header.next
            return old_header.data
        else:
            raise ValueError("Queue has no data left to pop!")

    def is_empty(self):
        return self.header is None
    '''
    def __init__(self):
        self.items = []

    def push(self, data):
        self.items.append(data)

    def pop(self):
        return self.items.pop(len(self.items)-1)

    def is_empty(self):
        return len(self.items) == 0
