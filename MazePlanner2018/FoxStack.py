"""####################################################
A Stack class
Susan Fox
Spring 2007
Revised Spring 2014 to update style.
Revised Fall 2018 to match changes to Queue and PriorityQueue classes, adding a "contains" method.
"""

class Stack:
    """A stack is a linear collection used to hold qData that is waiting
    for some purpose.  Values are added at one end and removed from the
    same end, like a stack of plates"""

    def __init__(self, vallist=[]):
        """When creating a new stack, you can give a list of values to
        insert in the stack at the start.  The front of the list becomes
        the top of the stack."""
        self.data = vallist[:]
        self.size = len(self.data)

    def getSize(self):
        """Returns the size of the stack"""
        return self.size


    def isEmpty(self):
        """Returns true if the stack is empty, or false otherwise."""
        return self.size == 0


    def firstElement(self):
        """Another name for top"""
        return self.top()

    def peek(self):
        """Same as firstElement or top"""
        return self.top()

    def top(self):
        """Returns the first value in the stack, without removing it."""
        if self.isEmpty():
            return None
        else:
            return self.data[0]


    def insert(self, val):
        """Inserts a new value at the end of the stack."""
        self.data.insert(0,val)
        self.size = self.size + 1 

    def push(self, val):
        """Another name for inserting"""
        self.insert(val)


    def delete(self):
        """Removes the first element from the stack, returning its value."""
        first = self.data[0]
        self.data.pop(0)
        self.size = self.size - 1
        return first

    def pop(self):
        """Another name for deleting"""
        return self.delete()


    def contains(self, value):
        """Given a value, it searches the stack for a matching value, returning it if found, or returning False if not found"""
        for item in self.data:
            if item == value:
                return item
        return False


    def __str__(self):
        """Creates a string containing the qData, just for debugging."""
        stackStr = "Stack: <- "
        if self.size <= 3:
            for val in self.data:
                stackStr = stackStr + str(val) + " "
        else:
            for i in range(3):
                stackStr = stackStr + str(self.data[i]) + " "
            stackStr = stackStr + "..."
        stackStr = stackStr + "]"
        return stackStr
# end class Stack

