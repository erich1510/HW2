"""####################################################
A Queue class
Susan Fox
Spring 2007
Updated Spring 2014 to fix comment style
Updated Spring 2016 to add methods to priority queue for removing qData from the queue.
Updated Fall 2018 to make classes consistent, to switch the ordering of inputs to the insert methods,
and to add a contains method. Now the classes assume that the object being stored implements the == operator
in a meaningful way.
"""


class Queue:
    """A queue is a linear collection used to hold qData that is waiting
    for some purpose.  The first to enter the queue is the first to
    leave it."""

    def __init__(self, valList=None):
        """When creating a new queue, you can give a list of values to
        insert in the queue at the start."""
        if valList is None:
            self.qData = []
        else:
            self.qData = valList[:]
        self.size = len(self.qData)

    def getSize(self):
        """Return the size of the queue."""
        return self.size

    def isEmpty(self):
        """Returns true if the queue is empty, or false otherwise."""
        return self.size == 0

    def firstElement(self):
        """Returns the first value in the queue, without removing it."""
        if self.isEmpty():
            return None
        else:
            return self.qData[0]

    def peek(self):
        """Another name for firstElement."""
        return self.firstElement()


    def insert(self, val):
        """Inserts a new value at the end of the queue."""
        self.qData.append(val)
        self.size = self.size + 1

    def enqueue(self, val):
        """Another name for inserting."""
        self.insert(val)

    def delete(self):
        """Removes the first element from the queue, returning it as its value."""
        if self.isEmpty():
            return None
        else:
            firstData = self.qData.pop(0)
            self.size = self.size - 1
            return firstData

    def dequeue(self):
        """Another name for deleting: removes the first element from the queue, returning it as its value."""
        return self.delete()

    def contains(self, value):
        """Takes in a value and returns true if the item is in the queue. Assumes the item implements
        the __eq__ operation."""
        for item in self.qData:
            if value == item:
                return item
        return False


    def __str__(self):
        """Creates a string containing the qData, just for debugging."""
        qstr = "Queue: <- "
        if self.size <= 3:
            for val in self.qData:
                qstr = qstr + str(val) + " "
        else:
            for i in range(3):
                qstr = qstr + str(self.qData[i]) + " "
            qstr = qstr + "..."
        qstr = qstr + "<-"
        return qstr
# end class Queue



class PriorityQueue(Queue):
    """A priority queue puts lowest-cost elements first.
    Implemented with a MinHeap, which is internal to the class"""

    def __init__(self, valList=None):
        """Has two optional inputs. The first is a list to populate the queue with, which must
        be a list of tuples, where each tuple contains a value and that value's priority. The second
        optional input is a function used to compare elements of the priority queue to determine which has
        higher priority. This allows more complex priorities, including lists of priority values to be handled.
        """
        Queue.__init__(self)
        self.qData = []
        self.size = 0
        if valList is not None:
            for (val, prior) in valList:
                self.insert(val, prior)


    def insert(self, value, priority):
        """Inserts a new value at the end of the queue."""
        self.qData.append((value, priority))
        self.size = self.size + 1
        self._walkUp(self.size - 1)

    def enqueue(self, val, priority):
        """Another name for inserting"""
        self.insert(val, priority)

    def _walkUp(self, index):
        """Walk a value up the heap until it is larger than its parent
        This is really a *private* method, no one outside should call it.
        Thus the underscore leading the name."""
        inPlace = False
        while not(index == 0) and not inPlace:
            parentIndex = self._parent(index)
            curr = self.qData[index]
            par = self.qData[parentIndex]
            if curr[1] >= par[1]:
                inPlace = True
            else:
                self.qData[index] = par
                self.qData[parentIndex] = curr
                index = parentIndex

    def delete(self):
        """Removes the first element from the queue, returning it as its value, or returning None if
        the queue is already empty."""
        if self.size == 0:
            return None
        elif self.size == 1:
            poppedElement = self.qData[0]
            self.size = self.size - 1
            self.qData = []
            return poppedElement
        else:
            poppedElement = self.qData[0]
            self.size = self.size - 1
            lastItem = self.qData.pop(self.size)
            self.qData[0] = lastItem
            self._walkDown(0)
            return poppedElement

    def dequeue(self):
        """Another name for deleting, removes the first element from the queue, returning it as its value"""
        return self.delete()

    def _walkDown(self, index):
        """A private method, walks a value down the tree until it is smaller than both its children."""
        inPlace = False
        leftInd = self._leftChild(index)
        rightInd = self._rightChild(index)
        while not(leftInd >= self.size) and not inPlace:
            curr = self.qData[index]

            if (rightInd >= self.size):
                minInd = leftInd
            else:
                leftVal = self.qData[leftInd]
                rightVal = self.qData[rightInd]
                if leftVal[1] <= rightVal[1]:
                    minInd = leftInd
                else:
                    minInd = rightInd

            minVal = self.qData[minInd]
            if curr[1] <= minVal[1]:
                inPlace = True
            else:
                self.qData[minInd] = curr
                self.qData[index] = minVal
                index = minInd
                leftInd = self._leftChild(index)
                rightInd = self._rightChild(index)


    def update(self, value, newP):
        """Update finds the given value in the queue, changes its priority value, and then moves it 
        up or down the tree as appropriate."""
        pos = self._findValue(value)
        [oldP, v] = self.qData[pos]
        self.qData[pos] = [newP, value]
        if oldP > newP:
            self._walkUp(pos)
        else:
            self._walkDown(pos)


    def contains(self, value):
        """Takes in a value and searches for it in the priority queue. If it is there, it returns True, 
        otherwise False."""
        pos = self._findValue(value)
        if pos < 0:
            # value not found
            return False
        else:
            return self.qData[pos][0]


    def removeValue(self, value):
        """Takes in a value and searches for it, and then removes it from the queue, wherever it is."""
        pos = self._findValue(value)
        if pos < 0:
            # If value not found
            print("Value not found:", value)
        elif self.size == 1:
            # If only one value left, make heap empty
            self.size = self.size - 1
            self.qData = []
        elif pos == (self.size - 1):
            # if removed value is last one, just remove it
            self.size = self.size - 1
            self.qData.pop(self.size)
        else:
            self.size = self.size - 1
            lastItem = self.qData.pop(self.size)
            self.qData[pos] = lastItem
            self._walkDown(pos)


    def _findValue(self, value):
        """Find the position of a value in the priority queue."""
        i = 0
        for [val, prior] in self.qData:
            if val == value:
                return i
            i = i + 1
        return -1



    # The following helpers allow us to figure out
    # which value is the parent of a given value, and which
    # is the right child or left child.

    def _parent(self, index):
        """Private: find position of parent given position of heap node."""
        return (index - 1) // 2

    def _leftChild(self, index):
        """Private method: find position of left child given position of heap node."""
        return (index * 2) + 1

    def _rightChild(self, index):
        """Private method: find position of right child given position of heap node."""
        return (index + 1) * 2

    def __str__(self):
        """Provides a string with just the first element."""
        val = "PQueue: "
        if self.isEmpty():
            val += "<empty>"
        else:
            p, v = self.firstElement()
            val = val + "priority: " + str(p) + ", value: " + str(v)
        return val


