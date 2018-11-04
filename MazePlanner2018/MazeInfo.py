"""  =================================================================
File: MazeInfo.py

This file contains a class to represent the kind of maze being solved.  
It really just keeps track of the maze information, and checks if a given
cell is clear or filled.
 ==================================================================="""

import random
from FoxQueue import Queue


class MazeInfo:
    """Represents a square grid maze.  You can set it up and then ask about
    which cells are open or filled"""


    def __init__(self, mode, reqInput, numCols = None, startPos = (-1, -1), goalPos = (-1, -1), percBlocked = 0.0):
        """Two inputs are required, all others are optional. First required input is mode, which tells how to make the
        MazeInfo. If reading from a file, then the second required input is the filename. If generating, then the second required
        input is the number of rows. If copying, then the second required is another MazeInfo object.
        Inputs:
        * mode tells how to make the MazeInfo. Values include 'file' to read from a file, 'gen-flat' or 'gen-hilly'
        to generate flat or hilly terrain, 'copy' to copy.
        * reqInput is the minimum required input, either filename, number of rows, or MazeInfo object
        * numCols is the number of columns. If no numCols is given, then maze is made square
        * startPos is the starting position, if known. If not, then (-1, -1) is written to file
        * goalPos is the goal position, if known. If not given, then (-1, -1) is written to file
        * percBlocked is the percentage of randomly-scattered locations that are blocked off and cannot be visited
        """
        if type(mode) != str:
            raise ValueError("First input must be a string, one of 'file', 'gen-flat', 'gen-hilly', or 'copy")
        elif mode == 'file':  # ignores all other inputs
            self._readMaze(reqInput)
        elif mode == 'copy':
            print("Copying not implemented yet")
        elif mode in {'gen-hilly', 'gen-flat'}:
            self.numRows = reqInput
            if numCols == None:
                self.numCols = self.numRows
            else:
                self.numCols = numCols

            self.startPos = startPos
            self.goalPos = goalPos
            self.percBlocked = percBlocked
            self.blockedLocs = set()
            self.weightMatrix = {}

            if mode == 'gen-hilly':
                self.generateHillyLandscape()
            else:
                self.generateFlatLandscape()


    def generateFlatLandscape(self):
        """Generates a flat landscape of the specified size. All cells have weight = 1"""
        self.minCost = 1
        self.maxCost = 50
        self._initializeBlocks()

        for r in range(self.numRows):
            for c in range(self.numCols):
                self.weightMatrix[r, c] = 1



    def generateHillyLandscape(self):
        """Generates a hilly landscape with a number of high points based on the number of cells (1/20th of the
         world will be high points. Then the neighbors are random yet gradually falling toward flat."""
        self.minCost = 1
        self.maxCost = 50

        self._initializeBlocks()
        self._initializeMaxPoints()
        # self._printMaze()
        self._fillGrid()
        # self._printMaze()


    def _initializeBlocks(self):
        """This initializes generates random locations for blocked cells, ones the agent cannot enter. The number of
        cells is computed based on the self.percBlocked instance variable, and the values at those locations are set to -1."""
        numPts = self._percentageOfGrid(self.percBlocked)
        pointList = self._generateRandomPoints(numPts)
        self.blockedLocs = set(pointList)


    def _initializeMaxPoints(self):
        """Given the number of max points desired, this generates
        a unique random position for each max point (i.e., it won't
        accidentally place two max points at the same location). It
        randomly generates the max value to be within 80% of the given maximum cost."""
        numPts = self._percentageOfGrid(0.05)         # 5% of the grid will be peaks
        pointList = self._generateRandomPoints(numPts)
        for (row, col) in pointList:
            eightyPerc = 4 * self.maxCost / 5
            maxPtValue = random.randrange(eightyPerc, (self.maxCost + 1))
            self.weightMatrix[row, col] = maxPtValue


    def _percentageOfGrid(self, percent):
        """Takes in the percentage of the grid that we want to select for "special" treatment, and it
        computes and returns the number of special points there should be (based on the grid size).
        Give the percentage as a number between 0 and 1."""
        numSpecialPoints = int((self.numRows * self.numCols) * percent)
        return numSpecialPoints


    def _generateRandomPoints(self, numPts):
        """Given a number of points, randomly generates them at unused locations in the grid, and returns a list of
        (row, col) tuples. """
        chosenOnes = []
        for i in range(numPts):
            row = -1
            col = -1
            while True:
                row = random.randrange(self.numRows)
                col = random.randrange(self.numCols)
                if (row, col) not in self.weightMatrix:
                    break
            chosenOnes.append( (row, col) )
        return chosenOnes


    def _fillGrid(self):
        """Uses a flood-fill method to fill in the other values somewhat
        randomly. Each max point is a starting point, and it spreads
        outward from each, to all 8 neighboring cells. The value at each
        cell is some random change (almost always less) from the average of its current filled-in neighbors."""
        self.queue = Queue()
        self.seenPos = set()
        for pos in self.weightMatrix:
            (row, col) = pos
            neighs = self._generateNeighbors(row, col)
            self._addNeighsToQueue(neighs)

        while not self.queue.isEmpty():
            nextPos = self.queue.firstElement()
            self.queue.delete()
            if nextPos not in self.weightMatrix:
                (r, c) = nextPos
                nextNeighs = self._generateNeighbors(r, c)
                val = self._computeNextValue(nextNeighs)
                self.weightMatrix[r, c] = val
                self._addNeighsToQueue(nextNeighs)
                # self.printGrid()


    def _addNeighsToQueue(self, neighs):
        """This adds neighbors that have no current value and
        that have not been seen yet to the queue. It ensures that
        no value is overwritten, and that each cell is in the queue
        only once."""
        for n in neighs:
            if (n in self.weightMatrix) or (n in self.seenPos):
                pass
            else:
                self.seenPos.add(n)
                self.queue.insert(n)


    def _computeNextValue(self, neighList):
        """This takes a list of neighbors and computes the value for
        the center cell. It averages the filled-in neighbors' values.
        Then it computes the minimum delta values, which is always negative (we want weights to tend to decrease
        towards minimum). The minimum delta magnitude is 40% of the maximum cost possible.
        Then the actual delta is generated randomly between the minimum and 1 (so there is a small chance that the
        weight will stay the same or even increase by 1). It is then bounded to be between the minimum cost and
        maximum cost specified for this map, and the cell is set."""
        vals = []
        for (r, c) in neighList:
            if (r, c) in self.weightMatrix:
                vals.append(self.weightMatrix[r, c])
        avgVal = sum(vals) / len(vals)
        minDelta = -2 * self.maxCost / 5
        delta = random.randint(minDelta, 1)
        value = int(avgVal + delta)
        value = min(value, self.maxCost)
        value = max(value, self.minCost)
        return value


    def _generateNeighbors(self, row, col):
        """Takes a row and column and generates all the valid
        rows and columns for the eight possible neighbors. Returns
        a list of those valid positions."""
        neighs = []
        for r in [row - 1, row, row + 1]:
            for c in [col - 1, col, col + 1]:
                if (r == row) and (c == col):
                    pass
                elif (r < 0) or (r >= self.numRows) or \
                        (c < 0) or (c >= self.numCols):
                    pass
                else:
                    neighs.append((r, c))
        return neighs


    def isAccessible(self, row, col):
        """Given a row and column coordinate,r eturns True if the given cell is neither
        blocked nor out of bounds, and False otherwise"""
        return (not self.isBlocked(row, col)) and (not self.isOutOfBounds(row, col))


    def isBlocked(self, row, col):
        """Returns True if the given cell is blocked (the agent cannot go there), and False otherwise."""
        return (row, col) in self.blockedLocs

    def isOutOfBounds(self, row, col):
        return (row < 0) or (col < 0) or (row >= self.numRows) or (col >= self.numCols)


    def getNumRows(self):
        """Returns the number of rows"""
        return self.numRows


    def getNumCols(self):
        """Returns the number of rows"""
        return self.numCols

    def getMaxWeight(self):
        """Returns the maximum weight in the maze"""
        return self.maxCost

    def getMinWeight(self):
        """Returns the minimum weight in the maze"""
        return self.minCost


    def getStartPos(self):
        """Returns the current starting position"""
        return self.startPos


    def getGoalPos(self):
        """Returns the current goal position"""
        return self.goalPos


    def setStartPos(self, newPos):
        """Takes in a new position, checks that it is valid, and then sets the startPos to that value if it is."""
        (row, col) = newPos
        if self.isOutOfBounds(row, col) or self.isBlocked(row, col):
            return
        else:
            self.startPos = newPos

    def setGoalPos(self, newPos):
        """Takes in a new position, checks that it is valid, and then sets the startPos to that value if it is."""
        (row, col) = newPos
        if self.isOutOfBounds(row, col) or self.isBlocked(row, col):
            return
        else:
            self.goalPos = newPos


    def getWeight(self, row, col):
        """Given a row and column, look up the terrain value for that position."""
        if self.isOutOfBounds(row, col) or self.isBlocked(row, col):
            return -1
        else:
            return self.weightMatrix[row, col]


    def setWeight(self, row, col, newVal):
        """Takes in the row and column and a new weight value, and it updates the weight."""
        if self.isOutOfBounds(row, col) or self.isBlocked(row, col):
            return
        else:

            if newVal > self.maxCost:
                self.weightMatrix[row, col] = self.maxCost
            elif newVal < self.minCost:
                self.weightMatrix[row, col] = self.minCost
            else:
                self.weightMatrix[row, col] = newVal


    def increaseWeight(self, row, col):
        """Increases the weight at (row, col) by one."""
        if self.isOutOfBounds(row, col) or self.isBlocked(row, col):
            return
        else:
            newVal = self.weightMatrix[row, col] + 1
            self.setWeight(row, col, newVal)


    def decreaseWeight(self, row, col):
        """Decreases the weight at (row, col) by one."""
        if self.isOutOfBounds(row, col) or self.isBlocked(row, col):
            return
        else:
            newVal = self.weightMatrix[row, col] - 1
            self.setWeight(row, col, newVal)


    def addBlocked(self, row, col):
        """Adds (row, col) to the blocked set"""
        self.blockedLocs.add( (row, col) )


    def delBlocked(self, row, col):
        """Removes (row, col) from the blocked set."""
        try:
            self.blockedLocs.remove( (row, col) )
        except:
            pass


    def writeGridToFile(self, gridFile):
        """Takes a filename and writes the grid qData to the file."""
        try:
            filObj = open(gridFile, 'w')
        except:
            raise FileExistsError("ERROR OPENING FILE, ABORTING")

        filObj.write("# Gridmap generated by GridMapGenerator\n")
        filObj.write("# height width:\n")
        filObj.write(str(self.numRows) + " " + str(self.numCols) + '\n')
        filObj.write("# minCost maxCost\n")
        filObj.write(str(self.minCost) + ' ' + str(self.maxCost) + '\n')
        filObj.write("# starting position:\n")
        filObj.write(str(self.startPos[0]) + ' ' + str(self.startPos[1]) + '\n')
        filObj.write("# goal position:\n")
        filObj.write(str(self.goalPos[0]) + ' ' + str(self.goalPos[1]) + '\n')
        filObj.write("# Blocked cells list\n")
        filObj.write("[\n")
        for (r, c) in self.blockedLocs:
            filObj.write(str(r) + ' ' + str(c) + '\n')
        filObj.write(']\n')
        filObj.write("# Map: \n")
        for r in range(self.numRows):
            for c in range(self.numCols):
                val = self.weightMatrix[r, c]
                filObj.write(str(val) + " ")
            filObj.write('\n')
        filObj.close()


    def _readMaze(self, mapFile):
        """Takes in a filename for a grid-map file, and it reads in the qData from the file.
        It creates a grid representation using a dictionary, where the key is the (row, col) of each
         grid cell, and the value is the weight at the cell."""
        try:
            filObj = open(mapFile, 'r')
        except:
            raise FileExistsError("ERROR READING FILE, ABORTING")
        seeking = 'grid size'
        self.weightMatrix = {}
        self.blockedLocs = set()
        self.minCost = None
        self.maxCost = None
        row = 0
        for line in filObj:
            if line == "" or line.isspace() or line[0] == '#':
                continue
            elif seeking == 'grid size':   # Haven't seen first line, so it must be grid size
                [hgt, wid] = [int(s) for s in line.split()]
                self.numRows = wid
                self.numCols = hgt
                seeking = 'minmax'
            elif seeking == 'minmax':
                [minc, maxc] = [int(s) for s in line.split()]
                self.minCost = minc
                self.maxCost = maxc
                seeking = 'start'
            elif seeking == 'start':  # Have seen first line, so next must be start pos
                sPos = [int(s) for s in line.split()]
                self.startPos = sPos
                seeking = 'goal'
            elif seeking == 'goal': # Have seen first two, next must be goal pos
                gPos = [int(s) for s in line.split()]
                self.goalPos = gPos
                seeking = 'blocked'
            elif seeking == 'blocked': # Have seen all but blocked cells...
                if line[0] == '[':
                    # starting line, just keep going
                    pass
                elif line[0] == ']':
                    # final line, go on to the next
                    seeking = 'gridcells'
                else:
                    [blockRow, blockCol] = [int(s) for s in line.split()]
                    self.blockedLocs.add( (blockRow, blockCol) )
            elif seeking == 'gridcells':
                cellWeights = [int(s) for s in line.split()]
                for col in range(wid):
                    self.weightMatrix[row, col] = cellWeights[col]
                row += 1
            else:
                print("Uh-oh, should never get here")
        filObj.close()
        # self._printMaze()


    def _printMaze(self):
        """Helper to print the grid representation, mostly for debugging."""
        print("Size:", self.numRows, self.numCols)
        print("Starting position:", self.startPos)
        print("Goal position:", self.goalPos)
        for row in range(self.numRows):
            rowStr = ""
            for col in range(self.numCols):
                if (row, col) in self.weightMatrix:
                    val = self.weightMatrix[row, col]
                    valStr = str(val).rjust(3)
                    rowStr += valStr + " "
                else:
                    rowStr += "    "
            print(rowStr)
