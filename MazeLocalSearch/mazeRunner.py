
import random
from MazeInfo import MazeInfo
from localSearch import hillClimb, stochHillClimb, simAnnealing, beamSearch, geneticAlg


class MazeAgent(object):
    """An agent/state that encodes a sequence of moves for solving a specific maze. The sequence may have
    varying length, and is made up of the moves: n (north), s (south), e (east), w (west).

    Neighbors of this agent/state include any sequence that has one step changed, or a step added or removed from either end.
     This means a LOT of neighbors, making full hill-climbing difficult."""


    def __init__(self, maze, plan=None, full=True):
        """
        Sets up an agent for the maze with a planned movement sequence, an initial position and heading,
        a maze that it works on, and an option to print multiline or single-line descriptions of itself
        :param plan: string describing the sequence of moves of the agent
        :param maze: a MazeInfo object that encodes the maze that the agent runs
        :param full: a boolean for whether to print multiline or single-line
        """
        self.fullPrint = full
        self.sim = None
        self.value = -1.0

        self.row = 0
        self.col = 0

        self.maze = maze
        rows = self.maze.getNumRows()
        cols = self.maze.getNumCols()
        self.maxDist = rows + cols - 2  # max distance away from goal the agent can be
        self.maxSteps = rows * cols  # overestimate of maximum number of steps possible
        (sRow, sCol) = self.maze.getStartPos()
        (gRow, gCol) = self.maze.getGoalPos()
        self.minDist = abs(gRow - sRow) + abs(gCol - sCol)

        if plan is None:
            self.plan = self._randomPlan()
            self.planStep = 0
            self.planLength = len(self.plan)
        else:
            self.plan = plan
            self.planStep = 0
            self.planLength = len(self.plan)


        self.value = self.computeValue()


    def computeValue(self):
        """
        Runs this agent on the current maze. It gets the distance to goal and converts those into a value.
        The value has two components. First of all the smaller the distance to goal, the better. Secondly,
        we want to weigh shorter paths over longer paths, so we take the absolute longest a path could be in a given
        maze and subtract from that the difference between the current plan length and the heuristic city-block
        distance. We scale the distance to the goal so that it is more important than the number of steps.
        :return The value of this plan
        """
        self.sim = MazeRunnerSim(self.maze)
        dist, pathCost = self.sim.runSimulation(self)
        # print("Distance to goal =", dist)
        # flipDist = self.maxDist - dist   # Flip so smaller distances become larger values
        # flipSteps = self.maxSteps - abs(self.minDist - self.planLength)  # Flip here as well
        # score = 100 * flipDist + flipSteps
        score = -(100 * dist + pathCost)
        return score



    def _randomPlan(self):
        """Builds a random set of behaviors, balancing it so that moving and turning are equally likely."""
        options = 'nsew'
        planStr = ""
        if self.maxDist > 0:
            planLen = random.randint(int(0.1 * self.maxDist), self.maxDist)
        else:
            planLen = random.randint(2, 30)
        for i in range(planLen):
            planStr += random.choice(options)
        return planStr


    def getPos(self):
        """Return the row, and column indices of the agent."""
        return self.row, self.col

    def updatePos(self, row, col):
        """Updates the agent's pose to a new position and heading"""
        self.row = row
        self.col = col


    def getNextAction(self):
        """
        This produces the next action in the plan, and updates the counter for which step of the plan we're on.
        The plan steps may be 'forward', 'left', 'right', 'about-fact', or 'done', if the plan is complete.
        :return: Returns a string that describes the next action, if any.
        """
        if self.planStep >= self.planLength:  # Plan is done
            return 'done'
        action = self.plan[self.planStep]
        self.planStep += 1

        if action == 'n':
            return 'north'
        elif action == 'e':
            return 'east'
        elif action == 'w':
            return 'west'
        elif action == 's':
            return 'south'
        else:
            print("getNextAction: SHOULD NEVER GET HERE", action)
            return 'east'


    def copyState(self):
        """Makes a copy of itself.."""
        pose = (self.row, self.col)
        return MazeAgent(self.maze, plan=self.plan, full=self.fullPrint)


    def __eq__(self, other):
        """Compares two states, if they have the same maze and
        ruleset then they are equal."""
        if type(other) != type(self):
            return False
        return self.plan == other.plan and self.maze == other.maze


    def __str__(self):
        """Produces a printable string version of the object."""
        if self.fullPrint:
            formStr = """Agent:
            Pose: {2:>3d}  {3:>3d}
            Value: {4:6.2f}
            Plan: {0:s}  at step {1:>3d}"""
        else:
            formStr = "Agent: pose=({2:>3d}, {3:>3d}) value={4:6.2f}  plan={0:s}[{1:>3d}]  "
        return formStr.format(self.plan, self.planStep, self.row, self.col, self.value)

    def randomState(self):
        """Generates another random agent with the same maze attached."""
        return MazeAgent(self.maze)

    def setPrintMode(self, full):
        """Changes the print mode: full = True means print multiline representation.
        full = False means print one-line representation."""
        self.fullPrint = full

    def getValue(self):
        """Returns score associated with this agent's performance on the current maze"""
        return self.value

    def getMaxValue(self):
        """Returns the maximum possible value."""
        return 0
        # return 100 * self.maxDist + self.maxSteps


    def allNeighbors(self):
        """Generates all neighbors of this state. For the plan, that means
        all one-symbol changes, plus deleting first or last, plus adding a new symbol to front or back."""
        neighbors = []
        # Add neighbors that are one-symbol changes to each position in plan
        for i in range(self.planLength):
            currSym = self.plan[i]
            otherSyms = self._otherSymbols(currSym)
            for c in otherSyms:
                newPlan = self.plan[:i] + c + self.plan[i+1:]
                newState = MazeAgent(self.maze, plan=newPlan, full=self.fullPrint)
                neighbors.append(newState)
        # Next add plans that delete first or last, if plan is at least 2 long
        if self.planLength > 1:
            delFirst = self.plan[1:]
            delLast = self.plan[:-1]
            neighbors.append(MazeAgent(self.maze, plan=delFirst, full=self.fullPrint))
            neighbors.append(MazeAgent(self.maze, plan=delLast, full=self.fullPrint))
        # Next consider adding new symbol to front and back of plan
        for sym in "news":
            addFirst = sym + self.plan
            addLast = self.plan + sym
            neighbors.append(MazeAgent(self.maze, plan=addFirst, full=self.fullPrint))
            neighbors.append(MazeAgent(self.maze, plan=addLast, full=self.fullPrint))
        return neighbors

    def _otherSymbols(self, sym):
        """Given a symbol, return a string of the other symbols besides it."""
        if sym == 'n':
            return 'ews'
        elif sym == 'e':
            return 'nws'
        elif sym == 'w':
            return 'nes'
        elif sym == 's':
            return 'new'
        else:
            print("_otherSymbols: should never get here!")

    def randomNeighbors(self, num):
        """Generate num random neighbors of this state. Note that the same neighbor could be generated more than once."""
        neighbors = []
        for i in range(num):
            newS = self.makeRandomMove()
            neighbors.append(newS)
        return neighbors

    def makeRandomMove(self):
        """Takes a ruleset and returns a new ruleset identical to the original, but with one random change."""
        randElem = random.randrange(self.planLength + 4)
        if randElem < self.planLength:  # change is to modify a symbol in the current plan
            opts = self._otherSymbols(self.plan[randElem])
            newElem = random.choice(opts)
            newPlan = self.plan[:randElem] + newElem + self.plan[randElem+1:]
        else:
            randCode = randElem - self.planLength
            if randCode == 0:  # delete first symbol in plan
                newPlan = self.plan[1:]
            elif randCode == 1: # delete last symbol in plan
                newPlan = self.plan[:-1]
            elif randCode == 2:  # Add something to front of plan
                newPlan = random.choice("news") + self.plan
            elif randCode == 3:
                newPlan = self.plan + random.choice("news")
            else: 
                print("makeRandomMove: Should never get here")
        return MazeAgent(self.maze, plan=newPlan, full=self.fullPrint)


    def crossover(self, other):
        """Cross over this with another state. Since the states' plans may be varying lengths, this picks a random point
        in each plan and splits the plans, attaching pieces together."""

        myCrossPoint = random.randint(0, self.planLength)
        otherCrossPoint = random.randint(0, other.planLength)
        newPlan1 = self.plan[:myCrossPoint] + other.plan[otherCrossPoint:]
        newPlan2 = other.plan[:otherCrossPoint] + self.plan[myCrossPoint:]
        new1 = MazeAgent(self.maze, plan=newPlan1, full=self.fullPrint)
        new2 = MazeAgent(self.maze, plan=newPlan2, full=self.fullPrint)
        return new1, new2



class MazeRunnerSim(object):
    """Given a maze object, this simulates an agent moving about in the
    world, starting at the start location and continuing until the max
    number of steps is reached, or until the goal is reached."""

    def __init__(self, maze):
        """
        Sets up a simulation of an agent or series of agents in a given maze.
        Move the agent to the starting location given in the maze object
        :param maze: A MazeInfo object, holds the form of the maze
        :param agent: The MazeAgent object that is going to run the maze
        """
        self.maze = maze


    def runSimulation(self, agent):
        """
        Simulates the agent running in the maze; simulation continues through the agent's plan.
        Once plan execution is done, this reports back the distance of the agent from the goal.
        :param agent: The agent object to be simulated
        :return: distToGoal of agent
        """
        sRow, sCol = maze1.getStartPos()
        gRow, gCol = maze1.getGoalPos()
        agent.updatePos(sRow, sCol)
        pathCost = maze1.getWeight(sRow, sCol)
        # print(agent)
        while True:
            simContinue, cost = self.step(agent)
            # print(agent)
            if not simContinue:
                break
            pathCost += cost
        (newR, newC) = agent.getPos()
        distToGoal = abs(newR - gRow) + abs(newC - gCol)
        return distToGoal, pathCost


    def step(self, agent):
        """
        Update one step of the simulation. This means determining the
        agent's surroundings, asking the agent for an action, and performing that action.
        """
        (row, col) = agent.getPos()
        action = agent.getNextAction()
        if action == 'done':
            # print("Simulation done!")
            return False, 0
        elif action in ['north', 'south', 'east', 'west']:
            rAhead, cAhead = self._computeAhead(row, col, action)
            if self.maze.isAccessible(rAhead, cAhead):
                agent.updatePos(rAhead, cAhead)
                row, col = rAhead, cAhead
        else:
            print("step: Unknown action:", action)
        return True, self.maze.getWeight(row, col)


    def _computeAhead(self, row, col, heading):
        """Determine the cell that is one space ahead of current cell, given the heading."""
        if heading == 'north':   # agent is pointing north, row value decreases
            newR = (row - 1)
            return newR, col
        elif heading == 'south':  # agent is pointing south, row value increases
            newR = (row + 1)
            return newR, col
        elif heading == 'west':  # agent is pointing west, col value decreases
            newC = (col - 1)
            return row, newC
        else:  # agent is pointing east, col value increases
            newC = (col + 1)
            return row, newC

    def _leftTurn(self, heading):
        """return the new heading for a left turn"""
        if heading == 'n':
            return 'w'
        elif heading == 'w':
            return 's'
        elif heading == 's':
            return 'e'
        else:
            return 'n'

    def _rightTurn(self, heading):
        """return the new heading for a right turn"""
        if heading == 'n':
            return 'e'
        elif heading == 'e':
            return 's'
        elif heading == 's':
            return 'w'
        else:
            return 'n'

    def _turnAround(self, heading):
        """return the new heading for a full 180-degree turn"""
        if heading == 'n':
            return 's'
        elif heading == 'e':
            return 'w'
        elif heading == 's':
            return 'n'
        else:
            return 'e'


def makeMazeGen(maze):
    def MazeAGen():
        return MazeAgent(maze1, full=False)
    return MazeAGen

def testAgents(maze):
    finalResults = []
    for reps in range(100):
        print("============== Round", reps)
        randAg = MazeAgent(maze)
        print(randAg)
        finalResults.append(randAg.getValue())
    print(finalResults)


if __name__ == "__main__":

    maze1 = MazeInfo('file', 'SusanMazes/weightedmaze20x20.txt')
    maze1.printMaze()
    # testAgents(maze1)
    startAgent = MazeAgent(maze1, full=False)
    # hillClimb(startAgent)
    # stochHillClimb(startAgent)
    # simAnnealing(startAgent, 100.0)
    generator = makeMazeGen(maze1)
    beamSearch(generator, numStates=20)
    # geneticAlg(generator, maxGenerations=200) #, mutePerc=0.05)