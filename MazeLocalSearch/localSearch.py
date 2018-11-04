
import random
import math

verbose = True

# ==================================================================
# This section contains an implementation of straightforward
# Hill Climbing. It requires a state class that creates objects
# that implement the following methods: getValue, getMaxValue,
# allNeighbors, randomNeighbors, and that are printable


def hillClimb(startState, maxRounds=1000):
    """Perform the hill-climbing algorithm, starting with the given
    start state and going until a local maxima is found or the
    maximum rounds is reached"""
    curr = startState
    value = curr.getValue()
    maxValue = curr.getMaxValue()
    count = 0
    if verbose:
        print("============= START ==============")
    while value < maxValue and count < maxRounds:
        if verbose:
            print("--------- Count =", count, "---------")
            print(curr)
        neighs = curr.allNeighbors()
        bestNeigh = findBestNeighbor(neighs)
        nextValue = bestNeigh.getValue()
        if nextValue >= value:
            if verbose:
                print("Best neighbor:")
                print(bestNeigh)
            curr = bestNeigh
            value = nextValue
        else:
            break
        count += 1
    if verbose:
        print("============== FINAL STATE ==============")
        print(curr)
        print("   Number of steps =", count)
        if value == maxValue:
            print("  FOUND PERFECT SOLUTION")
    return value, maxValue, count

def findBestNeighbor(neighbors):
    """Given a list of neighbors and values, find and return a neighbor with
    the best value. If there are multiple neighbors with the same best value,
    a random one is chosen"""
    startBest = neighbors[0]
    bestValue = startBest.getValue()
    bestNeighs = [startBest]
    for neigh in neighbors:
        value = neigh.getValue()
        if value > bestValue:
            bestNeighs = [neigh]
            bestValue = value
        elif value == bestValue:
            bestNeighs.append(neigh)
    bestNeigh = random.choice(bestNeighs)
    return bestNeigh


# ==================================================================
# This section contains an implementation of stochastic
# Hill Climbing. Similar to the basic hill-climbing, this function
# generates a fixed number of neighbors, not all, and takes the best
# one


def stochHillClimb(startState, numNeighs = 5, maxRounds = 1000):
    curr = startState
    value = curr.getValue()
    maxValue = curr.getMaxValue()
    count = 0
    if verbose:
        print("============= START ==============")
    while value < maxValue and count < maxRounds:
        if verbose:
            print("--------- Count =", count, "---------")
            print(curr)

        neighs = curr.randomNeighbors(numNeighs)
        result = stochFindBestNeighbor(neighs, value)
        if verbose:
            printStateList(neighs, False)
        if result is not False:
            # found better neighbor
            bestNeigh = result
            nextValue = bestNeigh.getValue()
            if verbose:
                print("Best neighbor:")
                print(bestNeigh)
            curr = bestNeigh
            value = nextValue
        count += 1
    if verbose:
        print ("============== GOAL ==============")
        print(curr)
        print( "   Number of steps =", count)
    return value, maxValue, count




def stochFindBestNeighbor(neighbors, currValue):
    """Given a list of neighbors and values, find and return a neighbor with
    a better value. Uses roulette-wheel selection to choose a
    better-than-current neighbor."""
    bestNeighs = []
    for neigh in neighbors:
        value = neigh.getValue()
        if value >= currValue:
            bestNeighs.append(neigh)
    if bestNeighs == []:
        return False
    # Now we know that bestNeighs are the neighbors with a better or equal
    # value.  Now use roulette wheel selection with change in value as
    # the value measure: the larger, the better
    deltaValues = [neigh.getValue() - currValue for neigh in bestNeighs]
    bestPos = rouletteSelect(deltaValues)
    return bestNeighs[bestPos]


# ==================================================================
# This section contains an implementation of simulated annealing.  This
# algorithm randomly generates a move from the current state.  If the randomly
# generated move is better than the current one, then it makes that move.  If
# it is worse, then it decides stochastically whether to take the move or not.
# This involves both the difference in value, and also the current temperature.
# The states involved here need to implement the same set of methods as before,
# Plus a makeRandomMove method, that returns a new state one off from the
# previous one."""


def simAnnealing(startState, initTemp=5.0):
    """This takes in a start state and an initial temperature, and it runs
    until the temperature goes to zero. """
    currTemp = initTemp
    currState = startState
    currState.setPrintMode(full=False)
    currValue = currState.getValue()
    maxValue = currState.getMaxValue()
    count = 0
    if verbose:
        print("============= START ==============")
    while currTemp > 0 and currValue < maxValue:
        if verbose:
            tempStr = "{:6.2f}".format(currTemp)
            print("--------- Count =", count, "Temp =", tempStr, "---------")
            print(currState)

        nextState = currState.makeRandomMove()
        nextValue = nextState.getValue()
        diff = nextValue - currValue
        if diff >= 0:   # next state is better always move to it
            if verbose:
                print("Better next state:")
                print(nextState)
            currState = nextState
            currValue = nextValue
        else:
            threshold = math.e ** (diff / float(currTemp))
            randValue = random.random()
            if randValue <= threshold:
                if verbose:
                    print("Taking lesser step (", diff, threshold, currTemp, "):")
                    print(nextState)
                currState = nextState
                currValue = nextValue
            elif verbose:
                print("Next state was worse, trying again")

        currTemp -= 0.1
        count += 1
    if verbose:
        print("============== GOAL ==============")
        print(currState)
        print("   Number of steps =", count)
    return currValue, maxValue, count


# ==================================================================
# This section contains an implementation of beam search.  This algorithm
# randomly generates n starting points.  It then generates all the successors
# of each of the n states, and puts them in one pool.  The top n successors
# are kept at each round.


def beamSearch(stateGen, numStates = 10, stopLimit=500):
    currStates = []
    for i in range(numStates):
        nextState = stateGen()
        nextState.setPrintMode(full=False)
        currStates.append(nextState)
    maxValue = currStates[0].getMaxValue()
    sortByValue(currStates)
    if verbose:
        print("================ Initial States ================")
        printStateList(currStates)
        print("================================================")
    count = 0
    foundOptimal = False
    while (not foundOptimal) and (count < stopLimit):
        if verbose:
            print("Round", count)
        bestNNeighs = []
        for nextState in currStates:
            neighs = nextState.allNeighbors()
            (bestNNeighs, foundOptimal) = keepBestNNeighbors(bestNNeighs, neighs, numStates, maxValue)
            if foundOptimal:
                if verbose:
                    print("Found optimal!")
                break
        currStates = bestNNeighs
        if verbose:
            printStateList(currStates)
            print("================================================")
        count += 1
        state = currStates[0]
    if verbose:
        print("============== GOAL ==============")
        print(state)
        print("   Number of steps =", count)
    return state.getValue(), maxValue, count


def sortByValue(stateList):
    stateList.sort(key=lambda neigh: - neigh.getValue())


def keepBestNNeighbors(bestSoFar, neighs, n, maxVal):
    """Takes in a list of all neighbors, and the number to select, and it selects
    the best n neighbors.  If one of the neighbors is optimal, then it returns
    just that neighbor, and the flag True.  If none is optimal, it returns the best
    n of them, with the flag False."""
    sortByValue(neighs)
    bestNeigh = neighs[0]
    if bestNeigh.getValue() == maxVal:  # if we have found an optimal solution
        return ([bestNeigh], True)
    else:
        i = 0
        while i < len(neighs):
            nextNeigh = neighs[i]
            if len(bestSoFar) == n:
                worstOfBest = bestSoFar[-1]
                if nextNeigh.getValue() < worstOfBest.getValue():
                    break
            insertState(bestSoFar, nextNeigh, n)
            i = i + 1
        return (bestSoFar, False)


def insertState(sortedList, newState, limit):
    """Takes in a list sorted by value, with highest values at the front, and it
    inserts the new state in the proper place. There is a length limit; if exceeded
    then the last element (the one with lowest value) is removed."""
    i = 0
    for state in sortedList:
        if newState.getValue() > state.getValue():
            break
        i = i + 1
    sortedList.insert(i, newState)
    if len(sortedList) > limit:
        sortedList.pop(-1)


# ==================================================================
# This section contains an implementation of genetic algorithm search. This
# algorithm randomly generates n starting points.  It then chooses n "parents"
# from the population, based on roulette-wheel selection, which is based on
# the value/fitness of each state.  Another way to put this is that it samples
# with replacement from the probability distribution that corresponds to the
# amount of fitness the individual is responsible for. It crosses over parents
# with each other to create a new generation, and then continues.


def geneticAlg(stateGen, popSize=30, maxGenerations=2000, crossPerc=0.8, mutePerc=0.01):
    """ Given a population size and problem size, it generates a set of random
states of the given population size.  It then repeats until an optimal solution
is found (dangerous chance of infinite loop here) and selects a set of parents
using weighted roulette-wheel selection.  It then crosses the parents over to make
a new population, and repeats.
The stateGen input is a function that """
    if popSize % 2 == 1:  # if user puts in an odd population size, make it even
        print("Making population size even:")
        popSize += 1
    currStates = []
    for i in range(popSize):
        nextState = stateGen()
        currStates.append(nextState)
    maxFit = currStates[0].getMaxValue()

    if verbose:
        print("================ Initial States ================")
        printStateList(currStates, False)
        print("================================================")
    count = 0
    foundOptimal = False
    overallBest = currStates[0]
    while (not foundOptimal) and count < maxGenerations:
        count += 1
        if verbose:
            print("Generation", count)
        fits = [state.getValue() for state in currStates]
        if maxFit in fits:  # we have an optimal solution
            pos = fits.index(maxFit)
            bestOne = currStates[pos]
            foundOptimal = True
        else:
            if verbose:
                print("Average fitness:", sum(fits) / len(fits))
                print("Max fitness:", max(fits))
                print("Min fitness:", min(fits))
            bestLoc = fits.index(max(fits))
            bestOne = currStates[bestLoc]
            parentPool = selectParents(currStates, fits)
            if verbose:
                print("Parents:")
                printStateList(parentPool, False)
            currStates = mateParents(parentPool, crossPerc, mutePerc)
            if verbose:
                printStateList(currStates, False)
                print("==============================================")
        if bestOne.getValue() > overallBest.getValue():
            overallBest = bestOne
    if verbose:
        print("============== GOAL ==============")
        print("  Last generation best one:")
        print(bestOne)
        print("  Overall best discovered:")
        print(overallBest)
        print("   Number of steps =", count)
    return bestOne.getValue(), maxFit, count


def selectParents(states, fitnesses):
    """given a set of states, repeatedly select parents using roulette selection"""
    parents = []
    for i in range(len(states)):
        nextParentPos = rouletteSelect(fitnesses)
        parents.append(states[nextParentPos])
    return parents


def mateParents(parents, crossoverPerc, mutationPerc):
    """Given a set of parents, pair them up and cross them together to make
    new kids"""
    newPop = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[i + 1]
        doCross = random.random()
        if doCross < crossoverPerc:
            n1, n2 = p1.crossover(p2)
            newPop.append(n1)
            newPop.append(n2)
        else:
            newPop.append(p1.copyState())
            newPop.append(p2.copyState())
    for i in range(len(newPop)):
        nextOne = newPop[i]
        doMutate = random.random()
        if doMutate <= mutationPerc:
            newPop[i] = nextOne.makeRandomMove()
    return newPop



# ========================================================================
# This next section contains utility functions used by more than one of the algorithms


def rouletteSelect(valueList):
    """takes in a list giving the values for a set of entities.  It randomly
selects one of the positions in the list by treating the values as a kind of
probability distribution and sampling from that distribution.  Each entity gets
a piece of a roulette wheel whose size is based on comparative value: high-value
entities have the highest probability of being selected, but low-value entities have
*some* probability of being selected."""
    # First adjust values so that they are all greater than zero
    smallest = min(valueList)
    if smallest <= 0:
        valueList = [v + abs(smallest) + 1 for v in valueList]
    totalValues = sum(valueList)
    pick = random.random() * totalValues
    s = 0
    for i in range(len(valueList)):
        s += valueList[i]
        if s >= pick:
            return i
    return len(valueList) - 1


def addNewRandomMove(state, stateList):
    """Generates new random moves (moving one queen within her column) until
    it finds one that is not already in the list of boards. If it finds one,
    then it adds it to the list. If it tries 100 times and doesn't find one,
    then it returns without changing the list"""
    nextNeigh = state.makeRandomMove()
    count = 0

    while alreadyIn(nextNeigh, stateList):
        nextNeigh = state.makeRandomMove()
        count += 1
        if count > 100:
            # if tried 100 times and no valid new neighbor, give up!
            return
    stateList.append(nextNeigh)


def alreadyIn(state, stateList):
    """Takes a state and a list of state, and determines whether the state
    already appears in the list of states"""
    for s in stateList:
        if state == s:
            return True
    return False


def printStateList(stateList, full = True):
    """Takes a list of neighbors and values, and prints them all out"""
    for state in stateList:
        state.setPrintMode(full)
        print(state)
