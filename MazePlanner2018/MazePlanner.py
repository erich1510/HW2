"""  =================================================================
File: MazePlanner.py

This file contains code, including a Tkinter class, that implements the
GUI for this problem.  This must be run in Python 3!
 ==================================================================="""

#
from tkinter import *
import tkinter.filedialog as tkFileDialog

from SearchSolver import BestFirstSearchSolver, NoCostSearchSolver
from  MazeStateAdvisors import MazeTaskAdvisor, UCSMazeAdvisor, GreedyMazeAdvisor, AStarMazeAdvisor
from MazeInfo import MazeInfo



class MazeGUI:
    """Set up and manage all the variables for the GUI interface."""
    
    def __init__(self, dimension):
        """Given the dimension of the maze, set up a new Tk object of the right size"""
        self.root = Tk()
        self.root.title("Susan Fox's Maze Planner")
        self.numRows = dimension
        self.numCols = dimension
        self.blockPerc = 0.0
 
    def setupWidgets(self):
        """Set up all the parts of the GUI."""
        # Create title frame and main buttons
        self._initTitle()

        # Create control buttons
        self._initEditTools()

        # Create the maze grid
        self._initMazeGrid()

        # Create the search frame
        self._initSearchTools()

        # Create the message frame
        self._initMessage()
        
        # Create the legend frame
        self._initLegend()
    # end setupWidgets


              
    def goProgram(self):
        """This starts the whole GUI going"""
        self.root.mainloop()


    ### =================================================================
    ### Widget-creating helper functions
        
    def _initTitle(self):
        """Sets up the title section of the GUI, where the Quit and Help buttons are located"""
        titleButtonFrame = Frame(self.root, bd = 5, padx = 5, pady = 5)
        titleButtonFrame.grid(row = 1, column = 1)
        quitButton = Button(titleButtonFrame, text = "Quit", command = self.quit)
        helpButton = Button(titleButtonFrame, text = "Help", command = self.showHelp)
        quitButton.grid(row = 1, column = 1, padx = 5)
        helpButton.grid(row = 1, column = 2, padx = 5)

        titleFrame =  Frame(self.root, bd = 5, padx = 5, pady = 5)
        titleFrame.grid(row = 1, column = 1, columnspan = 3, padx = 5, pady = 5)

        titleLabel = Label(titleFrame, text = "Susan Fox's Maze Builder and Planner", font = "Arial 20 bold",
                           anchor=CENTER, padx = 5, pady = 5)
        titleLabel.grid(row = 1, column = 1)
    # end _initTitle
    

    def _initMessage(self):
        """Sets up the section of the window where messages appear, errors, failures, and numbers
        about how much work was done"""
        messageFrame = Frame(self.root, bd = 5, padx = 10, pady = 10, relief = "groove")
        messageFrame.grid(row = 2, column = 2,  padx = 5, pady = 5)
        self.messageVar = StringVar()
        self.messageVar.set("")
        message = Label(messageFrame, textvariable = self.messageVar, width=50, height=3, wraplength = 300)
        message.grid(row = 1, column = 1)


    def _initEditTools(self):
        """Sets up the edit tools frame and its parts, including buttons for clearing the maze, changing
        its numRows, changing modes (add walls, remove walls, place start, place goal), and loading and
        saving mazes from files"""
        self.editEnabled = True
        editFrame = Frame(self.root, bd = 5, padx = 5, pady = 5, relief="groove")
        editFrame.grid(row = 3, column = 1, rowspan = 2, padx = 5, pady = 5, sticky=N)
        editTitle = Label(editFrame, text = "Edit Maze Options", font="Arial 16 bold", anchor = CENTER)
        editTitle.grid(row = 0, column = 1, padx = 5, pady = 5)

        # Make a new maze subframe
        makerFrame = Frame(editFrame, bd=2, relief = "groove", padx = 5, pady = 5)
        makerFrame.grid(row = 1, column = 1, padx = 5, pady = 5)
        makerLabel = Label(makerFrame, text="Create New Maze", font="Arial 14 bold", anchor=CENTER)

        percLabel = Label(makerFrame, text = "% Blocked")
        rowLabel = Label(makerFrame, text = "# of Rows")
        colLabel = Label(makerFrame, text = "# of Cols")
        self.userPerc = StringVar()
        self.userRows = StringVar()
        self.userCols = StringVar()
        self.userPerc.set(str(0.0))
        self.userRows.set(str(self.numRows))
        self.userCols.set(str(self.numCols))
        self.percEntry = Entry(makerFrame, textvariable = self.userPerc, width=4, justify=CENTER)
        self.rowsEntry = Entry(makerFrame, textvariable = self.userRows, width=4, justify=CENTER)
        self.colsEntry = Entry(makerFrame, textvariable = self.userCols, width=4, justify=CENTER)

        self.flatButton = Button(makerFrame, text = "New Flat", command = self.createFlat)
        self.hillyButton = Button(makerFrame, text = "New Hilly", command = self.createHilly)

        # place the basic buttons for editing frames
        makerLabel.grid(row = 0, column = 1, columnspan=4, padx = 5)
        percLabel.grid(row=1, column = 1)
        rowLabel.grid(row = 1, column = 3)
        colLabel.grid(row = 2, column = 3)
        self.percEntry.grid(row=2, column = 1)
        self.rowsEntry.grid(row = 1, column = 4)
        self.colsEntry.grid(row = 2, column = 4)

        self.flatButton.grid(row = 3, column = 1, columnspan=2, pady = 5)
        self.hillyButton.grid(row = 3, column = 3, columnspan=2, pady = 5)

        # Edit existing maze subframe
        editSubFrame = Frame(editFrame, bd=2, relief = "groove", padx = 5, pady = 5)
        editSubFrame.grid(row = 2, column = 1, padx = 5, pady=5)

        editSubTitle = Label(editSubFrame, text = "Edit Maze", font = "Arial 14 bold", anchor = CENTER)
        editSubTitle.grid(row = 0, column = 1)#, columnspan = 2)
        # Variables related to action settings
        self.editChoice = StringVar()
        self.editChoice.set("start")


        # Create and place the radio buttons for maze editing
        self.addDelBlocks = Radiobutton(editSubFrame, variable = self.editChoice,
                                     text = "Add/Del Blocks", value = "addDelBlock", width=15, justify=LEFT)
        self.placeStart = Radiobutton(editSubFrame, variable = self.editChoice,
                                      text = "Move Start", value="start", width=15, justify=LEFT)
        self.placeGoal = Radiobutton(editSubFrame, variable = self.editChoice,
                                     text = "Move Goal", value="goal", width=15, justify=LEFT)
        self.incrWeight = Radiobutton(editSubFrame, variable = self.editChoice,
                                      text = "Increase Cost", value = "increase", width=15, justify=LEFT)
        self.decrWeight = Radiobutton(editSubFrame, variable = self.editChoice,
                                      text = "Decrease Cost", value = "decrease", width=15, justify=LEFT)
        self.addDelBlocks.grid(row = 1, column = 1)
        self.placeStart.grid(row = 3, column = 1)
        self.placeGoal.grid(row = 4, column = 1)
        self.incrWeight.grid(row = 5, column = 1)
        self.decrWeight.grid(row = 6, column = 1)

        # Load and save maze subframe
        loadSaveFrame = Frame(editFrame, bd = 2, relief = "groove", padx = 5, pady = 5)
        loadSaveFrame.grid(row = 3, column = 1, padx = 5, pady = 5)
        loadSaveTitle = Label(loadSaveFrame, text = "Load/Save Maze", font = "Arial 14 bold", anchor = CENTER)
        loadSaveTitle.grid(row = 0, column = 1)#, columnspan = 2)

        self.loadButton = Button(loadSaveFrame, text = "Load Maze", command = self.loadMaze)
        self.saveButton = Button(loadSaveFrame, text = "Save Maze", command = self.saveMaze)
        self.loadButton.grid(row = 1, column = 1, pady = 5)
        self.saveButton.grid(row = 2, column = 1, pady = 5)


    def _initMazeGrid(self):
        """sets up the maze with given dimensions, done as a helper because it may need to be done over later"""
        self.canvas = None
        self.canvasSize = 500
        self.canvasPadding = 10
        canvasFrame = Frame(self.root, bd=5, padx = 10, pady = 10, relief = "raise", bg="lemon chiffon")
        canvasFrame.grid(row = 3, column = 2, rowspan = 2, padx = 5, pady = 5)
        self.canvas = Canvas(canvasFrame,
                             width = self.canvasSize + self.canvasPadding,
                             height = self.canvasSize + self.canvasPadding)
        self.canvas.grid(row = 1, column = 1)
        self.canvas.bind("<Button-1>", self.leftClickCallback)
        self.canvas.bind("<B1-Motion>", self.motionCallback)
        self.maze = MazeInfo('gen-hilly', self.numRows, self.numCols,
                             (0, 0), (self.numRows - 1, self.numCols - 1),
                             self.blockPerc)

        self._createMazeGrid()
    # end _initMazeGrid


    def _initSearchTools(self):
        """Sets up the search frame, with buttons for selecting which search, for starting a search,
        stepping or running it, and quitting from it.  You can also choose how many steps should happen
        for each click of the "step" button"""
        searchFrame = Frame(self.root, bd = 5, padx = 10, pady = 10, relief = "groove")
        searchFrame.grid(row = 3, column = 3, padx = 5, pady = 5, sticky=N)
        searchTitle = Label(searchFrame, text="Search Options", font="Arial 16 bold")
        searchTitle.grid(row = 0, column = 1, padx = 5, pady = 5)
        self.searchType = StringVar()
        self.searchType.set("ucs")
        
        ucButton = Radiobutton(searchFrame, text = "Uniform Cost Search",
                               variable = self.searchType, value = "ucs")
        greedyButton = Radiobutton(searchFrame, text = "Greedy Best-First",
                               variable = self.searchType, value = "greedy")
        dfsButton = Radiobutton(searchFrame, text = "Depth-First",
                                   variable = self.searchType, value = "dfs")
        aStarButton = Radiobutton(searchFrame, text = "A-Star Search",
                                  variable = self.searchType, value = "astar")
        ucButton.grid(row = 1, column = 1, sticky=W)
        greedyButton.grid(row = 2, column = 1, sticky=W)
        dfsButton.grid(row = 3, column = 1, sticky=W)
        aStarButton.grid(row = 4, column = 1, sticky=W)

        resetSearch = Button(searchFrame, text = "Set Up Search", command = self.resetSearch)

        self.userStepSize = StringVar()
        self.userStepSize.set("1")
        stepSizeEntry = Entry(searchFrame, textvariable = self.userStepSize, width=3) 

        self.stepSearch = Button(searchFrame, text = "Step Search", command = self.stepSearch, state = DISABLED)
        self.runSearch = Button(searchFrame, text = "Run Search", command = self.runSearch, state = DISABLED)
        self.quitSearch = Button(searchFrame, text = "Quit Search", command = self.quitSearch, state = DISABLED)
        resetSearch.grid(row = 13, column = 1, pady = 5)
        stepSizeEntry.grid(row = 14, column = 1, pady = 5)
        self.stepSearch.grid(row = 15, column = 1)
        self.runSearch.grid(row = 16, column = 1)
        self.quitSearch.grid(row = 17, column = 1)
        self.currentSearch = None
        self.currentSearcher = None
        self.currentNode = None
    # end _initSearchTools


    def _initLegend(self):
        """Sets up the legend that describes what each color means in the maze grid"""
        legendFrame = Frame(self.root, bd = 5, padx = 10, pady = 10, relief = "groove")
        legendFrame.grid(row = 4, column = 3, padx = 5, pady = 5, sticky=N)
        legend = Canvas(legendFrame, width = 130, height = 205)
        legend.grid(row = 1, column = 1)
        legendValues = {("light gray", "white", "Open space"),
                        ("dark cyan", "black", "Blocked"),
                        ("light gray", "green", "Start cell"),
                        ("light gray", "red", "Goal cell"),
                        ("magenta", "white", "Current cell"),
                        ("light pink", "white", "Explored cell"),
                        ("light blue", "white", "Fringe cell"),
                        ("yellow", 'white', "Shortest path")}
        row = 0
        col = 0
        for (cellColor, outlineCol, cellText) in legendValues:
            (x1, y1, x2, y2) = self._posToCoords(row, col)
            legend.create_rectangle(x1, y1, x2, y2, outline=outlineCol, fill = cellColor)
            legend.create_text(x1 + 25, y1 + 8, text=cellText, anchor=NW)
            row += 1

    # end _initLegend


    ### =================================================================
    ### Helper functions


    def disableEdit(self):
        """Turn off access to the edit operations, by setting each of the GUI elements to DISABLED"""
        self.editEnabled = False
        self.flatButton.config(state=DISABLED)
        self.hillyButton.config(state=DISABLED)
        self.loadButton.config(state=DISABLED)
        self.saveButton.config(state=DISABLED)
        self.incrWeight.config(state=DISABLED)
        self.decrWeight.config(state=DISABLED)
        self.placeStart.config(state=DISABLED)
        self.placeGoal.config(state=DISABLED)
        self.addDelBlocks.config(state=DISABLED)
        self.percEntry.config(state=DISABLED)
        self.rowsEntry.config(state=DISABLED)
        self.colsEntry.config(state=DISABLED)


    def enableEdit(self):
        """Turn on access to the edit operations, by setting each GUI element to NORMAL"""
        self.editEnabled = True
        self.flatButton.config(state=NORMAL)
        self.hillyButton.config(state=NORMAL)
        self.loadButton.config(state=NORMAL)
        self.saveButton.config(state=NORMAL)
        self.incrWeight.config(state=NORMAL)
        self.decrWeight.config(state=NORMAL)
        self.placeStart.config(state=NORMAL)
        self.placeGoal.config(state=NORMAL)
        self.addDelBlocks.config(state = NORMAL)
        self.percEntry.config(state = NORMAL)
        self.rowsEntry.config(state = NORMAL)
        self.colsEntry.config(state = NORMAL)


    def disableSearch(self):
        """Turn off the search operations, by setting each GUI element to DISABLED."""
        self.stepSearch.config(state=DISABLED)
        self.runSearch.config(state=DISABLED)
        self.quitSearch.config(state=DISABLED)


    def enableSearch(self):
        """Turn on the search operations, by setting each GUI element to NORMAL"""
        self.stepSearch.config(state=NORMAL)
        self.runSearch.config(state=NORMAL)
        self.quitSearch.config(state=NORMAL)
        


    def _createMazeGrid(self):
        """This sets up the display of the maze, given the MazeInfo object.
        Re-called when dimensions changed."""
        self.idToPos = {}
        self.posToId = {}
        startPos = self.maze.getStartPos()
        goalPos = self.maze.getGoalPos()

        numRows = self.maze.getNumRows()
        numCols = self.maze.getNumCols()
        bigDim = max(numRows, numCols)
        if bigDim * 50 < self.canvasSize:
            self.cellSize = 50
        else:
            self.cellSize = self.canvasSize / bigDim

        for row in range(numRows):
            for col in range(numCols):
                (x1, y1, x2, y2) = self._posToCoords(row, col)
                currId = self.canvas.create_rectangle(x1, y1, x2, y2)
                self.idToPos[currId] = (row, col)
                self.posToId[row, col] = currId
        self._displayMazeGrid()


    def _displayMazeGrid(self):
        numRows = self.maze.getNumRows()
        numCols = self.maze.getNumCols()
        for row in range(numRows):
            for col in range(numCols):
                currId = self._posToId(row, col)
                (outlineColor, cellColor) = self._determineColor((row, col))
                self._setOutlineColor(currId, outlineColor)
                self._setCellColor(currId, cellColor)



    def _determineColor(self, currPos):
        (row, col) = currPos
        (sRow, sCol) = self.maze.getStartPos()
        (gRow, gCol) = self.maze.getGoalPos()
        if row == sRow and col == sCol:
            outlineColor = 'green'
        elif row == gRow and col == gCol:
            outlineColor = 'red'
        elif self.maze.isBlocked(row, col):
            outlineColor = 'black'
        else:
            outlineColor = 'white'

        weight = self.maze.getWeight(row, col)
        maxWgt = self.maze.getMaxWeight()
        if self.maze.isBlocked(row, col):
            cellColor = 'dark cyan'
        else:
            diff = maxWgt - weight
            ratio = diff / maxWgt
            grayColor = int( (ratio * 245) + 10 )
            cellColor = "#%02x%02x%02x" % (grayColor, grayColor, grayColor)
        return (outlineColor, cellColor)



    ### =================================================================
    ### The following are callbacks for the canvas itself
    ### for the grid of rectangles
    
    def leftClickCallback(self, event):
        """This is a callback that happens when the user clicks in the maze part of the window.
        If edit is not enabled, then nothing happens.  If edit is enabled, then this finds the
        square that was clicked on, and changes its color.  Exactly how the color changes depends
        on a helper, and what color the grid square already was."""
        if self.editEnabled:
            items = self.canvas.find_withtag("current")
            if items == ():
                return
            item = items[0]
            (row, col) = self._idToPos(item)
            self._changeSquare(row, col, item)


    def motionCallback(self, event):
        """This is a callback that happens when the user drags the mouse over the maze part.
        This finds the cell that corresponds to the location of the mouse, and changes its color.
        This allows the user to drag to create walls for the maze"""
        if self.editEnabled:
            (row, col) = self._coordToPos(event.x, event.y)
            item = self._posToId(row, col)
            self._changeSquare(row, col, item)


    def _changeSquare(self, row, col, item):
        """This function takes the row and column of an grid square item, and the item itself.
        It changes the color depending on what mode it is in.  The self.editChoice variable holds
        the mode the user has selected: black for adding walls, white for removing them, green for placing
        the start location, red for placing the goal.  This first handles the cases where the user
        is placing start or goal, because there can be only one start and goal location.  Thus, turning
        one grid cell green requires that we find the old green one (if any) and turn it white.  This calls a 
        helper that actually does the color change"""
        currEdit = self.editChoice.get()
        if currEdit == "start":
            (sRow, sCol) = self.maze.getStartPos()
            startObj = self._posToId(sRow, sCol)
            self._setOutlineColor(startObj, 'white')
            self.maze.setStartPos( (row, col) )
        elif currEdit == "goal":   # if setting goal position
            (gRow, gCol) = self.maze.getGoalPos()
            goalObj = self._posToId(gRow, gCol)
            self._setOutlineColor(goalObj, 'white')
            self.maze.setGoalPos( (row, col) )
        elif currEdit == "addDelBlock":  # if adding or blocks
            if self.maze.isBlocked(row, col):
                self.maze.delBlocked(row, col)
            else:
                self.maze.addBlocked(row, col)
        else:
            if currEdit == 'increase':
                self.maze.increaseWeight(row, col)
            else:
                self.maze.decreaseWeight(row, col)
        (outCol, cellCol) = self._determineColor((row, col))
        self._setCellColor(item, cellCol)
        self._setOutlineColor(item, outCol)
                
        


    ### =================================================================
    ### The following are callbacks for buttons

    def showHelp(self):
        """Pop up a new window and display help information"""
        self.helpWindow = Toplevel()
        helpTxt = """The Edit Options Frame:
        Create New Maze contains options for creating a new maze
        -- % Blocked is the percentage of the squares in the maze that should be blocked to the agent
        -- # of Rows is the number of rows in the maze
        -- # of Cols is the number of columns in the maze
        -- New Flat creates a new "flat" landscape where all cells except blocked ones have the same (low) cost
        -- New Hilly creates a new "hilly" landscape
        Edit Maze contains options for editing the existing maze
        -- Add/Del Blocks: click on squares to create or remove blocks from the maze
        -- Move Start: click on squares to move the start location
        -- Move Goal: click on squares to move the goal location
        -- Increase Cost: click on squares to increase the cost of passing through that cell
        -- Decrease Cost: click on squares to decrease the cost of passing through that cell
        Load/Save Maze has buttons for loading a maze from a file or saving a maze to a file

        The Search Frame:
        -- Select one of the implemented search algorithms with the radio buttons
        -- Set Up Search activates the search frame, and initializes the selected search
        -- Step Search will perform N steps of the search, where N appears in the text box above it
        -- Run Search will perform the whole search process
        -- Quit Search disables the search frame, and enables the edit frame

        Messages appear in the box above the maze grid.
        A "legend" for the maze grid appears below the search frame.
"""
        textField = Label(self.helpWindow, text = helpTxt, padx = 10, pady = 10, justify=LEFT)
        closeButton = Button(self.helpWindow, text = "Close", command = self.closeHelp)
        textField.grid(row = 1, column = 1)
        closeButton.grid(row = 2, column = 1)


    def closeHelp(self):
        """When the user closes the help window, destroy it"""
        self.helpWindow.destroy()


    def quit(self):
        """Callback for the quit button: destroy everything"""
        self.root.destroy()


    # ----------------------------------------------------------------
    # Button callbacks for Edit buttons

    
    def createFlat(self):
        """Creates a new flat maze of the given number of rows and columns. Note that this is a
        callback to the Create New Flat button, and it must first read the value of the StringVars
        to update the given size."""
        self._makeNewMaze('gen-flat')


    def createHilly(self):
        """Creates a new hilly maze of the given number of rows and columns. Note that this is a
        callback to the Create New Hilly button, and it must first read the value of the StringVars
        to update the given size."""
        self._makeNewMaze('gen-hilly')


    def _makeNewMaze(self, mode):
        self._removeMazeCells()
        userRowNum = self.userRows.get()
        userColNum = self.userCols.get()
        userPercNum = self.userPerc.get()
        try:
            self.numRows = int(userRowNum)
            self.numCols = int(userColNum)
            self.blockPerc = float(userPercNum)
        except:
            self._postMessage("# of Rows and # of Columns must be positive integers. % Blocked must be positive float.")
            return
        if self.blockPerc > 1.0:
            self.blockPerc = self.blockPerc / 100.0
        self.maze = MazeInfo(mode, self.numRows, self.numCols,
                             (0, 0), (self.numRows - 1, self.numCols - 1),
                             self.blockPerc)
        self._createMazeGrid()
        self.currentSearch = None


    def _removeMazeCells(self):
        """A helper that removes all the grid cell objects from the maze, prior to creating new
        ones when the "Change Dimension" button is clicked"""
        for row in range(self.numRows):
            for col in range(self.numCols):
                currId = self.posToId[row, col]
                self.canvas.delete(currId)
        self.canvas.update()
        self.posToId = {}
        self.idToPos = {}


    def loadMaze(self):
        """Callback for loading a maze. It uses a utility to ask the user to
        load a file. Then it clears the maze and resets it using the
        information from the file. The file contains text, first the
        numRows of the maze, and then that many lines. Each line contains
        words separated by spaces that describe the color for the
        corresponding cell. Thus there are numRows number of words per
        line."""
        fileName = tkFileDialog.askopenfilename(title = "Select the file to load")
        if fileName != None:
            self.maze = MazeInfo('file', fileName)
            self._removeMazeCells()
            self.numRows = self.maze.getNumRows()
            self.numCols = self.maze.getNumCols()
            self._createMazeGrid()
            self.currentSearch = None

            
            
                            
 
    def saveMaze(self):
        """This pops up a dialog box to save a maze to a file.  Note it won't save a maze with no
        start or goal.  It asks the MazeInfo object to write itself to this file."""
        fileName = tkFileDialog.asksaveasfilename(title = "Select the file to which to save the current maze",
                                                  initialfile = "maze.txt")
        self.maze.writeGridToFile(fileName)


    # ----------------------------------------------------------------
    # Button callbacks for Search buttons

    def resetSearch(self):
        """This is a callback for the Set Up Search button.  It generates a message
        if the maze is incomplete. It resets all search-related qData, and it creates
        a MazeInfo object to represent the maze (this is done only at this point, because
        the mazes are otherwise edit-able, until we are searching there is not point in having
        and updating a MazeInfo object). It then creates the right kind of MazeSolver object, and
        initializes the search.  It turns off the edit mode, and turns on the search mode"""
        self._clearMessage()

        self._displayMazeGrid()
        self.currentNode = None
        self.currentSearch = self.searchType.get()
        (sRow, sCol) = self.maze.getStartPos()
        (gRow, gCol) = self.maze.getGoalPos()
        if self.currentSearch == "ucs":
            taskAdvisor = UCSMazeAdvisor(self.maze, sRow, sCol, gRow, gCol)
            self.currentSearcher = BestFirstSearchSolver(taskAdvisor)
        elif self.currentSearch == 'greedy':
            taskAdvisor = GreedyMazeAdvisor(self.maze, sRow, sCol, gRow, gCol)
            self.currentSearcher = BestFirstSearchSolver(taskAdvisor)
        elif self.currentSearch == 'dfs':
            taskAdvisor = MazeTaskAdvisor(self.maze, sRow, sCol, gRow, gCol)
            self.currentSearcher = NoCostSearchSolver(taskAdvisor, 'DFS')
        elif self.currentSearch == 'astar':
            taskAdvisor = AStarMazeAdvisor(self.maze, sRow, sCol, gRow, gCol)
            self.currentSearcher = BestFirstSearchSolver(taskAdvisor)
        self.currentSearcher.initSearch()
        self.disableEdit()
        self.enableSearch()

        
        
    def runSearch(self):
        """This callback for the Run Search button keeps running steps of the search until the search is done
        or a problem crops up.  """
        keepLooping = True
        while keepLooping:
            keepLooping = self._handleOneStep()
        # end while
    # end runSearch
    

    def stepSearch(self):
        """This repeats the current stepCount number of steps of the search, stopping after that.
        Otherwise, this is very similar to the previous function"""
        stepCount = int(self.userStepSize.get())
        keepLooping = True
        while keepLooping and stepCount > 0:
            keepLooping = self._handleOneStep()
            stepCount -= 1
 

    def _handleOneStep(self):
        """This helper helps both the run search and step search callbacks, by handling the
        different outcomes for one step of the search.  
        NOTE: this could be improved, there is really no reason to have separate
        method names for the step function in the AStar maze searcher, versus the UCS maze searcher.  Change that,
        and you could eliminate the need for an if-else to pick what method to call."""
        if self._problemWithSearch():
            return False

        (nextState, newFringe, status) = self.currentSearcher.searchStep()

        if status == "Fail":
            self._postMessage("No path possible")
            return False
        elif status == "Done":
            self._postMessage("All done")
            self.markCells(nextState, newFringe)
            self.wrapUpSearch(nextState, newFringe)
            return False
        else:
            self.markCells(nextState, newFringe)
            return True

        
    def wrapUpSearch(self, nextState, newFringe):
        """This produces the ending statistics, finds and marks the final path, and then closes
        down the search so it will not continue"""
        finalPath = nextState.getPath()
        (currRow, currCol) = self.maze.getStartPos()
        totalCost = 0
        for move in finalPath:
            totalCost += self.maze.getWeight(currRow, currCol)
            (nextRow, nextCol) = self._makeMove(currRow, currCol, move)
            nextId = self._posToId(nextRow, nextCol)
            self._setCellColor(nextId, "yellow")
            currRow = nextRow
            currCol = nextCol

        printStr = "Total path cost = %d      " % totalCost
        printStr += "Path length = %d\n" % len(finalPath)
        printStr += "Nodes created = %d      " % self.currentSearcher.getNodesCreated()
        printStr += "Nodes visited = %d" % self.currentSearcher.getNodesVisited()
        self._postMessage(printStr)
        self.currentSearch = None
        self.currentNode = None        
 


    def quitSearch(self):
        """A callback for clearing away the search and returning to edit mode"""
        self._displayMazeGrid()
        self.disableSearch()
        self.enableEdit()
        self.currentSearch = None
        self.currentNode = None
        self._clearMessage()

        

    def markCells(self, nextState, newFringe):
        """This function changes the color of grid cells being searched to reflect the search. It
        keeps track of the old node, and changes its color, and changes all fringe nodes to light
        blue.  Visited nodes become light pink."""
        if self.currentNode != None:
            (oldR, oldC) = self.currentNode.getLocation()
            oldId = self._posToId(oldR, oldC)
            self._setCellColor(oldId, "light pink")
        self.currentNode = nextState
        (r, c) = nextState.getLocation()
        nextId = self._posToId(r,c)
        self._setCellColor(nextId, "magenta")
            
        for f in newFringe:
            (fr, fc) = f.getLocation()
            fId = self._posToId(fr, fc)
            self._setCellColor(fId, "light blue")
        self.canvas.update()

        


    def _problemWithSearch(self):
        """This looks for problems with the search, like no start or goal, or times when the user
        clicked the UCS/AStar button in the middle of a search, or when something else has
        caused a problem.  It displays a message for each problem, and then returns True.  If no problems
        appear, it returns False"""
        if self.currentSearch != self.searchType.get():
            self._postMessage("Search type has changed, or search has ended, click 'Set Up Search'")
            return True
        elif self.currentSearch == None or self.currentSearcher == None:
            self._postMessage("Search uninitialized, click 'Set Up Search'")
            return True
        else:
            return False
    
    # -------------------------------------------------
    # Utility functions


    def _postMessage(self, messageText):
        """Posts a message in the message box"""
        self.messageVar.set(messageText)

    def _clearMessage(self):
        """Clears the message in the message box"""
        self.messageVar.set("")
        
    
    def _setCellColor(self, cellId, color):
        """Sets the grid cell with cellId, and at row and column position, to have the
        right color.  Note that in addition to the visible color, there is also a colors 
        matrix that mirrors the displayed colors"""
        self.canvas.itemconfig(cellId, fill = color)


    def _setOutlineColor(self, cellId, color):
        """Sets the outline of the grid cell with cellID, and at row and column position, to
        have the right color."""
        self.canvas.itemconfig(cellId, outline=color)


    def _makeMove(self, row, col, move):
        """Given a row and column, and a direction to move, this returns the new row and column
        values associated with that move"""
        if move == "N":
            return row - 1, col
        elif move == "S":
            return row + 1, col
        elif move == "W":
            return row, col - 1
        elif move == "E":
            return row, col + 1
        else:
            return row, col


    def _posToId(self, row, col):
        """Given row and column indices, it looks up and returns the GUI id of the cell at that location"""
        return self.posToId[row, col]

    def _idToPos(self, currId):
        """Given the id of a cell, it looks up and returns the row and column position of that cell"""
        return self.idToPos[currId]

    
    def _posToCoords(self, row, col):
        """Given a row and column position, this converts that into a position on the frame"""
        x1 = col * self.cellSize + 5
        y1 = row * self.cellSize + 5
        x2 = x1 + (self.cellSize - 2)
        y2 = y1 + (self.cellSize - 2)
        return (x1, y1, x2, y2)

    def _coordToPos(self, x, y):
        """Given a position in the frame, this converts it to the corresponding row and column"""
        col = (x - 5) / self.cellSize
        row = (y - 5) / self.cellSize
        if row < 0:
            row = 0
        elif row >= self.numRows:
            row = self.numRows - 1

        if col < 0:
            col = 0
        elif col >= self.numRows:
            col = self.numRows - 1
            
        return (int(row), int(col))
# End of MazeGUI class


            
def RunMaze(size = 20):
    """This starts it all up.  Sets up the MazeGUI, and its widgets, and makes it go"""
    s = MazeGUI(size)
    s.setupWidgets()
    s.goProgram()



# The lines below cause the maze to run when this file is double-clicked or sent to a launcher, or loaded
# into the interactive shell.
if __name__ == "__main__":
    RunMaze()
