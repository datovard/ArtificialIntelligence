import Queue as Q

from collections import deque

import time

import resource

import sys

import math

import heapq as hq

#### SKELETON CODE ####

## The Class that Represents the Puzzle

class PuzzleState(object):
    """docstring for PuzzleState"""
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []

        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i / self.n
                self.blank_col = i % self.n
                break

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])

            print line

    def move_left(self):
        if self.blank_col == 0:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None

        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR

        if len(self.children) == 0:
            up_child = self.move_up()

            if up_child is not None:
                self.children.append(up_child)
            down_child = self.move_down()

            if down_child is not None:
                self.children.append(down_child)
            left_child = self.move_left()

            if left_child is not None:
                self.children.append(left_child)
            right_child = self.move_right()

            if right_child is not None:
                self.children.append(right_child)
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters

def getPathToGoal(state):
    ans = []

    while(state.parent != None):
        ans.append( state.action )
        state = state.parent

    return list(reversed(ans))

def writeOutput(state, count, max_search_depth, time):
    ### Student Code Goes here
    f = open("output.txt", "w+")

    path = getPathToGoal(state)
    f.write( "path_to_goal: %s\n" % str(path) )
    f.write( "cost_of_path: %d\n" % len(path) )
    f.write( "nodes_expanded: %d\n" % count )
    f.write( "search_depth: %d\n" % state.cost )
    f.write( "max_search_depth: %d\n" % max_search_depth )
    f.write( "running_time: %.5f\n" % time )
    f.write( "max_ram_usage: %.5f\n" % ( float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(1024.0) ) )
    f.close()

def bfs_search(initial_state):
    """BFS search"""
    d = deque()
    d.append( initial_state )
    memory = set([])
    memory.add(''.join(map(str, initial_state.config)))
    nodes_count = 0
    max_search_depth = 0

    while len(d):
        state = d.popleft()

        if test_goal(state):
            return state, nodes_count, max_search_depth

        nodes_count += 1
        children = state.expand()
        if children[0].cost > max_search_depth: max_search_depth = children[0].cost
        for i in children:
            if ''.join(map(str, i.config)) not in memory:
                d.append(i)
                memory.add(''.join(map(str, i.config)))

def dfs_search(initial_state):
    """DFS search"""
    d = deque()
    d.append(initial_state)
    memory = set([])
    memory.add(''.join(map(str, initial_state.config)))
    nodes_count = 0
    max_search_depth = 0

    while len(d):
        state = d.pop()

        if state.cost > max_search_depth: max_search_depth = state.cost

        if test_goal(state):
            return state, nodes_count, max_search_depth

        children = state.expand()

        nodes_count += 1
        #if children[0].cost > max_search_depth: max_search_depth = children[0].cost
        for i in xrange(len(children)-1, -1, -1):
            if ''.join(map(str, children[i].config)) not in memory:
                d.append(children[i])
                memory.add(''.join(map(str, children[i].config)))


def A_star_search(initial_state):
    """A * search"""
    h = []
    hq.heappush(h, (calculate_total_cost(initial_state), initial_state.cost, initial_state))
    memory = set([])
    memory.add(''.join(map(str, initial_state.config)))
    nodes_count = 0
    max_search_depth = 0

    while len(h):
        cost, action, state = hq.heappop(h)

        if test_goal(state):
            return state, nodes_count, max_search_depth

        nodes_count += 1
        children = state.expand()

        if children[0].cost > max_search_depth: max_search_depth = children[0].cost
        for i in children:
            if not ''.join(map(str, i.config)) in memory:
                action = 0
                if i.action == "Up": action = 0
                elif i.action == "Down": action = 1
                elif i.action == "Left": action = 2
                elif i.action == "Right": action = 3
                hq.heappush(h, (calculate_total_cost(i), i.cost, i))
                memory.add(''.join(map(str, i.config)))

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    total = 0
    for i in xrange(len(state.config)):
        if state.config[i] != 0:
            sum = calculate_manhattan_dist( i, state.config[i], state.n )
            total += sum
    return total

def calculate_manhattan_dist(idx, value, n):
    """calculatet the manhattan distance of a tile"""

    i_pos = idx / n
    j_pos = idx % n
    i_val = value / n
    j_val = value % n

    return abs(i_pos - i_val) + abs(j_pos - j_val)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    return ''.join(map(str, puzzle_state.config)) == "012345678"


# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)


    if sm == "bfs":
        start_time = time.time()
        state, nodes_count, max_search_depth = bfs_search(hard_state)
        writeOutput(state, nodes_count, max_search_depth, (time.time() - start_time))


    elif sm == "dfs":
        start_time = time.time()
        state, nodes_count, max_search_depth = dfs_search(hard_state)
        writeOutput(state, nodes_count, max_search_depth, (time.time() - start_time))

    elif sm == "ast":
        start_time = time.time()
        state, nodes_count, max_search_depth = A_star_search(hard_state)
        writeOutput(state, nodes_count, max_search_depth, (time.time() - start_time))

    else:
        print("Enter valid command arguments !")

if __name__ == '__main__':
    main()