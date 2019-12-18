
import random

#This function is the AI decicion making itself
def AI_attack_coordinates(player_board):


    #check single hits that have empty space next to them

    potential_targets = []

    for row_index,row in enumerate(player_board.board): #scan through every row of player board

        for element_index, element in enumerate(row): #scan through every element of each row
            if element == " Hit   ":

                #check corners for adjacent empty spaces
                if row_index == 0 and element_index == 0:
                    if player_board.board[row_index,element_index + 1] != " Hit   " and player_board.board[row_index,element_index + 1] != " Miss  ":
                        potential_targets.append((row_index,element_index + 1))
                    if player_board.board[row_index + 1,element_index] != " Hit   " and player_board.board[row_index + 1,element_index] != " Miss  ":
                        potential_targets.append((row_index + 1,element_index))

                elif row_index == 0 and element_index == 9:
                    if player_board.board[row_index,element_index - 1] != " Hit   " and player_board.board[row_index,element_index - 1] != " Miss  ":
                        potential_targets.append((row_index,element_index - 1))
                    if player_board.board[row_index + 1,element_index] != " Hit   " and player_board.board[row_index + 1,element_index] != " Miss  ":
                        potential_targets.append((row_index + 1,element_index))

                elif row_index == 9 and element_index == 0:
                    if player_board.board[row_index,element_index + 1] != " Hit   " and player_board.board[row_index,element_index + 1] != " Miss  ":
                        potential_targets.append((row_index,element_index + 1))
                    if player_board.board[row_index - 1,element_index] != " Hit   " and player_board.board[row_index - 1,element_index] != " Miss  ":
                        potential_targets.append((row_index - 1,element_index))

                elif row_index == 9 and element_index == 9:
                    if player_board.board[row_index,element_index - 1] != " Hit   " and player_board.board[row_index,element_index - 1] != " Miss  ":
                        potential_targets.append((row_index,element_index - 1))
                    if player_board.board[row_index - 1,element_index] != " Hit   " and player_board.board[row_index - 1,element_index] != " Miss  ":
                        potential_targets.append((row_index - 1,element_index))

                #check sides for adjacent empty spaces
                #top row
                elif row_index == 0 and element_index != 0 and element_index != 9:
                    if player_board.board[row_index,element_index + 1] != " Hit   " and player_board.board[row_index,element_index + 1] != " Miss  ":
                        potential_targets.append((row_index,element_index + 1))
                    if player_board.board[row_index,element_index - 1] != " Hit   " and player_board.board[row_index,element_index - 1] != " Miss  ":
                        potential_targets.append((row_index,element_index - 1))
                    if player_board.board[row_index + 1,element_index] != " Hit   " and player_board.board[row_index + 1,element_index] != " Miss  ":
                        potential_targets.append((row_index + 1,element_index))
                #bottom row
                elif row_index == 9 and element_index != 0 and element_index != 9:
                    if player_board.board[row_index,element_index + 1] != " Hit   " and player_board.board[row_index,element_index + 1] != " Miss  ":
                        potential_targets.append((row_index,element_index + 1))
                    if player_board.board[row_index,element_index - 1] != " Hit   " and player_board.board[row_index,element_index - 1] != " Miss  ":
                        potential_targets.append((row_index,element_index - 1))
                    if player_board.board[row_index - 1,element_index] != " Hit   " and player_board.board[row_index - 1,element_index] != " Miss  ":
                        potential_targets.append((row_index - 1,element_index))

                #left side
                elif row_index != 0 and row_index != 9 and element_index == 0:
                    if player_board.board[row_index - 1,element_index] != " Hit   " and player_board.board[row_index - 1,element_index] != " Miss  ":
                        potential_targets.append((row_index - 1,element_index))
                    if player_board.board[row_index,element_index + 1] != " Hit   " and player_board.board[row_index,element_index + 1] != " Miss  ":
                        potential_targets.append((row_index,element_index + 1))
                    if player_board.board[row_index + 1,element_index] != " Hit   " and player_board.board[row_index + 1,element_index] != " Miss  ":
                        potential_targets.append((row_index + 1,element_index))

                #right side
                elif row_index != 0 and row_index != 9 and element_index == 9:
                    if player_board.board[row_index - 1,element_index] != " Hit   " and player_board.board[row_index - 1,element_index] != " Miss  ":
                        potential_targets.append((row_index - 1,element_index))
                    if player_board.board[row_index,element_index - 1] != " Hit   " and player_board.board[row_index,element_index - 1] != " Miss  ":
                        potential_targets.append((row_index,element_index - 1))
                    if player_board.board[row_index + 1,element_index] != " Hit   " and player_board.board[row_index + 1,element_index] != " Miss  ":
                        potential_targets.append((row_index + 1,element_index))

                #somewhere in the middle for adjacent empty spaces
                elif row_index != 0 and row_index != 9 and element_index != 0 and element_index != 9:
                    if player_board.board[row_index - 1,element_index] != " Hit   " and player_board.board[row_index - 1,element_index] != " Miss  ":
                        potential_targets.append((row_index - 1,element_index))
                    if player_board.board[row_index,element_index - 1] != " Hit   " and player_board.board[row_index,element_index - 1] != " Miss  ":
                        potential_targets.append((row_index,element_index - 1))
                    if player_board.board[row_index + 1,element_index] != " Hit   " and player_board.board[row_index + 1,element_index] != " Miss  ":
                        potential_targets.append((row_index + 1,element_index))
                    if player_board.board[row_index,element_index + 1] != " Hit   " and player_board.board[row_index,element_index + 1] != " Miss  ":
                        potential_targets.append((row_index,element_index + 1))


    valid_coordinates = False
    if len(potential_targets) == 0: # if no hits with empty adjacent spaces are found, do random
        while(valid_coordinates == False):
            x_coordinate = random.choice([0,1,2,3,4,5,6,7,8,9])
            y_coordinate = random.choice([0,1,2,3,4,5,6,7,8,9])
            if player_board.board[y_coordinate,x_coordinate] != " Hit   " and player_board.board[y_coordinate,x_coordinate] != " Miss  ":
                valid_coordinates = True
    else:
        coordinates = random.choice(potential_targets) # if there are potential adjacent spaces, pick one at random
        x_coordinate = coordinates[1]
        y_coordinate = coordinates[0]


    return x_coordinate, y_coordinate