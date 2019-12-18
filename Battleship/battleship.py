import random
import numpy as np
import sys
import time

import battleshipAI as BS_AI

# By using the Python programming language, implement a Battleship board game that can be
# played against an AI. The rules are the following:

# The board is a 10x10 grid. There are two boards in the game which are updated on each
# turn: the one where the player’s ships are positioned and the one where the opponent
# positions their ships.

# Both players have the ships of following lengths to use: 2, 3, 3, 4, 5.

# Before the game starts, both players position their ships on their own board. The ships may
# be positioned vertically or horizontally and must not overlap.

# After this the player who starts is chosen randomly. The game proceeds in series of rounds.
# Each player in turn chooses a square they want to target on the opponent’s board, and the
# game shows if it is a hit or miss. If all the squares occupied by the ship are hit, the ship is
# sunk. The player who first sinks all the ships of the opponent is the winner.

# The AI may use an algorithm of your choosing, but it should be smarter than just randomly
# choosing a square.

# A graphical UI is not mandatory but is regarded as a bonus.

# Evaluation criteria:
# ● Design
# ● Structure
# ● Use of language features
# ● Clarity of implementation
# ● Error checking
# ● Comments
# ● Testability
# ● Documentation
# ● Usability

#''' '''
class Gameboard():
	''' Create a gaming board for a particular player'''

	def __init__(self,playerType):
		''' When Gameboard object is created it will have a 10x10 board full of "None" and a playerType (which can be "player" or "AI")'''
		self.board = np.full((10,10),None)
		self.playerType = playerType

	def place_ships(self):
		''' place ships for each player. AI will place randomly. 
		Player will choose where to put his/her ships.'''

		ship_list = [("ship_5 ",5),("ship_4 ",4),("ship_3A",3),("ship_3B",3),("ship_2 ",2)] # list of ships and how much space they take

		if self.playerType == "AI":
			#''' AI player will place ships randomly'''

			for ship in ship_list: # go through the ship types
				ship_name, ship_length = ship #get ships name and length

				while(True):
					location = (random.choice([0,1,2,3,4,5,6,7,8,9]),random.choice([0,1,2,3,4,5,6,7,8,9])) #randomize location
					direction = random.choice(["right","left","down","up"])								   #randomize direction
					if direction == "right" and location[0] + (ship_length - 1) < 10 and location[0] + (ship_length - 1) >= 0: #if the two ends of the ships are within bounds of the board, it may be placed, elsewise pick new place and coordinates
						break
					elif direction == "left" and location[0] - (ship_length - 1) < 10 and location[0] - (ship_length - 1) >= 0:
						break
					elif direction == "down" and location[1] + (ship_length - 1) < 10 and location[1] + (ship_length - 1) >= 0:
						break
					elif direction == "up" and location[1] - (ship_length - 1) < 10 and location[1] - (ship_length - 1) >= 0:
						break

	
				while(True):
					temp_coordinates = []
					for i in range(ship_length):

						# get the coordinates of every part of each ship
						if direction == "right": 
							x_coordinate = location[0] + i
							y_coordinate = location[1]
						elif direction == "left":
							x_coordinate = location[0] - i
							y_coordinate = location[1]
						elif direction == "down":
							x_coordinate = location[0]
							y_coordinate = location[1] + i
						elif direction == "up":
							x_coordinate = location[0]
							y_coordinate = location[1] - i


						if self.board[x_coordinate, y_coordinate] is None: # if the coordinates are empty (i.e. contain "None"), the add it to temporary coordinates
							temp_coordinates.append((x_coordinate,y_coordinate))

						else: # if the coordinates are not empty, then we need to select new coordinates and direction for the ship

							while(True):
								location = (random.choice([0,1,2,3,4,5,6,7,8,9]),random.choice([0,1,2,3,4,5,6,7,8,9]))
								direction = random.choice(["right","left","down","up"])
								if direction == "right" and location[0] + (ship_length - 1) < 10 and location[0] + (ship_length - 1) >= 0:
									break
								if direction == "left" and location[0] - (ship_length - 1) < 10 and location[0] - (ship_length - 1) >= 0:
									break
								if direction == "down" and location[1] + (ship_length - 1) < 10 and location[1] + (ship_length - 1) >= 0:
									break
								if direction == "up" and location[1] - (ship_length - 1) < 10 and location[1] - (ship_length - 1) >= 0:
									break


					if len(temp_coordinates) == ship_length: # if every coordinate that ship needs is empty
						for coordinates in temp_coordinates: # then place it on the actual board 
							self.board[coordinates] = ship_name
						break 
		

		elif self.playerType == "Player":
			#''' Human player must choose the starting location of each ship and which direction it points'''

			for ship in ship_list: # go through the ship list
				 
				ship_name, ship_length = ship # get name of each ship and size

				while(True):

					while(True):
					
						display_board(self.board, role='own') # display the current status of player board
						
						location = insert_location() #get target coordinates
						
						break


					while(True): #get the direction which the ship will be facing
						direction = input("Enter direction('right','left','down','up')")
						if direction.lower() in ['right','left','down','up']: #once direction is correct, continue
							break
						else:
							print("ERROR! Must enter valid direction")

					# this checks that the ship coordinates are within bounds
					if direction == "right" and location[1] + (ship_length - 1) < 10 and location[1] + (ship_length - 1) >= 0:
						break
					elif direction == "left" and location[1] - (ship_length - 1) < 10 and location[1] - (ship_length - 1) >= 0:
						break
					elif direction == "down" and location[0] + (ship_length - 1) < 10 and location[0] + (ship_length - 1) >= 0:
						break
					elif direction == "up" and location[0] - (ship_length - 1) < 10 and location[0] - (ship_length - 1) >= 0:
						break
					else:
						print("ERROR! Ship does not fit! Choose different coordinates and/or direction")

				while(True):
					#With proper coordinates and direction, try to place ship on the board
					temp_coordinates = []
					for i in range(ship_length):
						if direction == "right":
							x_coordinate = location[0] 
							y_coordinate = location[1] + i
						elif direction == "left":
							x_coordinate = location[0] 
							y_coordinate = location[1] - i
						elif direction == "down":
							x_coordinate = location[0] + i
							y_coordinate = location[1] 
						elif direction == "up":
							x_coordinate = location[0] - i
							y_coordinate = location[1]

						if (10 > x_coordinate and x_coordinate >= 0 and # if the given coordinates are within bounds and contain "None"
							10 > y_coordinate and y_coordinate >= 0 and						
							self.board[x_coordinate, y_coordinate] is None):
							

							temp_coordinates.append((x_coordinate,y_coordinate)) # add wanted coordinate to temporary coordinate list

						else:
							#if the ship is out of bounds or placed on top of another ship, try again 
							print("ERROR! Invalid placement and/or direction! Choose better coordinates and/or direction")

							while(True):

								while(True):

									display_board(self.board, role='own') #display the current status of player board

									location = insert_location() #get target coordinates

									break


								while(True):
									direction = input("Enter direction('right','left','down','up')") #get direction
									if direction.lower() in ['right','left','down','up']: # if direction valid, break
										break
									else:
										print("ERROR. Must enter valid direction")

								if direction == "right" and location[0] + (ship_length - 1) < 10 and location[0] + (ship_length - 1) >= 0:
									break
								elif direction == "left" and location[0] - (ship_length - 1) < 10 and location[0] - (ship_length - 1) >= 0:
									break
								elif direction == "down" and location[1] + (ship_length - 1) < 10 and location[1] + (ship_length - 1) >= 0:
									break
								elif direction == "up" and location[1] - (ship_length - 1) < 10 and location[1] - (ship_length - 1) >= 0:
									break
								else:
									print("ERROR. Invalid location and/or direction. Choose better location and/or direction")
							break


					if len(temp_coordinates) == ship_length:

						for coordinates in temp_coordinates:
							self.board[coordinates] = ship_name
						break




# convert y-coordinate letter to a number
def y_coordinate_converter(y_coordinate):
	if   y_coordinate == "A":
		return 0
	elif y_coordinate == "B":
		return 1
	elif y_coordinate == "C":
		return 2
	elif y_coordinate == "D":
		return 3
	elif y_coordinate == "E":
		return 4
	elif y_coordinate == "F":
		return 5
	elif y_coordinate == "G":
		return 6
	elif y_coordinate == "H":
		return 7
	elif y_coordinate == "I":
		return 8
	elif y_coordinate == "J":
		return 9


# get location coordinates
def insert_location():

	while(True):
		try:
			temp_x_coordinate = int(input("Enter x_coordinate [123456789(10)]: ")) 
			if temp_x_coordinate > 0 and temp_x_coordinate <= 10: #if we get integer and it is between 1 and 10
				break
		except ValueError: #if not integer
			print("Number must be a integer")

	while(True):
		temp_y_coordinate = str(input("Enter y_coordinate [ABCDEFGHIJ]: "))
		if len(temp_y_coordinate) == 1: # string must be 1 long
			if temp_y_coordinate in "ABCDEFGHIJ": # the 1 long string must be one of "ABCDEFGHIJ"
				break
			else:
				print("Invalid character. Must be one of [ABCDEFGHIJ]")
		else:
			print("Must be only singe character")

	
	location = [y_coordinate_converter(temp_y_coordinate),temp_x_coordinate - 1] # y_coordinate must be converted to number, x_coordinate can be obtained by reducing x_coordinate by one
	
	return location

#display game board. board is either players or AIs, role is "own" for displaying players own view of that board, "opponent" display what opponent sees of that particular board
def display_board(board, role):
	print("         1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   10  ") # top row
	letter_list = ["A","B","C","D","E","F","G","H","I","J"] #for the left side row
	for index,row in enumerate(board): # go through every row
		temp_str = '  '+letter_list[index]+'  '

		for element in row: # go through every element in every row
			if role == 'own': # if viewing your own board. You can see everything.
				if element is None: # if empty, represent it with "."
					temp_str = temp_str + "|   .   " 
				else:				# if not empty, print the contents as they are
					temp_str = temp_str + "|" + str(element)
			elif role == 'opponent': # if viewing your opponents board. Then you can only see hits and misses
				if element is None: # if empty, represent it with "."
					temp_str = temp_str + "|   .   "
				elif  "Hit" in element: # if hit, print "hit"
					temp_str = temp_str + "|  Hit  "
				elif "Miss" in element: # if miss, print "miss"
					temp_str = temp_str + "| Miss  "
				else:					# if neither hit or miss, print "."
					temp_str = temp_str + "|   .   "
			else:
				temp_str = temp_str + "|   .   "

		print(temp_str) #print the particular row


# attack particular coordinates on a board
def attack_coordinate(x_coordinate,y_coordinate , board):
	ship_list = ["ship_5 ","ship_4 ","ship_3A","ship_3B","ship_2 "] #list of ships. If these are on coordinates, then it is a hit

	valid_target = True # assume valid target

	if board.board[y_coordinate,x_coordinate] is None: 			# if the particular coordinates is empty (i.e. has "None"), then it is a miss
		print("MISS!")
		board.board[y_coordinate,x_coordinate] = " Miss  "
	elif board.board[y_coordinate,x_coordinate] in ship_list: 	# if the particular coordinates contains one of the ships, then it is a hit
		print("HIT!")
		board.board[y_coordinate,x_coordinate] = " Hit   "

	else: 														# if the particular coordinates contain neither "None" or one of the ships, then it must be either "Hit" or "Miss". This is a invalid coordinate
		print("ERROR! Coordinate already used. Pick another one")
		valid_target = False

	return board, valid_target


#initialize player_board
player_board = Gameboard("Player")

#player places ships on his/her board
player_board.place_ships()


#initialize AI_board
AI_board = Gameboard("AI")

#AI places ships on its board
AI_board.place_ships()



# check if the game has finished
def check_game_finished(player_board, AI_board):
	
	#go through player board and add every element, that is not "None", to a list
	unique_item_player_board = []
	for row in player_board.board:
		for element in row:
			if element is not None:
				unique_item_player_board.append(element)

	#go through AI board and add every element, that is not "None", to a list
	unique_item_AI_board = []
	for row in AI_board.board:
		for element in row:
			if element is not None:
				unique_item_AI_board.append(element)

	# Check each unique type elements in each list
	unique_item_player_board = np.unique(unique_item_player_board)
	unique_item_AI_board = np.unique(unique_item_AI_board)


	#remove hits and misses from both lists
	unique_item_player_board = unique_item_player_board[unique_item_player_board != " Hit   "]
	unique_item_player_board = unique_item_player_board[unique_item_player_board != " Miss  "]

	unique_item_AI_board = unique_item_AI_board[unique_item_AI_board != " Hit   "]
	unique_item_AI_board = unique_item_AI_board[unique_item_AI_board != " Miss  "]


	#if either AI or player has no longer any ships left on the unique list, then they are all destroyed (and covered by hits)
	game_finished = False
	if len(unique_item_player_board) == 0:
		print("AI WINS!")
		game_finished = True
	elif len(unique_item_AI_board) == 0:
		print("PLAYER WINS!")
		game_finished = True
	print("check_game_finished ENDS")
	return game_finished


turn = random.choice(['AI','Player']) #choose starting player

game_finished = False
turn_number = 1

print("\n\n\nThe game begins!")
while(game_finished == False):
	
	turn_number += 1
	print("\n\nturn_number:",turn_number // 2) #each player gets one turn, thus divide by two (rounded down)
	print("Whos turn:", turn)


	if turn == 'AI': #if AI Turn
		valid_target = False
		while(valid_target == False):

			x_coordinate, y_coordinate = BS_AI.AI_attack_coordinates(player_board) #AI selects coordinates
	
			player_board, valid_target = attack_coordinate(x_coordinate,y_coordinate, player_board) #tries to attack coordinates

		turn = 'Player'  #once AI makes valid move, change to human player

	elif turn == 'Player': #if player turn
		valid_target = False
		while(valid_target == False):
			print("Player board")
			display_board(player_board.board, role="own")
			print("AI board")
			display_board(AI_board.board, role="opponent")
			while(True):
				
				location = insert_location() #get target coordinates
				
				x_coordinate = location[1]
				y_coordinate = location[0]
				break

			AI_board, valid_target = attack_coordinate(x_coordinate, y_coordinate, AI_board) # check if valid attack is possible, if so, register it as attack

		turn = 'AI' #once human makes valid move, change to AI player


	game_finished = check_game_finished(player_board, AI_board) #check if games has ended


