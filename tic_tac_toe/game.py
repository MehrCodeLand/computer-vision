import numpy as np
import cv2
import os
import random
import time

class TicTacToe:
    def __init__(self):
        self.board = np.full((3, 3), 'W', dtype=str)
        self.image_size = 300
        self.cell_size = self.image_size // 3
        self.result_folder = "result"
        self.board_image_path = "image/tic-tac-toe-board.png"
        self.current_image = None
        
        # Create result folder if it doesn't exist
        # if not os.path.exists(self.result_folder):
        #     os.makedirs(self.result_folder)
            
        self.initialize_board_image()
    
    def initialize_board_image(self):
        if os.path.exists(self.board_image_path):
            self.current_image = cv2.imread(self.board_image_path)
        else:
            # Create a blank white image
            self.current_image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
            # Draw the grid lines
            cv2.line(self.current_image, (self.cell_size, 0), (self.cell_size, self.image_size), (0, 0, 0), 2)
            cv2.line(self.current_image, (2 * self.cell_size, 0), (2 * self.cell_size, self.image_size), (0, 0, 0), 2)
            cv2.line(self.current_image, (0, self.cell_size), (self.image_size, self.cell_size), (0, 0, 0), 2)
            cv2.line(self.current_image, (0, 2 * self.cell_size), (self.image_size, 2 * self.cell_size), (0, 0, 0), 2)
            # Save the created board
            cv2.imwrite(self.board_image_path, self.current_image)
        
        # Ensure the image is 300x300
        self.current_image = cv2.resize(self.current_image, (self.image_size, self.image_size))
    
    def get_empty_cells(self):
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 'W':
                    empty_cells.append((i, j))
        return empty_cells
    
    def make_move(self, player):
        empty_cells = self.get_empty_cells()
        if not empty_cells:
            return False 
        
        # Randomly select an empty cell
        row, col = random.choice(empty_cells)
        self.board[row, col] = player
        
        # Update the image
        self.update_image(row, col, player)
        
        return True
    
    def update_image(self, row, col, player):
        # Calculate the center position of the cell
        center_x = col * self.cell_size + self.cell_size // 2
        center_y = row * self.cell_size + self.cell_size // 2
        
        # Cell top-left corner
        start_x = col * self.cell_size
        start_y = row * self.cell_size
        
        # Create a cell image
        cell_image = self.current_image[start_y:start_y+self.cell_size, start_x:start_x+self.cell_size].copy()
        
        # Draw X or O
        if player == 'X':
            # Draw X
            margin = int(self.cell_size * 0.2)
            cv2.line(cell_image, (margin, margin), (self.cell_size - margin, self.cell_size - margin), (0, 0, 255), 3)
            cv2.line(cell_image, (self.cell_size - margin, margin), (margin, self.cell_size - margin), (0, 0, 255), 3)
        else:  # O
            # Draw O
            radius = int(self.cell_size * 0.3)
            cv2.circle(cell_image, (self.cell_size // 2, self.cell_size // 2), radius, (0, 255, 0), 3)
        
        # Replace the cell in the main image
        self.current_image[start_y:start_y+self.cell_size, start_x:start_x+self.cell_size] = cell_image
        
        timestamp = int(time.time())
        
        file_path = f'images/result/move_{timestamp}.png'
        cv2.imwrite(file_path, self.current_image)
        
    def check_winner(self):
        # Check rows
        for i in range(3):
            if self.board[i, 0] != 'W' and self.board[i, 0] == self.board[i, 1] == self.board[i, 2]:
                return self.board[i, 0]
        
        # Check columns
        for i in range(3):
            if self.board[0, i] != 'W' and self.board[0, i] == self.board[1, i] == self.board[2, i]:
                return self.board[0, i]
        
        # Check diagonals
        if self.board[0, 0] != 'W' and self.board[0, 0] == self.board[1, 1] == self.board[2, 2]:
            return self.board[0, 0]
        if self.board[0, 2] != 'W' and self.board[0, 2] == self.board[1, 1] == self.board[2, 0]:
            return self.board[0, 2]
        
        # Check for tie
        if 'W' not in self.board:
            return 'Tie'
        
        # Game is still ongoing
        return None
    
    def print_board(self):
        print("Current Board State:")
        print(self.board)
        print()

    def play_game(self):
        players = ['X', 'O']
        current_player_index = 0
        
        print("Starting a new Tic Tac Toe game with random moves!")
        self.print_board()
        
        while True:
            current_player = players[current_player_index]
            print(f"Player {current_player}'s turn...")
            
            # Make a move
            self.make_move(current_player)
            self.print_board()
            
            # Check for a winner or tie
            result = self.check_winner()
            if result:
                if result == 'Tie':
                    print("Game ended in a tie!")
                else:
                    print(f"Player {result} wins!")
                break
            
            # Switch player
            current_player_index = (current_player_index + 1) % 2
            
            # Optional: Add a small delay to see the progress better
            time.sleep(0.5)

if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()