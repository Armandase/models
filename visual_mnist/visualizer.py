import pygame
import numpy as np
import os
import math

os.environ["KERAS_BACKEND"] = "torch"
import keras

NB_COLS = 28
NB_ROWS = 28
WINDOW_SIZE = 1100
FOOTER_HEIGHT = 100
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)

def center_image(img):
    total_mass = img.sum()
    if total_mass == 0:
        return img
    
    grid_x, grid_y = np.meshgrid(np.arange(NB_COLS), np.arange(NB_ROWS))
    x_center = (grid_x * img).sum() / total_mass
    y_center = (grid_y * img).sum() / total_mass
    
    dx = NB_COLS / 2 - x_center
    dy = NB_ROWS / 2 - y_center
    
    idx = int(round(dx))
    idy = int(round(dy))
    
    new_img = np.zeros_like(img)
    
    for r in range(NB_ROWS):
        for c in range(NB_COLS):
            src_r = r - idy
            src_c = c - idx
            if 0 <= src_r < NB_ROWS and 0 <= src_c < NB_COLS:
                new_img[r, c] = img[src_r, src_c]
                
    return new_img

def process_input(input, model):
    input = center_image(input)
    input = input.reshape(1, NB_ROWS, NB_COLS, 1)
    prediction = model.predict(input)
    res = prediction.argmax()
    print(res)
    return res

def draw_board(screen):
    screen.fill(GRAY)
    cell_width = screen.get_width() / NB_COLS
    cell_height = (screen.get_height() - FOOTER_HEIGHT) / NB_ROWS
    
    # Draw footer background
    pygame.draw.rect(screen, (255, 255, 255), (0, screen.get_height() - FOOTER_HEIGHT, screen.get_width(), FOOTER_HEIGHT))

    for i in range(NB_COLS + 1):
        pygame.draw.line(screen, BLACK, (i * cell_width, 0), (i * cell_width, screen.get_height() - FOOTER_HEIGHT), 1)

    for j in range(NB_ROWS + 1):
        pygame.draw.line(screen, BLACK, (0, j * cell_height), (screen.get_width(), j * cell_height), 1)

def draw_block(screen, col, line, intensity):
    cell_width = screen.get_width() / NB_COLS
    cell_height = (screen.get_height() - FOOTER_HEIGHT) / NB_ROWS

    x = col * cell_width
    y = line * cell_height
    # Clamp to avoid rounding issues when values slightly exceed [0, 1]
    intensity = max(0.0, min(1.0, intensity))
    # High intensity should appear darker on the canvas (draw to black)
    color_value = int((1.0 - intensity) * 255)
    color = (color_value, color_value, color_value)
    pygame.draw.rect(screen, color, (x, y, cell_width, cell_height))

# bloc autour de la position de la souris en fonction de la position  actuel dans un rayon de 1 cellule
def get_block_around(pos, screen):
    cell_width = screen.get_width() / NB_COLS
    cell_height = (screen.get_height() - FOOTER_HEIGHT) / NB_ROWS

    x, y = pos
    center_col = int(x // cell_width)
    center_row = int(y // cell_height)

    neighbors = []
    for col in range(max(0, center_col - 1), min(NB_COLS, center_col + 2)):
        cell_center_x = (col + 0.5) * cell_width
        dx = abs(cell_center_x - x)
        for row in range(max(0, center_row - 1), min(NB_ROWS, center_row + 2)):
            cell_center_y = (row + 0.5) * cell_height
            dy = abs(cell_center_y - y)
            distance = math.hypot(dx, dy) / cell_width
            
            if row == center_row and col == center_col:
                value = 1.0
            else:
                value = max(0.0, 1.0 - distance * 0.5)
            
            neighbors.append(((row, col), value))

    return neighbors

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + FOOTER_HEIGHT))
    pygame.display.set_caption("MNIST Visualizer")
    font = pygame.font.Font(None, 74)
    model = keras.models.load_model('models/model.h5')
    running = True
    start = None
    drawing = False
    end = None
    input = np.zeros((NB_ROWS, NB_COLS), dtype=np.float32)
    last_prediction = None
    draw_board(screen)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q)):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                start = event.pos
                drawing = True
            elif event.type == pygame.MOUSEMOTION and drawing:
                end = event.pos
                
                dist = math.hypot(end[0] - start[0], end[1] - start[1])
                steps = int(dist)
                
                for i in range(steps + 1):
                    if steps == 0:
                        t = 1.0
                    else:
                        t = i / steps
                        
                    current_pos = (
                        start[0] + (end[0] - start[0]) * t,
                        start[1] + (end[1] - start[1]) * t
                    )

                    blocks = get_block_around(current_pos, screen)
                    for (row, col), value in blocks:
                        new_value = max(input[row, col], value)
                        input[row, col] = new_value
                        draw_block(screen, col, row, new_value)
                start = end
            elif event.type == pygame.MOUSEBUTTONUP:
                last_prediction = process_input(input, model)
                drawing = False
                draw_board(screen)
                input = np.zeros((NB_ROWS, NB_COLS), dtype=np.float32)

        if last_prediction is not None:
            pygame.draw.rect(screen, (255, 255, 255), (0, screen.get_height() - FOOTER_HEIGHT, screen.get_width(), FOOTER_HEIGHT))
            text = font.render(f"Prediction: {last_prediction}", True, BLACK)
            text_rect = text.get_rect(center=(screen.get_width()/2, screen.get_height() - FOOTER_HEIGHT/2))
            screen.blit(text, text_rect)

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()