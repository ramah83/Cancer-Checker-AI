import pygame
import sys
import math
import random
import time
from typing import List, Dict, Tuple, Optional, Any

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
GRID_SIZE = 15
CELL_SIZE = 35
GRID_WIDTH = GRID_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_SIZE * CELL_SIZE
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 80
CONTROL_PANEL_WIDTH = 300

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 255, 0)
BACKGROUND_COLOR = (245, 245, 245)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Delivery Robot - A* Pathfinding")
clock = pygame.time.Clock()

font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)
subtitle_font = pygame.font.SysFont('Arial', 18)

class Cell:
    def __init__(self, x: int, y: int, obstacle: bool = False):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.obstacle = obstacle
        self.parent = None

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int], hover_color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.is_active = False
        
    def draw(self, surface: pygame.Surface):
        color = self.hover_color if self.is_hovered or self.is_active else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, BLACK, self.rect, 1, border_radius=4)
        text_surface = font.render(self.text, True, WHITE if self.is_active else BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def check_hover(self, pos: Tuple[int, int]) -> bool:
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            return True
        return False

class ToggleButton(Button):
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int], hover_color: Tuple[int, int, int], active_color: Tuple[int, int, int]):
        super().__init__(x, y, width, height, text, color, hover_color)
        self.active_color = active_color
        
    def draw(self, surface: pygame.Surface):
        if self.is_active:
            color = self.active_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, BLACK, self.rect, 1, border_radius=4)
        text_color = WHITE if self.is_active else BLACK
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def toggle(self) -> bool:
        self.is_active = not self.is_active
        return self.is_active

class Checkbox:
    def __init__(self, x: int, y: int, size: int, text: str, checked: bool = False):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.checked = checked
        self.size = size
        
    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 1)
        if self.checked:
            inner_rect = pygame.Rect(self.rect.x + 3, self.rect.y + 3, self.rect.width - 6, self.rect.height - 6)
            pygame.draw.rect(surface, BLACK, inner_rect)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        surface.blit(text_surface, text_rect)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                return True
        return False

class RadioButton:
    def __init__(self, x: int, y: int, size: int, text: str, group: str, checked: bool = False):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.checked = checked
        self.size = size
        self.group = group
        
    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, WHITE, self.rect.center, self.size // 2)
        pygame.draw.circle(surface, BLACK, self.rect.center, self.size // 2, 1)
        if self.checked:
            pygame.draw.circle(surface, BLACK, self.rect.center, self.size // 3)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        surface.blit(text_surface, text_rect)
        
    def handle_event(self, event: pygame.event.Event, radio_buttons: List['RadioButton']) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                for button in radio_buttons:
                    if button.group == self.group:
                        button.checked = False
                self.checked = True
                return True
        return False

class Slider:
    def __init__(self, x: int, y: int, width: int, height: int, min_val: int, max_val: int, value: int, text: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.text = text
        self.handle_rect = pygame.Rect(0, 0, 10, height + 6)
        self.update_handle_position()
        self.dragging = False
        
    def update_handle_position(self):
        value_range = self.max_val - self.min_val
        position_ratio = (self.value - self.min_val) / value_range
        handle_x = self.rect.x + int(position_ratio * self.rect.width) - self.handle_rect.width // 2
        self.handle_rect.x = handle_x
        self.handle_rect.centery = self.rect.centery
        
    def draw(self, surface: pygame.Surface):
        text_surface = font.render(f"{self.text}: {self.value}", True, BLACK)
        text_rect = text_surface.get_rect(bottomleft=(self.rect.left, self.rect.top - 5))
        surface.blit(text_surface, text_rect)
        pygame.draw.rect(surface, GRAY, self.rect, border_radius=3)
        pygame.draw.rect(surface, BLACK, self.handle_rect, border_radius=5)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = event.pos[0]
            relative_x = max(0, min(mouse_x - self.rect.x, self.rect.width))
            value_range = self.max_val - self.min_val
            self.value = self.min_val + int((relative_x / self.rect.width) * value_range)
            self.update_handle_position()
            return True
        return False

class SmartDeliveryRobot:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.grid = []
        self.start_pos = {"x": 2, "y": 2}
        self.end_pos = {"x": self.grid_size - 3, "y": self.grid_size - 3}
        self.robot_pos = {"x": 2, "y": 2}
        self.path = []
        self.explored = []
        self.tool_mode = "obstacle"
        self.algorithm = "astar"
        self.simulation_speed = 50
        self.show_explored = True
        self.is_running = False
        self.animation_timer = 0
        self.path_index = 0
        self.stats = {
            "path_length": 0,
            "nodes_explored": 0,
            "execution_time": 0
        }
        self.init_ui()
        self.initialize_grid()
        
    def init_ui(self):
        button_width = 80
        button_height = 35
        button_spacing = 10
        button_x = GRID_OFFSET_X + GRID_WIDTH + 50
        button_y = GRID_OFFSET_Y
        self.start_button = ToggleButton(
            button_x, button_y, button_width, button_height, 
            "Start", LIGHT_GRAY, GRAY, GREEN
        )
        self.end_button = ToggleButton(
            button_x + button_width + button_spacing, button_y, button_width, button_height, 
            "End", LIGHT_GRAY, GRAY, RED
        )
        self.obstacle_button = ToggleButton(
            button_x, button_y + button_height + button_spacing, button_width * 2 + button_spacing, button_height, 
            "Obstacle", LIGHT_GRAY, GRAY, DARK_GRAY
        )
        self.obstacle_button.is_active = True
        radio_y = button_y + 2 * (button_height + button_spacing) + 20
        self.algorithm_label_pos = (button_x, radio_y)
        self.astar_radio = RadioButton(
            button_x, radio_y + 30, 16, "A* Algorithm", "algorithm", True
        )
        self.dijkstra_radio = RadioButton(
            button_x, radio_y + 60, 16, "Dijkstra's Algorithm", "algorithm", False
        )
        self.radio_buttons = [self.astar_radio, self.dijkstra_radio]
        slider_y = radio_y + 100
        self.grid_size_slider = Slider(
            button_x, slider_y, 200, 10, 5, 25, self.grid_size, "Grid Size"
        )
        speed_slider_y = slider_y + 60
        self.speed_slider = Slider(
            button_x, speed_slider_y, 200, 10, 0, 100, self.simulation_speed, "Simulation Speed"
        )
        checkbox_y = speed_slider_y + 60
        self.show_explored_checkbox = Checkbox(
            button_x, checkbox_y, 16, "Show Explored Nodes", True
        )
        control_button_y = checkbox_y + 50
        self.start_sim_button = Button(
            button_x, control_button_y, 95, 35, "Start", (0, 180, 0), (0, 150, 0)
        )
        self.stop_sim_button = Button(
            button_x, control_button_y, 95, 35, "Stop", (180, 0, 0), (150, 0, 0)
        )
        self.reset_button = Button(
            button_x + 105, control_button_y, 95, 35, "Reset", LIGHT_GRAY, GRAY
        )
        self.regenerate_button = Button(
            button_x, control_button_y + 45, 200, 35, "Regenerate Grid", LIGHT_GRAY, GRAY
        )
        
    def initialize_grid(self):
        self.grid = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                row.append(Cell(x, y))
            self.grid.append(row)
        for _ in range(int(self.grid_size * self.grid_size * 0.2)):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x == self.start_pos["x"] and y == self.start_pos["y"]) or \
               (x == self.end_pos["x"] and y == self.end_pos["y"]):
                continue
            self.grid[y][x].obstacle = True
        self.path = []
        self.explored = []
        self.robot_pos = self.start_pos.copy()
        self.path_index = 0
        self.is_running = False
        self.stats = {
            "path_length": 0,
            "nodes_explored": 0,
            "execution_time": 0
        }
        
    def handle_cell_click(self, x: int, y: int):
        if self.is_running:
            return
        if self.tool_mode == "start":
            if not self.grid[y][x].obstacle and not (x == self.end_pos["x"] and y == self.end_pos["y"]):
                self.start_pos = {"x": x, "y": y}
                self.robot_pos = {"x": x, "y": y}
        elif self.tool_mode == "end":
            if not self.grid[y][x].obstacle and not (x == self.start_pos["x"] and y == self.start_pos["y"]):
                self.end_pos = {"x": x, "y": y}
        elif self.tool_mode == "obstacle":
            is_start = x == self.start_pos["x"] and y == self.start_pos["y"]
            is_end = x == self.end_pos["x"] and y == self.end_pos["y"]
            if not is_start and not is_end:
                self.grid[y][x].obstacle = not self.grid[y][x].obstacle
        self.path = []
        self.explored = []
        
    def start_simulation(self):
        if self.is_running:
            return
        self.is_running = True
        self.path_index = 0
        self.robot_pos = self.start_pos.copy()
        start_time = time.time()
        result = self.find_path(self.grid, self.start_pos, self.end_pos, self.algorithm)
        end_time = time.time()
        self.path = result["path"]
        self.explored = result["explored"]
        self.stats = {
            "path_length": len(self.path),
            "nodes_explored": len(self.explored),
            "execution_time": (end_time - start_time) * 1000
        }
        
    def stop_simulation(self):
        self.is_running = False
        
    def reset_simulation(self):
        self.stop_simulation()
        self.robot_pos = self.start_pos.copy()
        self.path = []
        self.explored = []
        self.stats = {
            "path_length": 0,
            "nodes_explored": 0,
            "execution_time": 0
        }
        
    def find_path(self, grid: List[List[Cell]], start: Dict[str, int], end: Dict[str, int], algorithm: str) -> Dict:
        grid_copy = []
        for y in range(len(grid)):
            row = []
            for x in range(len(grid[0])):
                cell = Cell(x, y, grid[y][x].obstacle)
                row.append(cell)
            grid_copy.append(row)
        open_set = []
        closed_set = []
        explored = []
        start_node = grid_copy[start["y"]][start["x"]]
        open_set.append(start_node)
        while open_set:
            current_index = 0
            for i in range(len(open_set)):
                if open_set[i].f < open_set[current_index].f:
                    current_index = i
            current = open_set[current_index]
            if current.x == end["x"] and current.y == end["y"]:
                path = []
                temp = current
                while temp:
                    path.append({"x": temp.x, "y": temp.y})
                    temp = temp.parent
                return {
                    "path": list(reversed(path)),
                    "explored": explored
                }
            open_set.pop(current_index)
            closed_set.append(current)
            explored.append({"x": current.x, "y": current.y})
            neighbors = []
            x, y = current.x, current.y
            if y > 0:
                neighbors.append(grid_copy[y - 1][x])
            if x < self.grid_size - 1:
                neighbors.append(grid_copy[y][x + 1])
            if y < self.grid_size - 1:
                neighbors.append(grid_copy[y + 1][x])
            if x > 0:
                neighbors.append(grid_copy[y][x - 1])
            for neighbor in neighbors:
                if neighbor.obstacle or any(cell.x == neighbor.x and cell.y == neighbor.y for cell in closed_set):
                    continue
                is_diagonal = neighbor.x != current.x and neighbor.y != current.y
                movement_cost = 1.414 if is_diagonal else 1
                tentative_g = current.g + movement_cost
                in_open_set = any(cell.x == neighbor.x and cell.y == neighbor.y for cell in open_set)
                if not in_open_set or tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    if algorithm == "astar":
                        dx = abs(neighbor.x - end["x"])
                        dy = abs(neighbor.y - end["y"])
                        neighbor.h = math.sqrt(dx * dx + dy * dy)
                    else:
                        neighbor.h = 0
                    neighbor.f = neighbor.g + neighbor.h
                    if not in_open_set:
                        open_set.append(neighbor)
        return {
            "path": [],
            "explored": explored
        }
        
    def update(self, dt: float):
        if self.is_running and self.path:
            self.animation_timer += dt
            delay = max(0.5 - (self.simulation_speed / 100) * 0.45, 0.05)
            if self.animation_timer >= delay:
                self.animation_timer = 0
                if self.path_index < len(self.path):
                    self.robot_pos = self.path[self.path_index]
                    self.path_index += 1
                else:
                    self.is_running = False
        mouse_pos = pygame.mouse.get_pos()
        self.start_button.check_hover(mouse_pos)
        self.end_button.check_hover(mouse_pos)
        self.obstacle_button.check_hover(mouse_pos)
        self.start_sim_button.check_hover(mouse_pos)
        self.reset_button.check_hover(mouse_pos)
        self.regenerate_button.check_hover(mouse_pos)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                grid_rect = pygame.Rect(GRID_OFFSET_X, GRID_OFFSET_Y, GRID_WIDTH, GRID_HEIGHT)
                if grid_rect.collidepoint(mouse_pos):
                    grid_x = (mouse_pos[0] - GRID_OFFSET_X) // self.cell_size
                    grid_y = (mouse_pos[1] - GRID_OFFSET_Y) // self.cell_size
                    if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                        self.handle_cell_click(grid_x, grid_y)
            if self.start_button.handle_event(event):
                self.start_button.is_active = True
                self.end_button.is_active = False
                self.obstacle_button.is_active = False
                self.tool_mode = "start"
            if self.end_button.handle_event(event):
                self.start_button.is_active = False
                self.end_button.is_active = True
                self.obstacle_button.is_active = False
                self.tool_mode = "end"
            if self.obstacle_button.handle_event(event):
                self.start_button.is_active = False
                self.end_button.is_active = False
                self.obstacle_button.is_active = True
                self.tool_mode = "obstacle"
            for button in self.radio_buttons:
                if button.handle_event(event, self.radio_buttons):
                    if button == self.astar_radio and button.checked:
                        self.algorithm = "astar"
                    elif button == self.dijkstra_radio and button.checked:
                        self.algorithm = "dijkstra"
            if self.show_explored_checkbox.handle_event(event):
                self.show_explored = self.show_explored_checkbox.checked
            if self.grid_size_slider.handle_event(event):
                new_size = self.grid_size_slider.value
                if new_size != self.grid_size:
                    self.grid_size = new_size
                    self.end_pos = {"x": self.grid_size - 3, "y": self.grid_size - 3}
                    self.initialize_grid()
            self.speed_slider.handle_event(event)
            self.simulation_speed = self.speed_slider.value
            if not self.is_running and self.start_sim_button.handle_event(event):
                self.start_simulation()
            if self.is_running and self.stop_sim_button.handle_event(event):
                self.stop_simulation()
            if self.reset_button.handle_event(event):
                self.reset_simulation()
            if self.regenerate_button.handle_event(event):
                self.initialize_grid()
                
    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        title_text = title_font.render("Smart Delivery Robot", True, BLACK)
        subtitle_text = subtitle_font.render("Autonomous Indoor Navigation using A* Pathfinding Algorithm", True, DARK_GRAY)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 10))
        screen.blit(subtitle_text, (SCREEN_WIDTH // 2 - subtitle_text.get_width() // 2, 40))
        grid_rect = pygame.Rect(GRID_OFFSET_X - 1, GRID_OFFSET_Y - 1, GRID_WIDTH + 2, GRID_HEIGHT + 2)
        pygame.draw.rect(screen, BLACK, grid_rect, 1)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_rect = pygame.Rect(
                    GRID_OFFSET_X + x * self.cell_size,
                    GRID_OFFSET_Y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                cell_color = WHITE
                if self.show_explored and any(pos["x"] == x and pos["y"] == y for pos in self.explored):
                    cell_color = LIGHT_BLUE
                if any(pos["x"] == x and pos["y"] == y for pos in self.path):
                    cell_color = YELLOW
                if self.grid[y][x].obstacle:
                    cell_color = DARK_GRAY
                pygame.draw.rect(screen, cell_color, cell_rect)
                pygame.draw.rect(screen, GRAY, cell_rect, 1)
        start_x, start_y = self.start_pos["x"], self.start_pos["y"]
        start_center = (
            GRID_OFFSET_X + start_x * self.cell_size + self.cell_size // 2,
            GRID_OFFSET_Y + start_y * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(screen, GREEN, start_center, self.cell_size // 2 - 2)
        end_x, end_y = self.end_pos["x"], self.end_pos["y"]
        end_center = (
            GRID_OFFSET_X + end_x * self.cell_size + self.cell_size // 2,
            GRID_OFFSET_Y + end_y * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(screen, RED, end_center, self.cell_size // 2 - 2)
        if self.is_running or self.path:
            robot_x, robot_y = self.robot_pos["x"], self.robot_pos["y"]
            robot_center = (
                GRID_OFFSET_X + robot_x * self.cell_size + self.cell_size // 2,
                GRID_OFFSET_Y + robot_y * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(screen, BLUE, robot_center, self.cell_size // 2 - 2)
        self.start_button.draw(screen)
        self.end_button.draw(screen)
        self.obstacle_button.draw(screen)
        algorithm_label = subtitle_font.render("Algorithm", True, BLACK)
        screen.blit(algorithm_label, self.algorithm_label_pos)
        self.astar_radio.draw(screen)
        self.dijkstra_radio.draw(screen)
        self.grid_size_slider.draw(screen)
        self.speed_slider.draw(screen)
        self.show_explored_checkbox.draw(screen)
        if not self.is_running:
            self.start_sim_button.draw(screen)
        else:
            self.stop_sim_button.draw(screen)
        self.reset_button.draw(screen)
        self.regenerate_button.draw(screen)
        stats_x = GRID_OFFSET_X + GRID_WIDTH + 50
        stats_y = SCREEN_HEIGHT - 120
        stats_rect = pygame.Rect(stats_x - 10, stats_y - 10, 220, 100)
        pygame.draw.rect(screen, WHITE, stats_rect, border_radius=5)
        pygame.draw.rect(screen, GRAY, stats_rect, 1, border_radius=5)
        stats_title = subtitle_font.render("Statistics", True, BLACK)
        screen.blit(stats_title, (stats_x, stats_y))
        path_length_text = font.render(f"Path Length: {self.stats['path_length']}", True, BLACK)
        nodes_explored_text = font.render(f"Nodes Explored: {self.stats['nodes_explored']}", True, BLACK)
        execution_time_text = font.render(f"Execution Time: {self.stats['execution_time']:.2f} ms", True, BLACK)
        screen.blit(path_length_text, (stats_x, stats_y + 30))
        screen.blit(nodes_explored_text, (stats_x, stats_y + 50))
        screen.blit(execution_time_text, (stats_x, stats_y + 70))
        pygame.display.flip()
        pygame.display.flip()
        
    def run(self):
        last_time = time.time()
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            self.handle_events()
            self.update(dt)
            self.draw()
            clock.tick(60)

if __name__ == "__main__":
    app = SmartDeliveryRobot()
    app.run()