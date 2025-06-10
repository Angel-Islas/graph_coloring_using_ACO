import numpy as np
import random
import tkinter as tk
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AntColonyGraphColoring:
    def __init__(self, graph, num_ants, max_iterations, alpha, beta, evaporation_rate):
        self.graph = graph
        self.num_nodes = len(graph)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

        # Inicialización de la matriz de feromonas
        self.pheromone = None  # Se inicializará dinámicamente cuando se determine el número de colores

    def heuristic(self, node, color, solution):
        # La heurística evalúa cuántos vecinos tienen ya el color propuesto
        conflicts = sum(1 for neighbor in range(self.num_nodes) if self.graph[node][neighbor] == 1 and solution[neighbor] == color)
        return 1 / (1 + conflicts)

    def seleccion_ruleta(self, poblacion, fitness_vals):
        max_fit = sum(fitness_vals)
        probabilidades = [fit / max_fit for fit in fitness_vals]

        ruleta_acumulada = []
        acumulado = 0

        # Construir ruleta acumulada
        for prob in probabilidades:
            acumulado += prob
            ruleta_acumulada.append(acumulado)

        r = random.random()

        # Seleccionar nodo correspondiente
        for i, prob in enumerate(ruleta_acumulada):
            if r <= prob:
                return poblacion[i]

    def construct_solution(self, num_colors):
        solution = [-1] * self.num_nodes  # Inicializar la solución como no coloreada

        for node in range(self.num_nodes):
            probabilities = []
            for color in range(num_colors):
                if all(solution[neighbor] != color for neighbor in range(self.num_nodes) if self.graph[node][neighbor] == 1):
                    pheromone_value = self.pheromone[node][color] ** self.alpha
                    heuristic_value = self.heuristic(node, color, solution) ** self.beta
                    probabilities.append(pheromone_value * heuristic_value)
                else:
                    probabilities.append(0)

            if sum(probabilities) == 0:
                return None

            color_indices = list(range(num_colors))
            chosen_color = self.seleccion_ruleta(color_indices, probabilities)

            solution[node] = chosen_color

        return solution

    def update_pheromones(self, solutions, num_colors):
        # Evaporación
        self.pheromone *= (1 - self.evaporation_rate)

        # Actualización basada en soluciones
        for solution in solutions:
            if solution is not None:
                score = self.evaluate_solution(solution)
                if score > 0:
                    for node in range(self.num_nodes):
                        self.pheromone[node][solution[node]] += 1 / score

    def evaluate_solution(self, solution):
        # Evaluar la calidad de la solución según el número de conflictos
        conflicts = 0
        for node in range(self.num_nodes):
            for neighbor in range(self.num_nodes):
                if self.graph[node][neighbor] == 1 and solution[node] == solution[neighbor]:
                    conflicts += 1
        return conflicts // 2  # Cada conflicto se cuenta dos veces

    def find_min_colors(self):
        for num_colors in range(1, self.num_nodes + 1):
            self.pheromone = np.ones((self.num_nodes, num_colors))
            best_solution = None
            best_score = float('inf')

            for _ in range(self.max_iterations):
                solutions = []
                for _ in range(self.num_ants):
                    solution = self.construct_solution(num_colors)
                    if solution is not None:
                        solutions.append(solution)
                        score = self.evaluate_solution(solution)
                        if score < best_score:
                            best_solution = solution
                            best_score = score

                self.update_pheromones(solutions, num_colors)

                if best_score == 0:
                    return best_solution, num_colors

        return None, self.num_nodes

    def draw_graph(self, solution=None):
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.graph[i][j] == 1:
                    G.add_edge(i, j)

        pos = nx.spring_layout(G)
        plt.clf()
        if solution:
            colors = [solution[node] for node in G.nodes()]
        else:
            colors = ["lightgray"] * self.num_nodes

        nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow)
        return plt

# Interfaz gráfica con tkinter
def visualize_graph():
    def load_graph():
        global graph
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return

        with open(file_path, "r") as file:
            matrix = [list(map(int, line.split())) for line in file]

        graph = matrix
        plt_left = AntColonyGraphColoring(graph, num_ants, max_iterations, alpha, beta, evaporation_rate).draw_graph()
        canvas_left.figure = plt_left.gcf()
        canvas_left.draw()


    def apply_algorithm():
        if not graph:
            return

        aco = AntColonyGraphColoring(graph, num_ants, max_iterations, alpha, beta, evaporation_rate)
        best_solution, num_colors = aco.find_min_colors()

        if best_solution:
            plt_right = aco.draw_graph(best_solution)
            canvas_right.figure = plt_right.gcf()
            canvas_right.draw()

            info_label.config(text=f"Mejor solución: {best_solution}\nColores mínimos necesarios: {num_colors}")

    root = tk.Tk()
    root.title("Coloreado de Grafos")

    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_center = tk.Frame(root)
    frame_center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Grafo sin colorear
    plt.figure(figsize=(5, 5))
    canvas_left = FigureCanvasTkAgg(plt.gcf(), master=frame_left)
    canvas_left.draw()
    canvas_left.get_tk_widget().pack()

    # Botones y texto centrados
    load_button = tk.Button(frame_center, text="Cargar Grafo", command=load_graph)
    load_button.pack(pady=20)

    run_button = tk.Button(frame_center, text="Aplicar ACO", command=apply_algorithm)
    run_button.pack(pady=20)

    info_label = tk.Label(frame_center, text="", font=("Arial", 12), justify=tk.CENTER)
    info_label.pack(pady=20)

    # Grafo coloreado
    plt.figure(figsize=(5, 5))
    canvas_right = FigureCanvasTkAgg(plt.gcf(), master=frame_right)
    canvas_right.draw()
    canvas_right.get_tk_widget().pack()

    root.mainloop()

graph = []
num_ants = 20
max_iterations = 200
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5

visualize_graph()
