import tkinter as tk
from typing import List
import numpy as np
import math



BACKGROUNG_COLOR                = "#111111"

NEURON_DISABLE_COLOR            = "#424242"
NEURON_ENABLE_COLOR             = "#991774"
NEURON_OUTLINE_DISABLE_COLOR    = NEURON_DISABLE_COLOR
NEURON_OUTLINE_ENABLE_COLOR     = NEURON_ENABLE_COLOR

MULTI_NEURON_DISABLE_COLOR          = "#111111"
MULTI_NEURON_ENABLE_COLOR           = "#991774"
MULTI_NEURON_DISABLE_OUTLINE_COLOR  = "#dedede"
MULTI_NEURON_ENABLE_OUTLINE_COLOR   = MULTI_NEURON_DISABLE_COLOR

LINE_COLOR      = "#802A67"

ERROR_COLOR     = "#BD3636"

# --------------------- DELAY -----------------------
DELAY = 40
# ---------------------------------------------------


# ------------------ NEURON SIZE --------------------
CIRCLE_RADIUS = 18
# -----------------  WEIGHT SIZE --------------------
WEIGHT_SIZE = 12
# ---------------------------------------------------


NEURONS_MARGIN_Y = CIRCLE_RADIUS
NEURONS_MARGIN_X = CIRCLE_RADIUS*1.5

root = tk.Tk()
root.title("Neural Network")
root.attributes('-fullscreen', True)
root.bind("<Escape>", lambda e: root.destroy())

SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()

MAX_NEURONS = int(SCREEN_H / (CIRCLE_RADIUS*2+NEURONS_MARGIN_Y))
MAX_NEURONS = MAX_NEURONS - (1 if MAX_NEURONS % 2 == 0 else 0)

MAX_WEIGHTS = int(SCREEN_W/4/WEIGHT_SIZE)


canvas = tk.Canvas(root, width=SCREEN_W, height=SCREEN_H, bg=BACKGROUNG_COLOR)
canvas.pack()


class Neuron:
    def __init__(
            self,
            input_size: int, 
            lr : float, 
            x : int, 
            y : int, 
            entity_id : int, 
            enable_color : str = NEURON_ENABLE_COLOR, 
            disable_color : str = NEURON_DISABLE_COLOR,
            outline_enable_color : str = NEURON_OUTLINE_ENABLE_COLOR,
            outline_disable_color : str = NEURON_OUTLINE_DISABLE_COLOR
        ):
        
        limit = np.sqrt(1 / input_size)
        self.w = np.random.uniform(-limit, limit, input_size)

        self.b = np.random.randn() * 0.1
        self.lr = lr
        self.entity_id = entity_id
        self.enable_color = enable_color
        self.disable_color = disable_color
        self.outline_enable_color = outline_enable_color
        self.outline_disable_color = outline_disable_color

        self.X = x
        self.Y = y

    def predict(self, x):
        s : float = 0
        for i in range(len(self.w)):
            s += x[i] * self.w[i]
        s += self.b
        y = self.active(s)
        return y

    def active(self, s : float):
        return 1/(1 + math.e**-s)

    def fit(self, x, t=None, delta = None, ids_matrix=[], scaler=100, draw = False):
        if(delta is None):
            y = self.predict(x)
            delta = (y-t) * y*(1-y)

        self.b = self.b - self.lr * (delta)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.lr * (delta * x[i])
        

        if draw:
            for i in range(len(x)):
                l = int(i/len(ids_matrix[0]))
                c = i%len(ids_matrix[0])
                id = ids_matrix[l][c]
                
                w_min = np.min(self.w)
                w_max = np.max(self.w)

                intensity = int((self.w[i]-w_min)/(w_max-w_min) * scaler)
                canvas.itemconfig(
                    id, 
                    fill = f"#{intensity if self.w[i] < 0 else 0:02x}{intensity if self.w[i] > 0 else 0:02x}{0:02x}"
                )
            canvas.update()
        
        return delta

class Network:
    def __init__(self, outputs, features, lr = 0.1, neuron_layers = [], line_limit = None, scaler = 100):
        self.layers : List[List[Neuron]] = []
        self.lr = lr
        self.outputs = outputs
        self.features = features # quantidade de features do dataset
        self.line_limit = line_limit
        self.scaler = scaler
        self.w_matrix = []
        self.x_matrix = []

        nLs = neuron_layers # Neuronios por camada

        nLs.append(self.outputs)

        nO = self.features
        for i in range(len(nLs)):
            
            neurons = []
            x = (SCREEN_W/4 - ((len(nLs))*(CIRCLE_RADIUS*2+NEURONS_MARGIN_X))/2) + (i*(CIRCLE_RADIUS*2+NEURONS_MARGIN_X))

            multi_element_id = None
            if(nLs[i] >= MAX_NEURONS):
                multi_element_id = canvas.create_oval(
                    x - CIRCLE_RADIUS,
                    SCREEN_H/2 - CIRCLE_RADIUS,
                    x + CIRCLE_RADIUS,
                    SCREEN_H/2 + CIRCLE_RADIUS,
                    fill=MULTI_NEURON_DISABLE_COLOR,
                    outline="#dedede"
                )

            m_neurons = min(MAX_NEURONS, nLs[i])
            y_mid = (SCREEN_H/2 - ((m_neurons-1)*(CIRCLE_RADIUS*2+NEURONS_MARGIN_Y))/2)
            step = 0
            passed = False
            for j in range(nLs[i]):
                enable_color = NEURON_ENABLE_COLOR
                disable_color = NEURON_DISABLE_COLOR
                outline_enable_color = NEURON_OUTLINE_ENABLE_COLOR
                outline_disable_color = NEURON_OUTLINE_DISABLE_COLOR

                y = y_mid + (step*(CIRCLE_RADIUS*2+NEURONS_MARGIN_Y))
                if(j < (MAX_NEURONS/2 - 0.5) or j > (nLs[i] - (MAX_NEURONS/2 + 0.5))):
                    id = canvas.create_oval(
                        x - CIRCLE_RADIUS, 
                        y - CIRCLE_RADIUS, 
                        x + CIRCLE_RADIUS, 
                        y + CIRCLE_RADIUS, 
                        fill=NEURON_DISABLE_COLOR, 
                        outline=NEURON_DISABLE_COLOR
                    )
                    step += 1
                else:
                    if(not passed):
                        step += 1
                        passed = True
                    enable_color = MULTI_NEURON_ENABLE_COLOR
                    disable_color = MULTI_NEURON_DISABLE_COLOR
                    outline_enable_color = MULTI_NEURON_ENABLE_OUTLINE_COLOR
                    outline_disable_color = MULTI_NEURON_DISABLE_OUTLINE_COLOR
                    id = multi_element_id
                    y = SCREEN_H / 2

                neuron = Neuron(nO, self.lr, x, y, id, enable_color, disable_color, outline_enable_color, outline_disable_color)
                neurons.append(neuron)

            nO = nLs[i]
            self.layers.append(neurons)
        canvas.update()
          
    def predict(self, x : List[float]):

        for layer, i in zip(self.layers, range(len(self.layers))):
            y_results = []        
            for neuron in layer:
                y = neuron.predict(x)

                self.set_color_neuron(neuron, NEURON_ENABLE_COLOR, neuron.outline_enable_color)
                if(i < len(self.layers)-1):
                    self.connect_next_layer(neuron, self.layers[i+1], LINE_COLOR)                    
                self.set_color_neuron(neuron, NEURON_DISABLE_COLOR, neuron.outline_disable_color)

                y_results.append(y)

            x = y_results

        return y_results

    def transformY(self, y : int):
        a = [0 for _ in range(self.outputs)]
        if(y >= len(a)):
            return a
        a[y] = 1
        return a

    def epoch(self, x : List[float], y : float):
        
        x = x/self.scaler

        # X_Matrix
        for i in range(len(x)):
            l = int(i/self.line_limit)
            c = i%self.line_limit
            id = self.x_matrix[l][c]
            intensity = int(x[i] * self.scaler)
            canvas.itemconfig(
                id, 
                fill = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            )
        # Forwardpass
        x_for = [x[:]]
        x_values = x[:]
        for layer, i in zip(self.layers, range(len(self.layers))):
            results = []
            
            for neuron in layer:                
                p = neuron.predict(x_values)
                self.set_color_neuron(neuron, neuron.enable_color, neuron.outline_enable_color)
                if(i < len(self.layers)-1):
                    self.connect_next_layer(neuron, self.layers[i+1], LINE_COLOR)            
                self.set_color_neuron(neuron, neuron.disable_color, neuron.outline_disable_color)
                results.append(p)

            x_values = results[:]
            x_for.append(x_values)

        targets : List[float] = self.transformY(y)
        output_error = []

        # Output layer
        output_layer = self.layers[-1]
        for i in range(len(output_layer)):
            delta = output_layer[i].fit(x_for[-2], targets[i], ids_matrix=self.w_matrix, scaler=self.scaler, draw=len(self.layers) == 1)
            output_error.append(delta)

        # Hidden layers
        for i in range(len(self.layers)-2, -1, -1):
            next_layer = self.layers[i+1]
            crr_layer_error = []
            for j in range(len(self.layers[i])):
                p = x_for[i+1][j]
                neuron = self.layers[i][j]

                w_error  = sum(next_layer[k].w[j] * output_error[k] for k in range(len(output_error)))
                
                self.set_color_neuron(neuron, neuron.enable_color, neuron.outline_enable_color)
                self.connect_next_layer(neuron, next_layer, LINE_COLOR, ERROR_COLOR)
                self.set_color_neuron(neuron, neuron.disable_color, neuron.outline_disable_color)
                
                delta = p * (1 - p) * w_error
                self.layers[i][j].fit(x_for[i], delta=delta, ids_matrix=self.w_matrix, scaler=self.scaler, draw = i == 0)
                crr_layer_error.append(delta)

            output_error = crr_layer_error[:]
        
        for i in range(len(x)):
            l = int(i/len(self.w_matrix[0]))
            c = i%len(self.w_matrix[0])
            id = self.w_matrix[l][c]

            canvas.itemconfig(
                id, 
                fill = f"#{0:02x}{0:02x}{0:02x}"
            )
       
    def score(self, x : List[float], y : float):
        r : List[float] = self.transformY(y)
        y : List[float] = self.predict(x)
        score = sum((ri-yi)**2 for ri, yi in zip(r,y))

        return score**0.5
    
    def scoreAll(self, x : List[List[float]], y : List[float]):

        score = 0
        for xi, yi in zip(x,y):
            score += self.score(xi, yi)
        return score / len(y)

    def fit(self, x : List[List[float]], y : List[float], epochs : int = 100):

        if self.line_limit == None:
            self.line_limit = MAX_WEIGHTS

        self.line_limit = min(len(x[0]), self.line_limit, MAX_WEIGHTS)
        total_lines = int(len(x[0])/self.line_limit)
        mid_x = SCREEN_W - SCREEN_W/4
        mid_y = SCREEN_H/4

        x_init = mid_x - int(self.line_limit/2 * WEIGHT_SIZE)
        y_init = mid_y - int(total_lines/2 * WEIGHT_SIZE)

        for i in range(total_lines):

            pos_y = y_init + i*WEIGHT_SIZE
            w_line = []
            x_line = []
            for j in range(min(len(x[0])-i*self.line_limit, self.line_limit)):
                pos_x = x_init + j*WEIGHT_SIZE

                w_id = canvas.create_rectangle(
                    pos_x - WEIGHT_SIZE/2,
                    pos_y - WEIGHT_SIZE/2,
                    pos_x + WEIGHT_SIZE/2,
                    pos_y + WEIGHT_SIZE/2,
                    fill="#000000", 
                    outline="#717171"
                )

                x_id = canvas.create_rectangle(
                    pos_x - WEIGHT_SIZE/2,
                    pos_y - WEIGHT_SIZE/2 + SCREEN_H/2,
                    pos_x + WEIGHT_SIZE/2,
                    pos_y + WEIGHT_SIZE/2 + SCREEN_H/2,
                    fill="#000000", 
                    outline="#717171"
                )
                w_line.append(w_id)
                x_line.append(x_id)

            self.w_matrix.append(w_line)
            self.x_matrix.append(x_line)



        
        for i in range(epochs):
            print(f'starting epoch\t [{i+1}]')
            for xj, yj in zip(x,y):
                self.epoch(xj,yj)
            print('score:\t\t', self.scoreAll(x,y))
            print()

    def set_color_neuron(self, neuron : Neuron, color : str, outline : str):
        canvas.itemconfig(
            neuron.entity_id, 
            fill=color,
            outline=outline
        )
        root.update()
        root.after(DELAY)

    def connect_next_neuron(self, neuron : Neuron, next_neuron : Neuron, color : str = "#ffffff"):
        line_id = canvas.create_line(
                neuron.X + CIRCLE_RADIUS, 
                neuron.Y, 
                next_neuron.X - CIRCLE_RADIUS, 
                next_neuron.Y, 
                fill=color, 
                width=2
            )
        return line_id
    
    def connect_next_layer(self, neuron : Neuron, next_layer : List[Neuron], color : str = "#ffffff", next_neuron_color = None):

        line_ids = []
        for next_neuron in next_layer:
            
            if(next_neuron_color == None):
                next_neuron_color = next_neuron.disable_color

            self.set_color_neuron(next_neuron, next_neuron_color, next_neuron.outline_disable_color)
            line_id = self.connect_next_neuron(neuron, next_neuron, color)
            self.set_color_neuron(next_neuron, next_neuron.disable_color, next_neuron.outline_disable_color)
            line_ids.append(line_id)

        root.update()    
        for line_id in line_ids:
            canvas.delete(line_id)
        root.update()