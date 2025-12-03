from typing import List
from Kernel import Kernel, init
import copy

def filter_function(image: List[List[int]], kernel: List[List[int]]):
    
    stride = (2,2)
    # filtered_image = copy.deepcopy(image)
    filtered_image = []

    for linha in range(0, len(image), stride[0]):
        
        new_lin = []
        for coluna in range(0, len(image[0]), stride[1]):
            
            pesos = 0
            pixellindo = 0
            for i, i2 in zip(range(-1, 2, 1), range(0, len(kernel), 1)):
                for j, j2 in zip(range(-1, 2, 1), range(0, len(kernel[0]), 1)):
                    
                    if coluna + j > len(image)-1 or coluna + j < 0 or linha + i > len(image) -1 or linha + i < 0:
                        pixel = 0
                    else:
                        pixel = image[linha + i][coluna + j]
                        
                    pixellindo += pixel * kernel[i2][j2]
                    pesos += kernel[i2][j2]
   
            if pesos < 1:
                pesos = 1
            pixellindo = pixellindo / pesos
            
            
            if pixellindo > 255:
                pixellindo = 255
            if pixellindo < 0:
                pixellindo = 0
                          
            new_lin.append(pixellindo)
        filtered_image.append(new_lin)
            
            # filtered_image[linha][coluna] = pixellindo
    
    # print(pesos)
    
    return filtered_image

Kernel = Kernel("minion.png", filter_function)

init()