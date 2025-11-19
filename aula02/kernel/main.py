from typing import List
from Kernel import Kernel, init

def filter_function(image: List[List[int]], kernel: List[List[int]]):
    filtered_image = image[:]
    
    for linha in range(len(image)):
        # print('linha: ', linha)
        for coluna in range(len(image)):
            # print('coluna: ', coluna)
            pesos = 0
            pixellindo = 0
            for i, i2 in zip(range(-1, 2, 1), range(0, 3, 1)):
                for j, j2 in zip(range(-1, 2, 1), range(0, 3, 1)):
                    
                    if coluna + j > len(image)-1 or linha + i > len(image) -1:
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
                          
            filtered_image[linha][coluna] = pixellindo
    
    
    return filtered_image

Kernel = Kernel("minion.png", filter_function)

init()