from typing import List, Callable

from PIL import Image, ImageTk
import tkinter as tk
import cv2
import numpy as np

root = tk.Tk()
root.title("Kernel")
root.attributes('-fullscreen', True)
root.bind("<Escape>", lambda e: root.destroy())

PIXEL_SIZE = 5
KERNEL_VALUE_SIZE = 40
KERNEL_VALUE_INPUT = 40
KERNEL_VALUE_INPUT_GAP = 20


SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()
BACKGROUND_COLOR = "#070707"

canvas = tk.Canvas(root, width=SCREEN_W, height=SCREEN_H, bg=BACKGROUND_COLOR)
canvas.pack()



class Kernel:
	def __init__(self, image_path : str, f: Callable[[List[List[int]], int], List[List[int]]]):
		image_archive = cv2.imread(image_path, 0)
		image_archive = cv2.resize(image_archive, (100, 100))
		self.image = np.array(image_archive)
		self.image_original = Image.fromarray(self.image.astype(np.uint8))
		self.image_original = self.image_original.resize((100 * PIXEL_SIZE, 100 * PIXEL_SIZE), Image.NEAREST)
		self.image_original = ImageTk.PhotoImage(self.image_original)

		self.filter_function = f

		original_image_y_init = SCREEN_H/2 - (len(   self.image)*PIXEL_SIZE)/2
		original_image_x_init = SCREEN_W/6 - (len(self.image[0])*PIXEL_SIZE)/2
			
		canvas.create_image(original_image_x_init, original_image_y_init, anchor="nw", image=self.image_original, tags="original_img")

		
		kernel_size_y_mid = SCREEN_H/6
		kernel_size_x_mid = SCREEN_W/2
		kernel_size_input_width = 60
		kernel_size_input_gap = 30
		
		x = tk.Label(root, text="X", fg= "white", bg= BACKGROUND_COLOR)
		x.place(
			x= kernel_size_x_mid - kernel_size_input_gap/2, 
			y= kernel_size_y_mid, 
			width= kernel_size_input_gap
		)
		
		vcmd_int = (root.register(self.validate_kernel_size), "%P")
		
		self.kernel_size_var_x = tk.StringVar(value="3")
		self.kernel_size_var_y = tk.StringVar(value="3")
		
		self.kernel_size_var_x.trace_add("write", self.load_kernel_size)
		self.kernel_size_var_y.trace_add("write", self.load_kernel_size)
		
		self.kernel_size_entry = (
			tk.Entry(
				root, 
				textvariable=self.kernel_size_var_y, 
				validate="key", 
				validatecommand=vcmd_int, 
				justify="center"
			),
			tk.Entry(
				root, 
				textvariable=self.kernel_size_var_x, 
				validate="key",
				validatecommand=vcmd_int, 
				justify="center"
			)
		)
		
		self.kernel_size_entry[0].place(
			x= kernel_size_x_mid - (kernel_size_input_gap/2) - kernel_size_input_width, 
			y= kernel_size_y_mid, 
			width= kernel_size_input_width
		)
		
		self.kernel_size_entry[1].place(
			x= kernel_size_x_mid + (kernel_size_input_gap/2), 
			y= kernel_size_y_mid, 
			width= kernel_size_input_width
		)
		
		
		self.kernel_values_entry = []
		self.kernel_string_vars = [] 
		self.kernel = []
		self.load_kernel_size()
		self.apply_kernel()
	
	def load_kernel_size(self, *args):
		
		for line in self.kernel_values_entry:
			for entry, id in line:
				entry.destroy()
				canvas.delete(id)
		self.kernel_values_entry = []
				

		vcmd_int = (root.register(self.is_int), "%P")  

		y_mid = SCREEN_H/2
		x_mid = SCREEN_W/2

		y_quanti = int(self.kernel_size_entry[0].get()) if self.kernel_size_entry[0].get() != "" and self.kernel_size_entry[0].get() != "-" else 0
		x_quanti = int(self.kernel_size_entry[1].get()) if self.kernel_size_entry[1].get() != "" and self.kernel_size_entry[1].get() != "-" else 0
		
		y_init = y_mid - ((y_quanti * KERNEL_VALUE_INPUT)/2 + (y_quanti/2-1)*KERNEL_VALUE_INPUT_GAP)
		x_init = x_mid - ((x_quanti * KERNEL_VALUE_INPUT)/2 + (x_quanti/2-1)*KERNEL_VALUE_INPUT_GAP)

		for i in range(y_quanti):
			line_entry = []
			line_vars = []
			for j in range(x_quanti):
				
				sv = tk.StringVar(value="1")
				sv.trace_add("write", self.update_kernel)
				
				entry = tk.Entry(
					root,
					validate= "key", 
					validatecommand= vcmd_int, 
					textvariable= sv,
					justify= "center"
				)
				
				id = canvas.create_rectangle(
						x_init + j*KERNEL_VALUE_INPUT + (j-1)*KERNEL_VALUE_INPUT_GAP,
						y_init + i*KERNEL_VALUE_INPUT + (i-1)*KERNEL_VALUE_INPUT_GAP,
						x_init + (j+1)*KERNEL_VALUE_INPUT + (j-1)*KERNEL_VALUE_INPUT_GAP,
						y_init + (i+1)*KERNEL_VALUE_INPUT + (i-1)*KERNEL_VALUE_INPUT_GAP,
						fill=    "#d3d3d3", 
						outline= "#d3d3d3"
					)
				
				entry.place(
					x= x_init + j*KERNEL_VALUE_INPUT + (j-1)*KERNEL_VALUE_INPUT_GAP,
					y= y_init + i*KERNEL_VALUE_INPUT + (i-1)*KERNEL_VALUE_INPUT_GAP + (KERNEL_VALUE_INPUT - 10)/2, # entry.winfo_height()/2
					width= KERNEL_VALUE_INPUT
				) 
				entry.config(bg="#d3d3d3", highlightthickness=0, bd=0)

				line_entry.append((entry,id))
				
				line_vars.append(sv)
			self.kernel_string_vars.append(line_vars)
			self.kernel_values_entry.append(line_entry)

		self.update_kernel()

	def validate_kernel_size(self, v):
		if v == "":
			return True
		return v.isdigit() and int(v) <= 6

	def is_int(self, v):
		if v == "" or v == "-":
			return True
		
		if v[0] == "-":
			v = v[1:]

		if "." in v:
			if v.count('.') > 1:
				return False
			return v.replace('.', '').isdigit()

		return v.isdigit()
	
	def update_kernel(self, *args):

		fltr = []
		for line in self.kernel_values_entry:
			f = []
			for entry, id in line:
				if entry.get() != "" and entry.get() != "-":
					_ = float(entry.get())
				else:
					_ = 0
				f.append(_)
			fltr.append(f)
		
		self.kernel = fltr[:]
		self.apply_kernel()

	def apply_kernel(self):

		filtered_image = self.filter_function(self.image.copy(), self.kernel.copy())
		filtered_image = np.array(filtered_image)

		img = Image.fromarray(filtered_image.astype(np.uint8))
		width, height = filtered_image.shape[1], filtered_image.shape[0]
		img = img.resize((width*PIXEL_SIZE, height*PIXEL_SIZE), Image.NEAREST)

		self.tk_img = ImageTk.PhotoImage(img)

		filtered_image_y_init = SCREEN_H/2 - (len(filtered_image   )/2*PIXEL_SIZE)
		filtered_image_x_init = (SCREEN_W - SCREEN_W/5) - (len(filtered_image[0])/2*PIXEL_SIZE)


		canvas.delete("filtered_img")
		
		canvas.create_image(filtered_image_x_init, filtered_image_y_init, anchor="nw", image=self.tk_img, tags="filtered_img")

def init():
	root.mainloop()