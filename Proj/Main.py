import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import preprocess as pre
import RFALG as RF
import DTALG as DT
import SVMALG as SV

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
	global window
	def clear():
	    print("Clear1")
	    txt1.delete(0, 'end')    



	window = tk.Tk()
	window.title("Liver Disease Prediction")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Liver Disease Prediction" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl.place(x=100, y=200)
	
	txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt.place(x=400, y=215)


	def browse():
		path=filedialog.askopenfilename()
		print(path)
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input error", "Select Train Dataset")	

	def preproc():
		sym=txt.get()
		if sym != "" :
			pre.process(sym)
			print("preprocess")
			tm.showinfo("Input", "Preprocess Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def RFprocess():
		sym=txt.get()
		if sym != "" :
			RF.process(sym)
			tm.showinfo("Input", "RANDOM FOREST Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def DTprocess():
		sym=txt.get()
		if sym != "" :
			DT.process(sym)
			tm.showinfo("Input", "DECISION TREE Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")
	

	def SVMprocess():
		sym=txt.get()
		if sym != "" :
			SV.process(sym)
			tm.showinfo("Input", "SVM Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse.place(x=650, y=200)

	clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
	clearButton.place(x=950, y=200)
	 
	proc = tk.Button(window, text="Preprocess", command=preproc  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	proc.place(x=160, y=600)
	

	RFbutton = tk.Button(window, text="RANDOM FOREST", command=RFprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	RFbutton.place(x=400, y=600)


	NNbutton = tk.Button(window, text="SVM", command=SVMprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	NNbutton.place(x=620, y=600)

	DCbutton = tk.Button(window, text="DECISION TREE", command=DTprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	DCbutton.place(x=800, y=600)



	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=1030, y=600)

	window.mainloop()
Home()

