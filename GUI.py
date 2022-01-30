from tkinter import *
import numpy as np
from PIL import ImageGrab
from model import *
 
window = Tk()
window.title("Digit Recognition BPN")
l1 = Label()
 
 
def Create_GUI():
    global l1
 
    widget = cv
    # Setting co-ordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
 
    # Image is captured from canvas and is resized to (28 X 28) px
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
 
    # Converting rgb to grayscale image
    img = img.convert('L')
 
    # Extracting pixel matrix of image and converting it to a input_vector of (1, 784)
    x = np.asarray(img)
    input_vector = np.zeros((1, 784))
    k = 0
    for i in range(28):
        for j in range(28):
            input_vector[0][k] = x[i][j]
            k += 1
 
    # Loading Thetas
    theta1 = np.loadtxt('theta1.txt')
    theta2 = np.loadtxt('theta2.txt')
 
    # Calling function for prediction
    pred = predict(theta1, theta2, input_vector/255)
 
    # Displaying the result
    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Times New Roman', 20))
    l1.place(x=230, y=420)
 
 
lastx, lasty = None, None
 
 
# Clears the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()
 
 
# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y
 
 
# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='#ddccff', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y
 
 
# Label
L1 = Label(window, text="Draw a digit in the canvas space provided below", font=('Times New Roman', 20), fg="#660066")
L1.place(x=35, y=10)
 
# Button to clear canvas
b1 = Button(window, text="Clear Canvas", font=('Times New Roman', 13), bg="#ddccff", fg="black", command=clear_widget)
b1.place(x=120, y=370)
 
# Button to predict digit drawn on canvas
b2 = Button(window, text="Predict from image capture", font=('Times New Roman', 13), bg="#ddccff", fg="black", command=Create_GUI)
b2.place(x=270, y=370)
 
# Setting properties of canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)
 
cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
