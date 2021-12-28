from tkinter import StringVar, Tk, Canvas, Button
from tkinter.ttk import Frame, Label

from constants import Constants

class Drawer(Tk):

    def __init__(self):
        Tk.__init__(self)

        self.prediction_value = StringVar()
        self.prediction_value.set("Prediction :: ")

        # Build the UI
        self.build_ui()

    def build_ui(self):
        self.title("Digit Classifier")
        self.geometry(f"{Constants.HEIGHT.value}x{Constants.WIDTH.value}")
        self.config(bg=Constants.WHITE.value)

        # Prediction Box
        self.prediction_frame = Frame(self)
        self.prediction_frame.pack(fill='x')
        
        self.prediction_text = Label(self.prediction_frame, textvariable=self.prediction_value, font=("Arial", 16))
        self.prediction_text.pack(side='left')

        # Canvas for drawing
        self.canvas_frame = Frame(self)
        self.canvas_frame.pack(fill='both', expand=True)

        self.canvas = Canvas(self.canvas_frame, bg=Constants.BLACK.value)
        self.canvas.pack(fill='both', expand=True)

        # Button Box
        self.button_frame = Frame(self)
        self.button_frame.pack(fill='x')

        self.predict_button = Button(self.button_frame, text='Predict', width=6, height=2)
        self.predict_button.pack(side='right', padx=5, pady=5)

        self.clear_button = Button(self.button_frame, text='Clear', width=6, height=2)
        self.clear_button.pack(side='right', pady=5, padx=5)