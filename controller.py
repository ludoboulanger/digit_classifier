import torch
from constants import Constants
from drawer import Drawer
from model import DigitClassifier

from PIL import Image, ImageDraw

import numpy as np

class Controller:
    
    def __init__(self) -> None:

        # Drawing Variables
        self.previous_coords = (0, 0)

        self.active_line = []
        self.lines = []

        # Init the UI
        self.drawer = Drawer()

        # Init a model
        self.model = DigitClassifier()
        self.model.load_state_dict(torch.load(Constants.MODEL_PATH.value))
        self.model.eval()

        # Events
        self.map_events()

    def map_events(self):
        self.drawer.canvas.bind("<Button-1>", self.handle_mouse_clicked)
        self.drawer.canvas.bind("<B1-Motion>", self.handle_mouse_dragged)
        self.drawer.canvas.bind('<ButtonRelease-1>', self.handle_mouse_released)

        self.drawer.clear_button.bind("<Button-1>", self.handle_clear_clicked)

        self.drawer.predict_button.bind("<Button-1>", self.handle_prediction_clicked)

    # Events
    def handle_mouse_dragged(self, event):
        self.drawer.canvas.create_line(
            self.previous_coords[0],
            self.previous_coords[1],
            event.x,
            event.y,
            width=10,
            fill=Constants.WHITE.value
            )

        self.previous_coords = (event.x, event.y)

        self.active_line.append((event.x, event.y))

    def handle_mouse_clicked(self, event):
        self.previous_coords = (event.x, event.y)

    def handle_mouse_released(self, event):
        self.lines.append(self.active_line)
        self.active_line = []

    def handle_clear_clicked(self, event):
        self.active_line = []
        self.lines = []
        self.drawer.prediction_value.set(f"Prediction :: ")
        self.drawer.canvas.delete("all")

    def handle_prediction_clicked(self, event):
        image = Image.new(mode="L", size=(Constants.WIDTH.value, Constants.HEIGHT.value))

        drawing = ImageDraw.Draw(image)

        for line in self.lines:
            drawing.line(line, fill=Constants.WHITE.value, width=35)

        image = np.array(image.resize((28,28), Image.ANTIALIAS))

        torch_img = torch.Tensor(image).unsqueeze(0).unsqueeze(0)

        model_predictions = self.model(torch_img)
        predicted_digit = torch.max(model_predictions, 1)[1].item()

        self.drawer.prediction_value.set(f"Prediction :: {predicted_digit}")

    def start_app(self):
        self.drawer.mainloop()


if __name__ == "__main__":
    Controller().start_app()
