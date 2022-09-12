# Input
from pynput.mouse import Listener as MouseListener
from pynput import mouse

def on_click(x,y,a,b):
    print('clicked!')


mouse_listener = MouseListener(on_click=on_click)
mouse_listener.start()