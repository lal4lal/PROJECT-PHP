from pynput import keyboard

esc_pressed = False
space_pressed = False

def start_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener

def on_press(key):
    try:
        k = key.char
    except:
        k = key.name
    if k == 'esc':
        global esc_pressed
        esc_pressed = True
    elif k == 'space':
        global space_pressed
        space_pressed = True
