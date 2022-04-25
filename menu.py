from tkinter import *
from tkinter import filedialog
import os
import shutil
import main


font = ("Arial", 36, "bold")


def open_files():
    filename = filedialog.askopenfilename(
        initialdir="/", title="Select A Image", filetype=(("jpeg files", "*.jpg"), ("png files", "*.png*")))
    shutil.move(filename, "./train/" + os.path.split(filename)[1])


window = Tk()

window.title("Face Recognition")
window.geometry("400x368+760+200")
window.configure(bg="#230043", pady=20)
window.resizable(False, False)

images_button = Button(window, text="PUT IMAGES", font=font, command=lambda: open_files(
), width="12", fg="#230043", bg="#b28baf", height="1")
images_button.pack()

run_button = Button(window, text="FIND FACES", font=font, command=lambda: main.main(
), width="12", fg="#230043", bg="#b28baf", height="1")
run_button.pack()

quit_button = Button(window, text="QUIT", font=font, command=lambda: window.destroy(
), width="12", fg="#230043", bg="#b28baf", height="1")
quit_button.pack()

credits_label = Label(window, text="Developed by Keyan de Klerk", font=(
    "Arial", 14, "bold"), fg="#b28baf", bg="#230043")
credits_label.pack()

window.mainloop()
