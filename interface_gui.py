from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from PIL import ImageFile
from keras import backend as k
ImageFile.LOAD_TRUNCATED_IMAGES = True


from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog

batch_size = 25
#epochs = 2
img_height = 150
img_width = 150


classes = ["Heart","Oblong","Oval","Round","Square"]

root = Tk()
root.title("Image Classifier")
root.geometry("850x900")
#load image function
def load():
	global image_1
	global img_1
	global label_2
	root.filename = filedialog.askopenfilename(initialdir = "/home/ragon/Tensorflow_python/venv/tkinter/images/",title = "Open image",filetypes = (("png files","*.png"),("All files","*.*")))
	image_1 = ImageTk.PhotoImage(Image.open(root.filename))
	label_2.grid_forget()
	label_2 = Label(root,image = image_1 , relief = 'sunken')
	label_2.grid(row=4,column = 0,pady=10)
	img_1 = cv2.imread(root.filename)
	img_1 = np.asarray(img_1)
	img_1 = cv2.resize(img_1,(img_width,img_height))
	img_1 = img_1[np.newaxis,...]


def load2():

	#newmodel
	new_model = load_model("model_data_trial.h5")
	predict_image_generator = ImageDataGenerator(rescale = 1.0/255)
	#predict_data = predict_image_generator.flow_from_directory(batch_size = 2,directory = "/home/ragon/Tensorflow_python/venv/tkinter/images/",target_size = (img_height,img_width),class_mode = 'binary')
	#test_ac = new_model.predict_generator(predict_data)
	global label_3
	test_ac = new_model.predict_generator(predict_image_generator.flow(img_1))
	print(max(max(test_ac)))
	print(test_ac)
	a=np.where(test_ac == max(max(test_ac)))
	d=classes[a[1][0]]
	label_3.grid_forget()
	label_3 = Label(root,text=d , relief = 'sunken',font = ('times',20,'bold'))
	label_3.grid(row=3,column=0)

label_3 = Label(root,text='result', relief = 'sunken',font = ('times',20,'bold'))
label_3.grid(row=3,column=0)
label_2 = Label(root,image = None , relief = 'sunken')
label_2.grid(row=4,column = 0,pady=10)	

label_1 = Label(root,text = "FaceShape Application",relief = 'solid',width = 68,anchor = 'center',bg = 'red',font=('times',20,'bold'))
label_1.grid(row=0,column = 0,padx=(10,0),pady=10)


#buttons
button=Button(root,text = "Add Image",command = load,width = 100)
button.grid(row = 1,column = 0,pady=10)
button2 = Button(root,text = "Run Model",command = load2,width = 100)
button2.grid(row=2,column=0,pady = 10)


root.mainloop()
