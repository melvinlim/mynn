import gtk
import time
import threading
import gobject
import gtk.gdk
import os,subprocess
from os import system
import csv
from threading import BoundedSemaphore
import random
import signal

XDIM=320
YDIM=135

class GTKThread(threading.Thread):
	def testLoopCB(self,widget):
		self.task='testLoop'
	def pngToArrayCB(self,widget):
		self.task='pngToArray'
	def readAndDisplayCB(self,widget):
		self.task='readAndDisplay'
	def enterCrit(self):
		self.busySem.acquire()
		gtk.gdk.threads_enter()
	def exitCrit(self):
		gtk.gdk.threads_leave()
		self.busySem.release()
	def gtkQuitCB(self,widget):
		self.gtkQuit()
	def gtkQuit(self):
		self.stop=True
		self.mainloop.quit()
	def handler(self,signum,frame):
		print('received signal #',signum)
		self.gtkQuit()
		#raise IOError('signal error')
	def __init__(self,a,b):
		print("init thread")
		super(GTKThread,self).__init__()
		self.mainloop = gobject.MainLoop()
		signal.signal(signal.SIGQUIT,self.handler)
		self.image=gtk.Image()
		self.box1=gtk.VBox(False,0)
		self.box1.pack_start(self.image)
		self.quitButton=gtk.Button('Quit')
		self.quitButton.connect('clicked',self.gtkQuitCB)
		self.testLoopButton=gtk.Button('testLoop')
		self.testLoopButton.connect('clicked',self.testLoopCB)
		self.pngToArrayButton=gtk.Button('pngToArray')
		self.pngToArrayButton.connect('clicked',self.pngToArrayCB)
		self.readAndDisplayButton=gtk.Button('readAndDisplay')
		self.readAndDisplayButton.connect('clicked',self.readAndDisplayCB)
		self.box1.pack_start(self.testLoopButton,True,True,0)
		self.box1.pack_start(self.pngToArrayButton,True,True,0)
		self.box1.pack_start(self.readAndDisplayButton,True,True,0)
		self.box1.pack_start(self.quitButton,True,True,0)
		self.mainWindow = gtk.Window(gtk.WINDOW_TOPLEVEL)
		self.task=0
		try:
			self.mainWindow.set_icon_from_file('icon.png')
		except:
			print('could not find icon.png.  using question mark icon.')

		self.mainWindow.set_border_width(10)
		self.mainWindow.move(1024,0)
		#self.mainWindow.connect("event",event)
		#self.mainWindow.connect("expose_event",expose)
		#self.mainWindow.connect("window_state_event",window_state_changed)
		self.mainWindow.connect("destroy",lambda w: self.gtkQuitCB())
		self.mainWindow.connect("delete_event",lambda a,b: self.gtkQuitCB())
		self.mainWindow.add(self.box1)
		self.mainWindow.show_all()
		self.i=0
		self.j=0
		self.a=a
		self.b=b
		self.busySem=BoundedSemaphore(value=1)
		self.updatePixelBuf()
		self.setCropped(0,0)
		self.stop=False
		self.start()
		try:
			self.mainloop.run()
		except KeyboardInterrupt:
			self.stop=True
			self.mainloop.quit()

	def xdotool(self,cmdList):
		self.enterCrit()
		with open(os.devnull, 'wb') as devnull:
			subprocess.check_call(cmdList,stdout=devnull,stderr=subprocess.STDOUT)
		self.exitCrit()

	def minimize(self,title):
		self.xdotool(['xdotool','search','--class',title,'windowminimize','%@'])

	def select(self,title):
		self.xdotool(['xdotool','search','--class',title,'windowmap','--sync','%@'])

	def moveWindow(self,title,x,y):
		self.xdotool(['xdotool','search',title,'windowmove','%@',str(x),str(y)])
		#system('xdotool search --name "'+title+'" windowmove %@ '+str(x)+' '+str(y))

	def updatePixelBuf(self):
		self.enterCrit()
		self.rootWindow = gtk.gdk.get_default_root_window()
		self.sz = self.rootWindow.get_size()
		self.pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,self.sz[0],self.sz[1])
		self.pb = self.pb.get_from_drawable(self.rootWindow,self.rootWindow.get_colormap(),0,0,0,0,self.sz[0],self.sz[1])
		#pb.scale(pb,0,0,200,200,0,0,1,1,gtk.gdk.INTERP_NEAREST)
		#pb=pb.scale_simple(200,200,gtk.gdk.INTERP_NEAREST)
		self.exitCrit()
			
	def setCropped(self,x,y):
		self.enterCrit()
		self.subpb=self.pb.subpixbuf(x,y,self.a,self.b)
		self.image.clear()
		self.image.set_from_pixbuf(self.subpb)
		self.image.queue_draw()
		self.exitCrit()

	def update_image(self,img):
		self.enterCrit()
		self.image.clear()
		self.image.set_from_file(img)
		self.exitCrit()
		return False

	def set_from_pixbuf(self):
		self.enterCrit()
		self.image.clear()
		self.image.set_from_pixbuf(self.pb)
		self.exitCrit()
		return False

	def writeCSV(filename):
		try:
			with open(filename,'w') as csvFile:
				csvWriter=csv.writer(csvFile,delimiter=',')
				csvWriter.writerow(self.array)
		except:
			print('unable to write to '+filename)

	def getCropped(self,x,y,filename):
		self.enterCrit()
		if self.subpb:
			self.subpb.save(filename,'png')
			#self.array=self.subpb.get_pixels_array()#.flatten()
			#self.array=self.subpb.get_pixels_array().flatten()
		else:
			print('failed:'+str(x)+','+str(y))
		self.exitCrit()

	def generateExamples(self):
		#self.minimize('Terminal')
		#time.sleep(1)
		#gobject.idle_add(self.updatePixelBuf)
		#print("generating negative examples")
		#self.generateNegative(0.10)
		#gobject.idle_add(self.select,'Terminal')
		print("generating positive examples")
		self.generatePositive(0.10)
	
	def generatePositive(self,delay=0.5):
		yoffset=0
		n=0
		for j in range(0,1920,XDIM):
			for i in range(0,1080,YDIM):
				for k in range(5):
					m=random.randint(0,XDIM/5)
					l=random.randint(0,YDIM/5)
					xtarg=j+m
					ytarg=i+l+yoffset
					if xtarg>XDIM:
						xtarg=j-m
					if ytarg>YDIM:
						ytarg=i-l-yoffset
					m=random.randint(0,XDIM/5)
					l=random.randint(0,YDIM/5)
					wx=j+m
					wy=i+l+yoffset
					gobject.idle_add(self.moveWindow,'Terminal',wx,wy)
					time.sleep(0.5)
					gobject.idle_add(self.updatePixelBuf)
					time.sleep(delay)
					gobject.idle_add(self.setCropped,xtarg,ytarg)
					filename='data/'+'pos'+str(n)+'.png'
					gobject.idle_add(self.getCropped,xtarg,ytarg,filename)
					n+=1

	def generateNegative(self,delay=0.5):
		yoffset=25
		n=0
		for j in range(0,1920,XDIM):
			for i in range(0,1080,YDIM):
				if j>0:
					gobject.idle_add(self.moveWindow,'img.py',0,0)
					gobject.idle_add(self.updatePixelBuf)
					time.sleep(delay)
				for k in range(5):
					m=random.randint(0,XDIM/5)
					l=random.randint(0,YDIM/5)
					xtarg=j+m
					ytarg=i+l+yoffset
					if xtarg>XDIM:
						xtarg=j-m
					if ytarg>YDIM:
						ytarg=i-l-yoffset
					gobject.idle_add(self.setCropped,xtarg,ytarg)
					filename='data/'+'neg'+str(n)+'.png'
					gobject.idle_add(self.getCropped,xtarg,ytarg,filename)
					n+=1
					time.sleep(delay)

	def pngToArray(self,filename):
		pb=gtk.gdk.pixbuf_new_from_file(filename)
		print(pb.get_pixels_array())#.flatten())

	def readAndDisplay(self,filename):
		results=[]
		try:
			with open(filename,'r') as csvFile:
				csvReader=csv.reader(csvFile,delimiter=',')
				for row in csvReader:
					print(row)
		except:
			print('failed to open '+filename)

	def testLoop(self):
		gobject.idle_add(self.setCropped,self.j,self.i)
		self.j += 100
		if self.j>=(1920-self.a):
			self.j = 0
			self.i += 100
			if self.i>=(1080-self.b):
				self.task=0
				self.i = 0
		self.moveWindow('Terminal',self.j,self.i)

	def run(self):
		while not self.stop:
			if(self.task=='testLoop'):
				self.testLoop()
			elif(self.task=='pngToArray'):
				for i in range(10):
					self.pngToArray('data/'+'neg'+str(self.i)+'.png')
				self.task=0
			elif(self.task=='readAndDisplay'):
				self.i+=1
				if self.i>=10:
					self.i=0
					self.task=0
				self.readAndDisplay('data/'+'neg'+str(self.i)+'.csv')
			elif(self.task=='quitTask'):
				self.gtkQuit()
			#self.generateExamples()
			#self.gtkQuit()
			#self.testLoop()
			time.sleep(0.001)

def event(widget,event):
	print(event)
	print(widget)

def window_state_changed(widget,event):
	print(event)

def expose(widget,event):
	image=widget.get_child()
	image.set_from_file('san_francisco.jpg')

gobject.threads_init()
gtkThread=GTKThread(XDIM,YDIM)

gtkThread.stop=True
