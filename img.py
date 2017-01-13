import gtk
import time
import threading
import gobject
import gtk.gdk
import os,subprocess
#from os import system
import csv
from threading import BoundedSemaphore
import random
import signal

XDIM=320
YDIM=135

def handler(signum,frame):
	print('received signal #',signum)
	gtkQuit()
	raise IOError('signal error')

signal.signal(signal.SIGQUIT,handler)

class GTKThread(threading.Thread):
	def __init__(self,image,a,b):
		print("init thread")
		super(GTKThread, self).__init__()
		self.image = image
		self.quit = False
		self.i=0
		self.j=0
		self.a=a
		self.b=b
		self.busySem=BoundedSemaphore(value=1)
		self.updatePixelBuf()
		#cmdList=['firefox','https://en.wikipedia.org/wiki/Main_Page']
		#with open(os.devnull, 'wb') as devnull:
		#	subprocess.check_call(cmdList,stdout=devnull,stderr=subprocess.STDOUT)

	def xdotool(self,cmdList):
		self.busySem.acquire()
		with open(os.devnull, 'wb') as devnull:
			subprocess.check_call(cmdList,stdout=devnull,stderr=subprocess.STDOUT)
		self.busySem.release()

	def minimize(self,title):
		self.xdotool(['xdotool','search',title,'windowminimize','%@'])

	def select(self,title):
		self.xdotool(['xdotool','search',title,'windowmap','%@'])

	def moveWindow(self,title,x,y):
		self.xdotool(['xdotool','search',title,'windowmove','%@',str(x),str(y)])
#		system('xdotool search "'+title+'" windowmove '+str(x)+' '+str(y))

	def updatePixelBuf(self):
		self.busySem.acquire()
		gtk.gdk.threads_enter()
		try:
			self.rootWindow = gtk.gdk.get_default_root_window()
			self.sz = self.rootWindow.get_size()
			self.pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,self.sz[0],self.sz[1])
			self.pb = self.pb.get_from_drawable(self.rootWindow,self.rootWindow.get_colormap(),0,0,0,0,self.sz[0],self.sz[1])
			#pb.scale(pb,0,0,200,200,0,0,1,1,gtk.gdk.INTERP_NEAREST)
			#pb=pb.scale_simple(200,200,gtk.gdk.INTERP_NEAREST)
		finally:
			gtk.gdk.threads_leave()
		self.busySem.release()
			
	def setCropped(self,x,y):
		self.busySem.acquire()
		gtk.gdk.threads_enter()
		try:
			self.subpb=self.pb.subpixbuf(x,y,self.a,self.b)
			self.image.clear()
			self.image.set_from_pixbuf(self.subpb)
		finally:
			gtk.gdk.threads_leave()
		self.busySem.release()

	def update_image(self,img):
		self.busySem.acquire()
		gtk.gdk.threads_enter()
		try:
			self.image.clear()
			self.image.set_from_file(img)
		finally:
			gtk.gdk.threads_leave()
		self.busySem.release()
		return False

	def set_from_pixbuf(self):
		self.busySem.acquire()
		gtk.gdk.threads_enter()
		try:
			self.image.clear()
			self.image.set_from_pixbuf(self.pb)
		finally:
			gtk.gdk.threads_leave()
		self.busySem.release()
		return False

	def writeCSV(filename):
		try:
			with open(filename,'w') as csvFile:
				csvWriter=csv.writer(csvFile,delimiter=',')
				csvWriter.writerow(self.array)
		except:
			print('unable to write to '+filename)

	def getCropped(self,x,y,filename):
		self.busySem.acquire()
		gtk.gdk.threads_enter()
		try:
			if self.subpb:
				self.subpb.save(filename,'png')
				#self.array=self.subpb.get_pixels_array()#.flatten()
				#self.array=self.subpb.get_pixels_array().flatten()
			else:
				print('failed:'+str(x)+','+str(y))
		finally:
			gtk.gdk.threads_leave()
		self.busySem.release()

	def generateExamples(self):
		self.minimize('Terminal')
		time.sleep(1)
		self.updatePixelBuf()
		self.generateNegative(0.10)
		self.select('Terminal')
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
		self.j += 1
		if self.j>=(1920-self.a):
			self.j = 0
			self.i += self.b
			if self.i>=(1080-self.b):
				self.i = 0
		self.moveWindow('Wikipedia',self.j,0)
		#self.moveWindow('Wikipedia, the free encyclopedia - Mozilla Firefox',self.j,0)

	def run(self):
		while not self.quit:
			#for i in range(10):
			#	self.readAndDisplay('data/'+'neg'+str(i)+'.csv')
			self.generateExamples()
			self.quit=1
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
mainloop = gobject.MainLoop()
image = gtk.Image()
gtkThread=GTKThread(image,XDIM,YDIM)

window = gtk.Window(gtk.WINDOW_TOPLEVEL)
try:
	window.set_icon_from_file('icon.png')
except:
	print('could not find icon.png.  using question mark icon.')

def gtkQuit():
	global mainloop,t
	mainloop.quit()
	gtkThread.quit=True

window.set_border_width(10)
window.move(1024,0)
#window.connect("delete_event", self.close_application)
#window.connect("event",event)
#window.connect("expose_event",expose)
#window.connect("window_state_event",window_state_changed)
#window.connect("destroy",lambda w: gtk.main_quit())
window.connect("destroy",lambda w: gtkQuit())
window.add(image)
window.show_all()

#image.set_from_file('san_francisco.jpg')

gtkThread.start()
#gtk.main()
try:
	mainloop.run()
except KeyboardInterrupt:
	gtkThread.quit=True
	mainloop.quit()
gtkThread.quit=True
