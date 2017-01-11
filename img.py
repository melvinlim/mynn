import gtk
import time
import threading
import gobject
import gtk.gdk
import os,subprocess
#from os import system

XDIM=320
YDIM=145

class MyThread(threading.Thread):
	def __init__(self,image,a,b):
		print("init thread")
		super(MyThread, self).__init__()
		self.image = image
		self.quit = False
		self.i=0
		self.j=0
		self.a=a
		self.b=b
		self.updatePixelBuf()

	def moveWindow(self,title,x,y):
		cmdList=['xdotool','search',title,'windowmove',str(x),str(y)]
		with open(os.devnull, 'wb') as devnull:
			subprocess.check_call(cmdList,stdout=devnull,stderr=subprocess.STDOUT)
#		system('xdotool search "'+title+'" windowmove '+str(x)+' '+str(y)+'>/dev/null')
#		system('xdotool search "'+title+'" windowmove '+str(x)+' '+str(y))
#		system('xdotool search "'+title+'" windowactivate --sync mousemove --window %1 500 10')
#		system('xdotool mousedown 1')
#		system('xdotool mousemove_relative --sync '+str(x)+' '+str(y))
#		system('xdotool mouseup 1')

	def updatePixelBuf(self):
		gtk.gdk.threads_enter()
		try:
			self.rootWindow = gtk.gdk.get_default_root_window()
			self.sz = self.rootWindow.get_size()
			self.pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,self.sz[0],self.sz[1])
			self.pb = self.pb.get_from_drawable(self.rootWindow,self.rootWindow.get_colormap(),0,0,0,0,self.sz[0],self.sz[1])
		finally:
			gtk.gdk.threads_leave()
			
	def setCropped(self,x,y):
		gtk.gdk.threads_enter()
		try:
			self.subpb=self.pb.subpixbuf(x,y,self.a,self.b)
			self.image.clear()
			self.image.set_from_pixbuf(self.subpb)
		finally:
			gtk.gdk.threads_leave()

	def update_image(self,img):
		gtk.gdk.threads_enter()
		try:
			self.image.clear()
			self.image.set_from_file(img)
		finally:
			gtk.gdk.threads_leave()
		return False

	def set_from_pixbuf(self):
		gtk.gdk.threads_enter()
		try:
			self.image.clear()
			self.image.set_from_pixbuf(self.pb)
		finally:
			gtk.gdk.threads_leave()
		return False

	def run(self):
		while not self.quit:
			#pb = pb.get_from_drawable(self.rootWindow,self.rootWindow.get_colormap(),0,0,0,0,200,200)
			#pb.scale(pb,0,0,200,200,0,0,1,1,gtk.gdk.INTERP_NEAREST)
			#pb=pb.scale_simple(200,200,gtk.gdk.INTERP_NEAREST)
			gobject.idle_add(self.setCropped,self.j,self.i)
			self.j += 1
			if self.j>=(1920-self.a):
				self.j = 0
				self.i += self.b
				if self.i>=(1080-self.b):
					self.i = 0
			#gobject.idle_add(self.set_from_pixbuf)
			#self.moveWindow('Wikipedia, the free encyclopedia - Mozilla Firefox',self.j,0)
			self.moveWindow('Wikipedia',self.j,0)
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

window = gtk.Window(gtk.WINDOW_TOPLEVEL)
window.set_border_width(10)
window.move(1024,0)
#window.connect("delete_event", self.close_application)
#window.connect("event",event)
#window.connect("expose_event",expose)
#window.connect("window_state_event",window_state_changed)
window.connect("destroy",lambda w: gtk.main_quit())
image = gtk.Image()
window.add(image)
window.show_all()

#image.set_from_file('san_francisco.jpg')

t=MyThread(image,XDIM,YDIM)
t.start()
gtk.main()
t.quit=True
