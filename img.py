import gtk
import time
import threading
import gobject
import gtk.gdk

class MyThread(threading.Thread):
	def __init__(self, image):
		print("init thread")
		super(MyThread, self).__init__()
		self.image = image
		self.quit = False

	def update_image(self,img):
		gtk.gdk.threads_enter()
		try:
			self.image.clear()
			self.image.set_from_file(img)
		finally:
			gtk.gdk.threads_leave()
		return False

	def set_from_pixbuf(self,pb):
		gtk.gdk.threads_enter()
		try:
			self.image.clear()
			self.image.set_from_pixbuf(pb)
		finally:
			gtk.gdk.threads_leave()
		return False

	def run(self):
		while not self.quit:
			w = gtk.gdk.get_default_root_window()
			sz = w.get_size()
			pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
			pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0,sz[0],sz[1])
			#pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0,200,200)
			#pb.scale(pb,0,0,200,200,0,0,1,1,gtk.gdk.INTERP_NEAREST)
			#pb=pb.scale_simple(200,200,gtk.gdk.INTERP_NEAREST)
			pb=pb.subpixbuf(0,0,200,200)
			#pb.save("screenshot.png","png")
			gobject.idle_add(self.set_from_pixbuf,pb)
			time.sleep(5)
#			gobject.idle_add(self.update_image,'san_francisco.jpg')
#			time.sleep(1)
#			gobject.idle_add(self.update_image,"screenshot.png")
#			time.sleep(1)

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

t=MyThread(image)
t.start()
gtk.main()
t.quit=True
