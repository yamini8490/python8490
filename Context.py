class Context(object):

	def __init__(self,name):
		self.lifespan = 2
		self.name = name
		self.active = False

	def activate_context(self):
		self.active = True

	def deactivate_context(self):
		self.active = False

		def decrease_lifespan(self):
			self.lifespan -= 1
			if self.lifespan==0:
				self.deactivate_context()

class FirstGreeting(Context): 

	def __init__(self):
		self.lifespan = 1
		self.name = 'FirstGreeting'
		self.active = True

class IntentComplete(Context):

	def __init__(self):
		self.lifespan = 1
		self.name = 'IntentComplete'
		self.active = True
