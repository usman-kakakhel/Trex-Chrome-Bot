import os
import sys
import math
import random
import pygame as pg

CAPTION = "T-Rex Runner"
SCREEN_SIZE = (1200, 300)
BACKGROUND_COLOR = (255, 255, 255) #white color is used as background
COLOR_KEY = (0, 0, 0) #Black color is used as color keys

DINO_STATES = {'jump':0, 'low1':1, 'low2':2, 'run1':3, 'run2':4, 'dead':5}
JUMP_STATES = {'up':0, 'down':1, 'none':-1}

DINO_RUN1 = None
DINO_RUN2 = None
DINO_JUMP = None
DINO_LOW1 = None
DINO_LOW2 = None
DINO_DEAD = None
CLOUD = None
GROUND1 = None
GROUND2 = None
CACTUS1 = None
CACTUS2 = None
CACTUS3 = None
CACTUS4 = None
CACTUS5 = None
CACTUS = []
CACTUS.append(CACTUS1)
CACTUS.append(CACTUS2)
CACTUS.append(CACTUS3)
CACTUS.append(CACTUS4)
CACTUS.append(CACTUS5)
BIRD1 = None
BIRD2 = None
RESTART = None

dino = None
ground1 = None
ground2 = None
font = None
control = None

cloud_count = 0
bird_count = 0
cactus_count = 0

high_score = 0

global_speed = 8
global_acceleration = 1.01

possible_y_of_birds = [80, 160, 214]

def get_nearest_enemy(dino, cactus_objects, bird_objects):
	global bird_count
	global cactus_count
	
	type = 2
	x = -1
	y = -1
	
	if bird_count == 0:
		if cactus_count == 1:
			type = 1
			for cactus in cactus_objects:
				if cactus.rect.right > dino.rect.right:
					x = cactus.rect.x
					y = cactus.rect.y
		elif cactus_count == 2:
			type = 1
			list = []
			for cactus in cactus_objects:
				list.append(cactus)
			if list[0].rect.right < dino.rect.left:
				x = list[1].rect.x
				y = list[1].rect.y
			elif list[1].rect.right < dino.rect.left:
				x = list[0].rect.x
				y = list[0].rect.y
			else:
				if list[0].rect.x < list[1].rect.x:
					x = list[0].rect.x
					y = list[0].rect.y
				else:
					x = list[1].rect.x
					y = list[1].rect.y
		
	else:
		if cactus_count != 0:
			for cactus in cactus_objects:
				for bird in bird_objects:
					if cactus.rect.x < bird.rect.x and cactus.rect.right > dino.rect.right:
						type = 1
						x = cactus.rect.x
						y = cactus.rect.y
					elif bird.rect.right < dino.rect.left:
						type = 1
						x = cactus.rect.x
						y = cactus.rect.y
					else:
						type = 0
						x = bird.rect.x
						y = bird.rect.y
		else:
			for bird in bird_objects:
				if bird.rect.right > dino.rect.left:
					type = 0
					x = bird.rect.x
					y = bird.rect.y
					
	if type == 2:
		x = 0
		y = 0
	
	return type, x, y

class Dino(pg.sprite.Sprite):
	def __init__(self):
		pg.sprite.Sprite.__init__(self)
		self.image = DINO_RUN1
		self.rect = DINO_RUN1.get_rect()
		self.rect.bottom = 0.98 * SCREEN_SIZE[1]
		self.rect.left = SCREEN_SIZE[0]/15
		self.state = DINO_STATES['run1']
		self.jump_count = 0
		self.jump_direction = JUMP_STATES['none']
		self.frame_count = 0
		self.jump_speed = (global_speed * 0.8)
		self.mid_jump = False
	
	def setRect1(self):
		self.rect = DINO_RUN1.get_rect()
		self.rect.bottom = 0.98 * SCREEN_SIZE[1]
		self.rect.left = SCREEN_SIZE[0]/15
	
	def setRect2(self):
		self.rect = DINO_LOW1.get_rect()
		self.rect.bottom = 0.98 * SCREEN_SIZE[1]
		self.rect.left = SCREEN_SIZE[0]/15
		
	def setRect3(self):
		global global_speed
		self.rect = DINO_JUMP.get_rect()
		self.rect.left = SCREEN_SIZE[0]/15
		if self.jump_direction is JUMP_STATES['up']:
			self.rect.bottom = (0.98 * SCREEN_SIZE[1]) - self.jump_count
			self.jump_count = self.jump_count + self.jump_speed
		elif self.jump_direction is JUMP_STATES['down']:
			self.rect.bottom = (0.98 * SCREEN_SIZE[1]) - self.jump_count
			self.jump_count = self.jump_count - self.jump_speed
		else:
			self.state = DINO_STATES['run1']
			self.image = DINO_RUN1
			self.setRect1()
			
	def jump(self):
		if self.state is not DINO_STATES['dead'] and self.jump_direction is JUMP_STATES['none']:
			self.jump_count = 0
			self.image = DINO_JUMP
			self.setRect3()
			self.state = JUMP_STATES['up']
			self.jump_direction = DINO_STATES['jump']
	
	def low(self):
		if self.state is not DINO_STATES['jump'] and self.state is not DINO_STATES['dead']:
			self.state = DINO_STATES['low1']
			self.image = DINO_LOW1
			self.setRect2()
		elif self.state is DINO_STATES['jump']:
			self.mid_jump = True

	def un_low(self):
		if (self.state is DINO_STATES['low1'] or self.state is DINO_STATES['low2']) and self.state is not DINO_STATES['dead']:
			self.state = DINO_STATES['run1']
			self.image = DINO_RUN1
			self.setRect1()
			
	def draw(self, surface):
		surface.blit(self.image, self.rect)

	def update(self, screen_rect):
		if self.frame_count == 3:
			self.frame_count = 0
			if self.mid_jump:
				self.jump_speed = (global_speed * 0.8 * 3)
			else:
				self.jump_speed = (global_speed * 0.8)
			
			if self.state is DINO_STATES['run1']:
				self.state = DINO_STATES['run2']
				self.image = DINO_RUN2
			elif self.state is DINO_STATES['run2']:
				self.state = DINO_STATES['run1']
				self.image = DINO_RUN1
			elif self.state is DINO_STATES['low1']:
				self.state = DINO_STATES['low2']
				self.image = DINO_LOW2
			elif self.state is DINO_STATES['low2']:
				self.state = DINO_STATES['low1']
				self.image = DINO_LOW1
		else:
			self.frame_count += 1
		if self.state is DINO_STATES['jump']:
			if self.jump_direction is JUMP_STATES['up'] and self.rect.bottom >= (0.35 * SCREEN_SIZE[1]):
				self.jump_direction = JUMP_STATES['up']
				self.setRect3()
			elif self.jump_direction is JUMP_STATES['up'] and self.rect.bottom <= (0.35 * SCREEN_SIZE[1]):
				self.jump_direction = JUMP_STATES['down']
				self.setRect3()
			if self.jump_direction is JUMP_STATES['down'] and self.rect.bottom <= (0.98 * SCREEN_SIZE[1]):
				self.jump_direction = JUMP_STATES['down']
				self.setRect3()
			elif self.jump_direction is JUMP_STATES['down'] and self.rect.bottom >= (0.98 * SCREEN_SIZE[1]):
				self.jump_direction = JUMP_STATES['none']
				self.jump_count = 0
				self.mid_jump = False
				self.setRect3()
				
	def delete(self):
		self.kill()
				
class Cloud(pg.sprite.Sprite):
	def __init__(self):
		pg.sprite.Sprite.__init__(self)
		self.image = CLOUD
		self.rect = self.image.get_rect()
		self.rect.y = random.randrange(SCREEN_SIZE[1]/5,SCREEN_SIZE[1]/2)
		self.rect.x = SCREEN_SIZE[0]
		
	def update(self, screen_rect):
		self.rect.left = self.rect.left - 10
		self.remove(screen_rect)
	
	def remove(self, screen_rect):
		global cloud_count
		if not self.rect.colliderect(screen_rect):
			self.kill()
			cloud_count -= 1
	def delete(self):
		self.kill()
			
class Bird(pg.sprite.Sprite):
	def __init__(self):
		pg.sprite.Sprite.__init__(self)
		self.image = BIRD1
		self.rect = self.image.get_rect()
		self.rect.y = random.choice(possible_y_of_birds)
		self.rect.x = SCREEN_SIZE[0]
		self.flap_count = 0
		
	def update(self, screen_rect):
		global global_speed
		self.rect.left = self.rect.left - global_speed
		if(self.flap_count == 6):
			self.flap_count = 0
			if(self.image == BIRD1):
				self.image = BIRD2
			else:
				self.image = BIRD1
		else:
			self.flap_count += 1
		self.remove(screen_rect)
	
	def remove(self, screen_rect):
		global bird_count
		if not self.rect.colliderect(screen_rect):
			self.kill()
			bird_count -= 1

	def delete(self):
		self.kill()
			
class Cactus(pg.sprite.Sprite):
	def __init__(self):
		pg.sprite.Sprite.__init__(self)
		self.image = random.choice(CACTUS)
		self.rect = self.image.get_rect()
		self.rect.bottom = SCREEN_SIZE[1]*0.98
		self.rect.x = SCREEN_SIZE[0]
		
	def update(self, screen_rect):
		global global_speed
		self.rect.left = self.rect.left - global_speed
		self.remove(screen_rect)
	
	def remove(self, screen_rect):
		global cactus_count
		if not self.rect.colliderect(screen_rect):
			self.kill()
			cactus_count -= 1
			
	def delete(self):
		self.kill()

class Ground(pg.sprite.Sprite):
	def __init__(self, img, lft, id):
		pg.sprite.Sprite.__init__(self)
		self.image = img
		self.rect = img.get_rect()
		self.rect.bottom = 0.98 * SCREEN_SIZE[1]
		self.rect.left = lft
		self.id = id
		
	def update(self, screen_rect):
		global global_speed
		self.rect.left = self.rect.left - global_speed
		if self.id == 1:
			if self.rect.right <= 0:
				self.rect.left = self.rect.width
				self.id = 2
		elif self.id == 2:
			if self.rect.left <= 0:
				self.id = 1
				
	def delete(self):
		self.kill()
		
class HUD():
	def __init__(self):
		global high_score
		pg.sprite.Sprite.__init__(self)
		self.score = 0
		self.text_surface = font.render("Hi " + str(high_score) + " " + str(self.score),False, (0,0,0))
		self.frame_count = 0
		
	def update(self, screen_rect, state):
		global font
		global high_score
		global global_speed
		global global_acceleration
		
		self.text_surface = font.render("Hi " + str(high_score) + " " + str(self.score), False, (0, 0, 0))
		
		if state != DINO_STATES['dead']:
			if self.score % 70 == 0 and self.score != 0:
				global_speed = global_speed * global_acceleration
			
			self.text_surface = font.render("Hi " + str(high_score) + " " + str(self.score), False, (0, 0, 0))
			if self.frame_count == 6:
				self.score += 1
				self.frame_count = 0
			else:
				self.frame_count += 1
		
	def draw(self, surface):
		global font
		global high_score
		surface.blit(self.text_surface,(SCREEN_SIZE[0] - (font.size("Hi " + str(high_score) + " " + str(self.score)))[0],0))

class Control(object):
	def __init__(self):
		self.screen = pg.display.get_surface()
		self.screen_rect = self.screen.get_rect()
		self.done = False
		self.dino = None
		self.clock = pg.time.Clock()
		self.tempcounter = 1
		self.fps = 60.0
		self.keys = pg.key.get_pressed()
		self.objects = pg.sprite.Group()
		self.cactus_objects = pg.sprite.Group()
		self.bird_objects = pg.sprite.Group()
		self.hud = None

	def event_loop(self):
		for event in pg.event.get():
			self.keys = pg.key.get_pressed()
			if event.type == pg.QUIT:
				self.done = True
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_UP:
					dino.jump()
				elif event.key == pg.K_DOWN:
					dino.low()
			elif event.type == pg.KEYUP:
				if event.key == pg.K_DOWN:
					dino.un_low()

	def draw(self):
		self.screen.fill(BACKGROUND_COLOR)
		self.objects.draw(self.screen)
		self.cactus_objects.draw(self.screen)
		self.bird_objects.draw(self.screen)
		self.hud.draw(self.screen)
		self.dino.draw(self.screen)
		
		if self.dino.state == DINO_STATES['dead']:
			restart_rect = RESTART.get_rect()
			restart_rect.x = SCREEN_SIZE[0]/2 - 0.5*restart_rect.width
			restart_rect.y = SCREEN_SIZE[1]/2 - 0.5*restart_rect.height
			self.screen.blit(RESTART, restart_rect)

	def main_loop(self):
		global cloud_count
		global bird_count
		global cactus_count
		global high_score
		while not self.done:
			self.event_loop()
			self.draw()	
			self.clock.tick(self.fps)
			pg.display.flip()
			if not self.paused:
				self.update()
				if self.dino.state != DINO_STATES['dead']:
					if cloud_count < 5 and random.randrange(0, 40) == 15:
						cloud = Cloud()
						cloud_count += 1
						self.objects.add(cloud)
					
					obstacle_present = False
					for x in self.cactus_objects:
						if x.rect.right >= (SCREEN_SIZE[0] * 0.5):
							obstacle_present = True
							break
						
					if bird_count < 1 and random.randrange(0, 15) == 12 and self.hud.score > 500:
						if not obstacle_present:
							bird = Bird()
							bird_count += 1
							self.bird_objects.add(bird)
					
					obstacle_present = False			
					for x in self.bird_objects:
						if x.rect.right >= (SCREEN_SIZE[0] * 0.5):
							obstacle_present = True
							break
						
					if cactus_count == 0  and random.randrange(0, 6) == 4:
						if not obstacle_present:
							cactus = Cactus()
							cactus_count += 1
							self.cactus_objects.add(cactus)
					elif cactus_count == 1 and random.randrange(0, 6) == 4:
						for x in self.cactus_objects:
							if x.rect.right < (SCREEN_SIZE[0] * 0.5):
								if not obstacle_present:
									cactus = Cactus()
									cactus_count += 1
									self.cactus_objects.add(cactus)
				else:
					if self.hud.score > high_score:
						high_score = self.hud.score
	
	def update(self):
		self.hud.update(self.screen_rect, self.dino.state)
		if self.dino.state != DINO_STATES['dead']:
			self.objects.update(self.screen_rect)
			self.cactus_objects.update(self.screen_rect)
			self.bird_objects.update(self.screen_rect)
			self.dino.update(self.screen_rect)
			self.detectCollision()
			
		if self.tempcounter % 10 == 0:
			self.tempcounter = 1
		else:
			self.tempcounter += 1
		
	def shrink(self, rect, shrink_factor):
		w = rect.width
		h = rect.height
		x = rect.x
		y = rect.y
		
		rect.width = rect.width * shrink_factor
		rect.height = rect.height * shrink_factor
		rect.x = rect.x + (w-rect.width)/2
		rect.y = rect.y + (h-rect.height)/2
		
		return rect, w, h, x, y
		
	def expand(self, rect, w, h, x, y):
		rect.x = x
		rect.y = y
		rect.width = w
		rect.height = h
		return rect
		
	def detectCollision(self):
		self.dino.rect, dino_w, dino_h, dino_x, dino_y = self.shrink(self.dino.rect, 0.8)
		for x in self.cactus_objects:
			x.rect, x_w, x_h, x_x, x_y = self.shrink(x.rect, 0.6)
			if self.dino.rect.colliderect(x.rect):
				self.dino.state = DINO_STATES['dead']
			x.rect = self.expand(x.rect, x_w, x_h, x_x, x_y)
				
		for x in self.bird_objects:
			x.rect, x_w, x_h,x_x, x_y = self.shrink(x.rect, 0.7)
			if self.dino.rect.colliderect(x.rect):
				self.dino.state = DINO_STATES['dead']
			x.rect = self.expand(x.rect, x_w, x_h, x_x, x_y)
			
		self.dino.rect = self.expand(self.dino.rect, dino_w, dino_h, dino_x, dino_y)
		
	def delete(self):
		self.dino.delete()
		for x in self.cactus_objects:
			x.delete()
		for x in self.bird_objects:
			x.delete()
		for x in self.objects:
			x.delete()

def load(add):
	temp = pg.image.load(add).convert()
	temp.set_colorkey(COLOR_KEY)
	return temp
	
def restart():
	global control
	global cactus_count
	global bird_count
	global global_speed
	global dino
	global ground1
	global ground2
	
	control.delete()
	dino = None
	ground1 = None
	ground2 = None
	cloud_count = 0
	bird_count = 0
	cactus_count = 0
	global_speed = 8
	
	control = Control()

	hud = HUD()
	control.hud = hud
	ground1 = Ground(GROUND1, 0, 1)
	control.objects.add(ground1)
	ground2 = Ground(GROUND2, GROUND1.get_rect().width, 2)
	control.objects.add(ground2)
	dino = Dino()
	control.dino = dino

	control.main_loop()
	
if __name__ == "__main__":
	os.environ['SDL_VIDEO_CENTERED'] = '1'
	pg.init()
	pg.display.set_caption(CAPTION)
	pg.display.set_mode(SCREEN_SIZE)
	pg.font.init()
	font = pg.font.SysFont('Comic Sans MS', 30)

	DINO_RUN1 = load("sprites/touched/run1.png")
	DINO_RUN2 = load("sprites/touched/run2.png")
	DINO_JUMP = load("sprites/touched/jump.png")
	DINO_LOW1 = load("sprites/touched/low1.png")
	DINO_LOW2 = load("sprites/touched/low2.png")
	DINO_DEAD = load("sprites/touched/dead.png")
	CLOUD = load("sprites/touched/1x-cloud.png")
	GROUND1 = load("sprites/touched/floor-1.png")
	GROUND2 = load("sprites/touched/floor-1.png")
	CACTUS[0] = load("sprites/touched/CACTUS1.png")
	CACTUS[1] = load("sprites/touched/CACTUS2.png")
	CACTUS[2] = load("sprites/touched/CACTUS3.png")
	CACTUS[3] = load("sprites/touched/CACTUS4.png")
	CACTUS[4] = load("sprites/touched/CACTUS5.png")
	BIRD1 = load("sprites/touched/enemy1.png")
	BIRD2 = load("sprites/touched/enemy2.png")
	RESTART = load("sprites/touched/1x-restart.png")

	control = Control()

	hud = HUD()
	control.hud = hud
	ground1 = Ground(GROUND1, 0, 1)
	control.objects.add(ground1)
	ground2 = Ground(GROUND2, GROUND1.get_rect().width, 2)
	control.objects.add(ground2)
	dino = Dino()
	control.dino = dino

	control.main_loop()

	pg.quit()
	sys.exit()