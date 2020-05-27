import cv2
from Dino_game import Game
import time

class Dino_env:
    def __init__(self):
        self.dino = Game()
        self.img_show = self.show_img()
        self.img_show.__next__()
        time.sleep(1)
    def pause(self):
        self.dino.pause()
    def play(self):
        self.dino.play()
    def getScore(self):
        return self.dino.getScore()
    def getHighScore(self):
        return self.dino.getHighScore()
    def stop(self):
        return self.dino.close()
    def step(self, action):
        reward = 0.1
        dead = False
        if action == 1:
            self.dino.up()
        img = self.dino.screenshot()
        self.img_show.send(img)
        if self.dino.isDead():
            self.dino.restartGame()
            reward = -1
            dead = True
        return img, reward, dead
    
    def show_img(self):
        while True:
            screen = (yield)
            window_title = "game_play"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
            imS = cv2.resize(screen, (800, 400)) 
            cv2.imshow(window_title, screen)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break

# import time
# dino = Game()
# img_show = show_img()
# img_show.__next__()
# time.sleep(1)
# for i in range(500):
#     if i %10 is 0:
#         dino.up()
#     img_show.send(dino.screenshot())