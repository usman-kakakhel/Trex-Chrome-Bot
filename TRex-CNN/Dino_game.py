from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import numpy as np
from PIL import Image
import io
import base64
import cv2

#get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self.browser = webdriver.Chrome(executable_path = "/usr/bin/chromedriver", chrome_options=chrome_options)
        # self.browser.set_window_position(x=-10,y=0)
        self.browser.get('chrome://dino')
        self.browser.execute_script("Runner.config.ACCELERATION=0")
        self.browser.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self.browser.maximize_window()

    def isDead(self):
        return self.browser.execute_script("return Runner.instance_.crashed")
    def isPlaying(self):
        return self.browser.execute_script("return Runner.instance_.playing")
    def restartGame(self):
        self.browser.execute_script("Runner.instance_.restart()")
    def up(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def down(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN) 
    def getScore(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)
    def getHighScore(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.highScore")     
        score = ''.join(score_array[score_array.index(''):])
        return int(score)
    def pause(self):
        return self.browser.execute_script("return Runner.instance_.stop()")
    def play(self):
        return self.browser.execute_script("return Runner.instance_.play()")
    def close(self):
        self.browser.close()
    def process_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:300, :600]
        image = cv2.resize(image, (84,84))
        # image = np.reshape(image, (84,84,1))
        image[0:int(image.shape[1]/7),int(image.shape[0]/2):] = 0
        return  image
    def screenshot(self):
        return self.process_img(np.array(Image.open(io.BytesIO(base64.b64decode(self.browser.execute_script(getbase64Script))))))
    