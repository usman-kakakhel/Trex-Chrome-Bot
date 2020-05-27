import dino_env as env
import time

myEnv = env.DinoEnv()

death_status = False
i = 0

for j in range(0,5):
	while not death_status:
		if i % 10 == 0:
			result = env.step(0)
			print(result)
		else:
			result = env.step(2)
			print(result)
		env.render()
		death_status = result[2]
		i += 1
	death_status = False
	myEnv.start()
myEnv.stop()



