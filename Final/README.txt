Links to video demos:
	
	https://photos.app.goo.gl/5aEUzZFGT5ASYBFC9

Dependencies:

	pip install ipython
	pip install matplotlib
	pip install odrive
	pip install cloudpickle
	pip install pydy
	pip install numpy
	pip install pyglet
	pip install pywavefront


Recording a path:
	
	python handGuidedPathV4.py
		not possible without access to physical robot


Serialization of "full_EOM_func_NO_GRAVITY":

	python roboticArmNoGravity.py
		if done correctly this should open a browser window with a simple visualization of the robot and save the EOM function object to full_EOM_func_NO_GRAVITY.txt

		This is done so that the EOM function does not have to be recalculated each time the program is run


Serialization of "robot_inertia_func":

	python roboticArmEndpointForces.py
		if done correctly this should open a browser window with a simple visualization of the robot and save the end effector inertia function object to robot_inertia_func.txt

		This is done so that the endpoint inertia function does not have to be recalculated each time the program is run


Force visualization from path data:

	python lookAtForces.py
		uncomment line 17 or 18 depending on which path is desired
			armPath1.txt = standing to the left of robot base
			armPath2.txt = standing to the right of robot base


Simulation:
	
	python main.py
		uncomment lines 13 and 14 if using a computer with a dedicated graphics card
		uncomment line 17 if using a computer without a dedicated graphics card

		uncomment lines 23 or 24 depending on which path is desired
			armPath1.txt = standing to the left of robot base
			armPath2.txt = standing to the right of robot base

		W/S to move camera forwards/ backwards
		A/D to rotate scene CW/CCW