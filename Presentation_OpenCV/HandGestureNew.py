from collections import deque
import numpy as np
import imutils
import cv2
import pyautogui

#low and high range of green color
#greenLRange = (60,100,50)
#greenHRange = (60,255,255)
lower_green = np.array([110,50,50])#green np.array([50,100,100]) # blue(110,50,50 : 130,255,255)
upper_green = np.array([130,255,255])#green np.array([60,255,255])

#initializing varibales for tracking points, frame, change in x and y position & direction
trackPoints = deque(maxlen=32)
frameCount = 0
(dx,dy) = (0 ,0)
direction = ""
(dirx,diry) = (0,0)
#video input
camera = cv2.VideoCapture(0)
#pyautogui.click(100, 100)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (800,480))
while True:
	(ret,frame) = camera.read()
	if ret is False:
		print("No input Video :(")
		break
	resizedFrame = imutils.resize(frame,width = 800)
	#blurredFrame = cv2.GaussianBlur(resizedFrame,(11,11),0)
	hsvFrame = cv2.cvtColor(resizedFrame,cv2.COLOR_BGR2HSV)

	#cv2.imshow("Input Window",frame)
	#cv2.imshow("Resized Window",resizedFrame)
	#cv2.imshow("Blurred Window",blurredFrame)
	#cv2.imshow("HSV Window",hsvFrame)

	maskColor = cv2.inRange(hsvFrame,lower_green,upper_green)
	#maskedObject = cv2.bitwise_and(resizedFrame,resizedFrame, mask= maskColor )
	maskColor = cv2.erode(maskColor,None,iterations=2)
	maskColor = cv2.dilate(maskColor,None,iterations=2)
	#cv2.imshow("Mask Window",maskColor)
	#cv2.imshow("Detected Window",maskColor)

	contours,hierarchy = cv2.findContours(maskColor.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	center = 0
	#cv2.drawContours(resizedFrame,countours,0,(0,255,0),2)


	if(len(contours) > 0):
		c = max(contours,key=cv2.contourArea)
		((x,y),radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if(radius > 10):
			cv2.circle(resizedFrame,(int(x),int(y)),int(radius),(0,255,255),2)
			cv2.circle(resizedFrame,center,5,(0,255.0),-1)
			trackPoints.appendleft(center)

		for i in np.arange(1,len(trackPoints)):
			if trackPoints[i - 1] is None or trackPoints[i] is None:
				continue
			if frameCount >= 10 and i == 1 and trackPoints[-10] is not None:
				dx = trackPoints[-10][0] - trackPoints[i][0]
				dy = trackPoints[-10][i] - trackPoints[i][1]
				(dirx,diry) = ("","")

			if np.abs(dx) > 100:
				if np.sign(dx) == 1 :
					dirx = "RIGHT"
					pyautogui.press('right',interval=5)
				else :
					dirx = "LEFT"
					pass
				#dirx = "RIGHT" if np.sign(dx) == 1 else "LEFT" # if else like ternary operator
				#pyautogui.press('right')

			if np.abs(dy) > 200: 
				if np.sign(dy) == 1: 
					diry = "TOP"
				else :
					diry = "BOTTOM"
					

			if dirx != "" and diry != "":
				direction = "{} -- {}".format(dirx,diry)

			else:
				direction = dirx if dirx != "" else diry

			thickness = int(np.sqrt(32 / float(i+1)) * 2.5)
			cv2.line(resizedFrame,trackPoints[i-1],trackPoints[i],(180,220,100),thickness)

		cv2.putText(resizedFrame,direction,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),3)
		cv2.putText(resizedFrame, "dx: {}, dy: {}".format(dx, dy),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.50, (0,255,0), 2)
	cv2.imshow("Video Window",resizedFrame)
	#out.write(resizedFrame)
	#cv2.imwrite("")
	key = cv2.waitKey(10)
	frameCount = frameCount + 1
	
	if key == ord("q"):
		break

camera.release()
out.release()
cv2.destroyAllWindows()
