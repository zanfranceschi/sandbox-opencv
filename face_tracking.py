import cv2
import numpy as np

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

ret, frame = video_capture.read()

height = np.size(frame, 0)
width = np.size(frame, 1)

center_threshold = (width / 100) * 20
max_distance = (width / 100) * 50

frame_center = width / 2

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mov', fourcc, 20.0, (640,480))

while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	parts = cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	highest = height

	if(type(parts) == np.ndarray):
		if (len(parts) > 1):
			highest = max([part[3] for part in parts])
			parts = [part for part in parts if part[3] == highest]

	#cv2.line(frame, (frame_center, 0), (frame_center, height), (255, 255, 0, 3))
	cv2.line(frame, ((frame_center + center_threshold), 0), ((frame_center + center_threshold), height), (255, 255, 0, 3))
	cv2.line(frame, ((frame_center - center_threshold), 0), ((frame_center - center_threshold), height), (255, 255, 0, 3))
	
	if (type(parts) == tuple):
		cv2.putText(frame, str("PARE (NAO SEI PARA ONDE IR)"), (width/2, height/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	for (x, y, w, h) in parts:
		
		center = x + (w / 2)
		textPos = (center, y - 5)
		
		
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
		
		if (w > max_distance):
			cv2.putText(frame, str("PARE (MUITO PROXIMO)"), textPos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			continue

		
		y = height / 2
		

		detection_x_center = frame_center - center

		if (abs(detection_x_center) <= center_threshold):
			cv2.putText(frame, str("FRENTE"), textPos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		elif (detection_x_center > center_threshold):
			cv2.putText(frame, str("VIRE PARA DIREITA"), textPos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		elif (detection_x_center < center_threshold):
			cv2.putText(frame, str("VIRE PARA ESQUERDA"), textPos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
		top = (center, 0)
		bottom = (center, height)
		
		cv2.line(frame, top, bottom, (255, 255, 255, 1))
		

	out.write(frame)
	cv2.imshow('face tracking', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
out.release()
cv2.destroyAllWindows()