import cv2
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # it loads the pre-trained Haar Cascade classifier for face detection
video_cap = cv2.VideoCapture(0)      # it captures video from the default camera
while True:                            
    ret, video = video_cap.read()       # it reads the video frame by frame
    colr = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)       
    face = face_cap.detectMultiScale(
        colr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in face:             # it draws rectangles around the detected faces
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("video", video)              # it displays the video frame
    if cv2.waitKey(10) == ord("n"):         # if 'n' is pressed, it breaks the loop
        break
video_cap.release()                         # it releases the video capture object