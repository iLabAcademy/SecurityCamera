import cv2

cascPath = "haarcascade_frontalface_default.xml"
api = "http://192.168.1.100:8080/video"
cap = cv2.VideoCapture(api)

faceCascade = cv2.CascadeClassifier(cascPath)
while True:
    # Read the frame
    ret, frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        
        )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Faces found", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
