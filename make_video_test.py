import cv2

cap = cv2.VideoCapture('./data/video/ice_shot.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
writer = cv2.VideoWriter('./data/output_video/test.avi', fourcc, 30, (width, height), True)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(33)

    writer.write(frame)

    if key == ord('q'): break

writer.release()
cap.release()
cv2.destroyAllWindows()
