import cv2
import imageio

cap = cv2.VideoCapture('../data/s1/bbal7s.mpg')
frames = []
ret = True

while ret:
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)


cap.release()

if frames:
    imageio.mimsave('test_animation.gif', frames, fps=10)
else:
    print('No frames were read from the video file')