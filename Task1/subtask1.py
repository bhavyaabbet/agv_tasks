import cv2 as cv
import numpy as np

# 1. Initialize Video
video_path = "OPTICAL_FLOW.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

start_time_sec = 12
end_time_sec = 22
cap.set(cv.CAP_PROP_POS_MSEC, start_time_sec * 1000)

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

frame = cv.resize(frame, (800, 450))
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
old_gray = cv.GaussianBlur(old_gray, (5, 5), 0)

feature_params = dict(maxCorners=150, qualityLevel=0.03, minDistance=10, blockSize=7)
corners = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

tracks = []
if corners is not None:
    for p in corners:
        tracks.append([p[0].tolist()]) 

window_size = 25  
half_window = window_size // 2

# TWEAKED: Shortened the trail length to 25 frames
trail_length = 25 

while True:
    if cap.get(cv.CAP_PROP_POS_MSEC) >= end_time_sec * 1000:
        print("Scene complete. Closing video.")
        break

    ret, frame = cap.read()
    if not ret:
        break 
        
    frame = cv.resize(frame, (800, 450))
    new_gray_raw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    new_gray = cv.GaussianBlur(new_gray_raw, (5, 5), 0)

    ix = cv.Sobel(old_gray, cv.CV_64F, 1, 0, ksize=3) / 8.0
    iy = cv.Sobel(old_gray, cv.CV_64F, 0, 1, ksize=3) / 8.0
    it = new_gray.astype(np.float64) - old_gray.astype(np.float64)

    new_tracks = []

    if len(tracks) > 0:
        for track in tracks:
            x, y = track[-1]
            x_int, y_int = int(round(x)), int(round(y))

            if (x_int - half_window < 0 or x_int + half_window >= old_gray.shape[1] or 
                y_int - half_window < 0 or y_int + half_window >= old_gray.shape[0]):
                continue 

            ix_window = ix[y_int-half_window : y_int+half_window+1, x_int-half_window : x_int+half_window+1].flatten()
            iy_window = iy[y_int-half_window : y_int+half_window+1, x_int-half_window : x_int+half_window+1].flatten()
            it_window = it[y_int-half_window : y_int+half_window+1, x_int-half_window : x_int+half_window+1].flatten()

            A = np.vstack((ix_window, iy_window)).T
            b = -it_window.reshape(-1, 1)

            A_T_A = A.T @ A
            
            if np.abs(np.linalg.det(A_T_A)) > 1e-6:
                nu = np.linalg.inv(A_T_A) @ (A.T @ b)
                u, v = nu.flatten()
                
                if np.isnan(u) or np.isnan(v) or np.sqrt(u**2 + v**2) > (window_size * 1.5):
                    continue
                
                new_x, new_y = x + u, y + v
                
                track.append([new_x, new_y])
                if len(track) > trail_length:
                    track.pop(0) 
                    
                new_tracks.append(track)
                
                for j in range(1, len(track)):
                    pt1 = (int(track[j-1][0]), int(track[j-1][1]))
                    pt2 = (int(track[j][0]), int(track[j][1]))
                    cv.line(frame, pt1, pt2, (0, 0, 255), 2)
                    
                cv.circle(frame, (int(new_x), int(new_y)), 3, (0, 255, 0), -1)

    tracks = new_tracks

    if len(tracks) < 40:
        new_corners = cv.goodFeaturesToTrack(new_gray_raw, mask=None, **feature_params)
        if new_corners is not None:
            for p in new_corners:
                tracks.append([p[0].tolist()])

    cv.imshow("Optical Flow - CJ Walking", frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

    old_gray = new_gray.copy()

cap.release()
cv.destroyAllWindows()