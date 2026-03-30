import cv2 as cv
import numpy as np

def manual_horn_schunck(prev_img, curr_img, alpha=15.0, iterations=50):
    """
    Computes Dense Optical Flow manually using the Horn-Schunck iterative method.
    """
    # Convert images to float64 to prevent uint8 overflow during subtraction
    I1 = prev_img.astype(np.float64)
    I2 = curr_img.astype(np.float64)

    # 1. Calculate Image Gradients (Ix, Iy, It)
    Ix = cv.Sobel(I1, cv.CV_64F, 1, 0, ksize=3) / 8.0
    Iy = cv.Sobel(I1, cv.CV_64F, 0, 1, ksize=3) / 8.0
    It = I2 - I1

    # 2. Setup the Laplacian Averaging Kernel
    # This matrix calculates the local average surrounding a pixel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,  0,   1/6],
                       [1/12, 1/6, 1/12]], dtype=np.float64)

    # Initialize flow vectors u and v to zero
    u = np.zeros_like(I1)
    v = np.zeros_like(I1)

    # Precompute the denominator to save processing time inside the loop
    denominator = (alpha ** 2) + (Ix ** 2) + (Iy ** 2)

    # 3. Iteratively solve for u and v
    for _ in range(iterations):
        # Calculate local averages
        u_avg = cv.filter2D(u, -1, kernel)
        v_avg = cv.filter2D(v, -1, kernel)

        # Horn-Schunck update formula
        P = (Ix * u_avg + Iy * v_avg + It) / denominator
        u = u_avg - (Ix * P)
        v = v_avg - (Iy * P)

    return u, v

# ==========================================
# Main Video Processing Loop
# ==========================================

video_path = "OPTICAL_FLOW.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Target the clean CJ walking scene
start_time_sec = 12
end_time_sec = 22
cap.set(cv.CAP_PROP_POS_MSEC, start_time_sec * 1000)

ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

# Setup resolutions
DISPLAY_SIZE = (800, 450)
CALC_SIZE = (200, 112) # 1/4th scale for mathematical performance
SCALE_FACTOR = DISPLAY_SIZE[0] / CALC_SIZE[0]

frame1 = cv.resize(frame1, DISPLAY_SIZE)
prev_gray = cv.cvtColor(cv.resize(frame1, CALC_SIZE), cv.COLOR_BGR2GRAY)

# Setup HSV image for coloring the flow
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255 

print("Running Manual Horn-Schunck Dense Flow...")
print("Note: This is computationally heavy and will play slower than normal.")

while True:
    if cap.get(cv.CAP_PROP_POS_MSEC) >= end_time_sec * 1000:
        break

    ret, frame2 = cap.read()
    if not ret:
        break
        
    frame2 = cv.resize(frame2, DISPLAY_SIZE)
    
    # Downscale for math processing to maintain performance
    curr_gray = cv.cvtColor(cv.resize(frame2, CALC_SIZE), cv.COLOR_BGR2GRAY)
    
    # Smooth the images slightly to reduce noise before the derivative math
    blur_prev = cv.GaussianBlur(prev_gray, (3, 3), 0)
    blur_curr = cv.GaussianBlur(curr_gray, (3, 3), 0)

    # --- EXECUTE MANUAL DENSE FLOW ---
    u, v = manual_horn_schunck(blur_prev, blur_curr, alpha=15.0, iterations=40)

    # Scale the resulting flow vectors back up to the display resolution
    u_display = cv.resize(u, DISPLAY_SIZE) * SCALE_FACTOR
    v_display = cv.resize(v, DISPLAY_SIZE) * SCALE_FACTOR

    # --- VISUALIZATION ---
    # Convert vectors to magnitude and angle
    mag, ang = cv.cartToPolar(u_display, v_display)

    # Map angle to Hue (Color direction)
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Map magnitude to Value (Brightness speed)
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    blended_view = cv.addWeighted(frame2, 0.5, bgr_flow, 0.8, 0)
    
    cv.imshow("Manual Horn-Schunck Dense Flow", blended_view)

    # waitKey(1) forces it to play as fast as your CPU can calculate the math
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray.copy()

cap.release()
cv.destroyAllWindows()