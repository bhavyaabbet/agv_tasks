import pybullet as p
import cv2 as cv
import numpy as np
import time
from simulation_setup import setup_simulation

# --- CONFIGURATION ---
IMG_WIDTH = 320
IMG_HEIGHT = 240
TARGET_SPEED = 6.0    
FRAME_SKIP = 3        
WINDOW_SIZE = 21      
HALF_W = WINDOW_SIZE // 2

def get_car_view(car_id):
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward_vec = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
    up_vec = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
    camera_eye = pos + forward_vec * 1.2 + up_vec * 0.3
    target_pos = camera_eye + forward_vec * 5.0
    view_matrix = p.computeViewMatrix(camera_eye, target_pos, up_vec)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(IMG_WIDTH)/IMG_HEIGHT, nearVal=0.1, farVal=100.0)
    _, _, rgb_img, _, _ = p.getCameraImage(width=IMG_WIDTH, height=IMG_HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return np.reshape(rgb_img, (IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3].astype(np.uint8)

def custom_lucas_kanade(prev_gray, curr_gray, points):
    flow_vectors, valid_points = [], []
    Ix = cv.Sobel(prev_gray, cv.CV_64F, 1, 0, ksize=3) / 8.0
    Iy = cv.Sobel(prev_gray, cv.CV_64F, 0, 1, ksize=3) / 8.0
    It = curr_gray.astype(np.float64) - prev_gray.astype(np.float64)
    
    for pt in points:
        x, y = int(pt[0][0]), int(pt[0][1])
        if x - HALF_W < 0 or x + HALF_W >= IMG_WIDTH or y - HALF_W < 0 or y + HALF_W >= IMG_HEIGHT: continue
        
        Ix_w = Ix[y-HALF_W : y+HALF_W+1, x-HALF_W : x+HALF_W+1].flatten()
        Iy_w = Iy[y-HALF_W : y+HALF_W+1, x-HALF_W : x+HALF_W+1].flatten()
        It_w = It[y-HALF_W : y+HALF_W+1, x-HALF_W : x+HALF_W+1].flatten()
        
        A = np.vstack((Ix_w, Iy_w)).T
        ATA = A.T @ A
        
        if np.linalg.det(ATA) > 1e-4: 
            nu = np.linalg.pinv(ATA) @ (A.T @ -It_w.reshape(-1, 1))
            valid_points.append([x, y])
            flow_vectors.append(nu.flatten())
            
    return np.array(valid_points), np.array(flow_vectors)

def run_autonomous_agent():
    car_id, steer_j, motor_j = setup_simulation(gui=True)
    
    prev_frame = get_car_view(car_id)
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_RGB2GRAY)
    
    feature_params = dict(maxCorners=30, qualityLevel=0.1, minDistance=10, blockSize=7)
    features = None 
    
    current_steering = 0.0
    frame_count = 0
    center_x = IMG_WIDTH // 2
    memory_target_x = center_x

    while True:
        try:
            for _ in range(FRAME_SKIP):
                for j in motor_j: p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=TARGET_SPEED, force=500)
                p.stepSimulation()
            
            curr_frame = get_car_view(car_id)
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_RGB2GRAY)
            hsv = cv.cvtColor(curr_frame, cv.COLOR_RGB2HSV)
            output_viz = curr_frame.copy()
            frame_count += 1
            
            # --- HORSE BLINDERS (Visual) ---
            IGNORE_MARGIN = 40 # Narrowed to ensure we fully clear the obstacle
            
            overlay = output_viz.copy()
            cv.rectangle(overlay, (0, 0), (IGNORE_MARGIN, IMG_HEIGHT), (50, 50, 50), -1)
            cv.rectangle(overlay, (IMG_WIDTH - IGNORE_MARGIN, 0), (IMG_WIDTH, IMG_HEIGHT), (50, 50, 50), -1)
            cv.addWeighted(overlay, 0.5, output_viz, 0.5, 0, output_viz)
            
            # Yellow Mask
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
            
            if frame_count % 8 == 0 or features is None or len(features) < 3:
                features = cv.goodFeaturesToTrack(curr_gray, mask=yellow_mask, **feature_params)
                prev_gray = curr_gray.copy()
                continue
                
            valid_points, flow_vectors = [], []
            if features is not None:
                valid_points, flow_vectors = custom_lucas_kanade(cv.GaussianBlur(prev_gray, (5,5), 0), cv.GaussianBlur(curr_gray, (5,5), 0), features)
            
            # --- 1. Attractive Force (Blue Wall & Decaying Memory) ---
            attractive_steer = 0.0
            blue_mask = cv.inRange(hsv, np.array([100, 80, 50]), np.array([140, 255, 255]))
            M = cv.moments(blue_mask)
            
            if M["m00"] > 0:
                target_x = int(M["m10"] / M["m00"])
                memory_target_x = target_x 
                cv.circle(output_viz, (target_x, 50), 10, (255, 0, 0), -1)
            else:
                target_x = memory_target_x
                memory_target_x = int(0.95 * memory_target_x + 0.05 * center_x)
                cv.circle(output_viz, (target_x, 50), 10, (150, 150, 150), 2)
                
            attractive_steer = (center_x - target_x) * 0.005 

            # --- 2. GRASS REPULSION ---
            ROAD_HALF_WIDTH = 110 
            safe_target_x = np.clip(target_x, 50, IMG_WIDTH - 50)
            grass_left = safe_target_x - ROAD_HALF_WIDTH
            grass_right = safe_target_x + ROAD_HALF_WIDTH
            
            cv.line(output_viz, (int(grass_left), 0), (int(grass_left), IMG_HEIGHT), (0, 0, 255), 2)
            cv.line(output_viz, (int(grass_right), 0), (int(grass_right), IMG_HEIGHT), (0, 0, 255), 2)

            boundary_steer = 0.0
            dist_to_left = center_x - grass_left
            dist_to_right = grass_right - center_x
            DANGER_ZONE = 45 
            
            if dist_to_left < DANGER_ZONE:
                boundary_steer = 0.015 * (DANGER_ZONE - dist_to_left)
            elif dist_to_right < DANGER_ZONE:
                boundary_steer = -0.015 * (DANGER_ZONE - dist_to_right)
            boundary_steer = np.clip(boundary_steer, -0.4, 0.4)

            # --- 3. Repulsive Force (Boxes WITH BLINDERS & EXPONENTIALIZATION) ---
            repulsive_steer = 0.0
            left_flow = 0.0
            right_flow = 0.0
            
            EXP_K = 0.55 # Starts the exponential curve earlier for a smoother dodge
            
            if len(valid_points) > 0:
                for (x, y), (u, v) in zip(valid_points, flow_vectors):
                    if x < IGNORE_MARGIN or x > IMG_WIDTH - IGNORE_MARGIN:
                        continue
                        
                    mag = np.sqrt(u**2 + v**2)
                    if mag > 0.5: 
                        cv.line(output_viz, (x, y), (int(x + u*2), int(y + v*2)), (0, 255, 0), 2)
                        
                        exp_weight = np.exp(EXP_K * mag) - 1.0
                        
                        if x < center_x: 
                            left_flow += exp_weight
                        else: 
                            right_flow += exp_weight
                
                STEER_GAIN = 0.0035 # Increased base multiplier to give the obstacle a wider berth
                repulsive_steer = (right_flow - left_flow) * STEER_GAIN 
                repulsive_steer = np.clip(repulsive_steer, -0.6, 0.6) 

            # --- 4. Subsumption ---
            if abs(repulsive_steer) > 0.1:
                attractive_steer = attractive_steer * 0.1 

            # --- 5. Combine and Steer ---
            target_steering = attractive_steer + repulsive_steer + boundary_steer
            target_steering = np.clip(target_steering, -0.6, 0.6)
            
            current_steering = 0.6 * current_steering + 0.4 * target_steering

            for j in steer_j: p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=current_steering)

            features = valid_points.reshape(-1, 1, 2) if len(valid_points) > 0 else None
            prev_gray = curr_gray.copy()
            
            cv.putText(output_viz, f"Total Steer: {current_steering:.2f}", (10, 20), 1, 1.2, (255,255,255), 2)
            cv.putText(output_viz, f"Grass Protect: {boundary_steer:.2f}", (10, 45), 1, 1.2, (0,0,255), 2)
            cv.putText(output_viz, f"Push (Box): {repulsive_steer:.2f}", (10, 70), 1, 1.2, (0,255,0), 2)
            
            mask_viz = cv.cvtColor(yellow_mask, cv.COLOR_GRAY2BGR)
            mask_viz = cv.resize(mask_viz, (80, 60))
            output_viz[0:60, IMG_WIDTH-80:IMG_WIDTH] = mask_viz

            cv.imshow("The Ultimate APF", cv.cvtColor(output_viz, cv.COLOR_RGB2BGR))
            
            if cv.waitKey(1) & 0xFF == ord('q'): break

        except KeyboardInterrupt:
            break

    p.disconnect()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_autonomous_agent()