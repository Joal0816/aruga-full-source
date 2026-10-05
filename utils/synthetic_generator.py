import cv2
import numpy as np
import os
import math

def generate_synthetic_fall_video(output_path: str = "assets/synthetic_fall_demo.mp4", duration_sec: int = 14, fps: int = 24) -> str:
    """
    Generates a synthetic demo video showing:
    - Phase 1 (0-3s): Normal walking across a room
    - Phase 2 (3-4.5s): Sudden stumble and rapid fall downwards to floor
    - Phase 3 (4.5-10s): Fallen stationary on the floor (triggers Inactivity Alert)
    - Phase 4 (10-14s): Recovery: stands back up to normal posture
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Ground floor line at y = 410
    floor_y = 410
    
    for f in range(total_frames):
        t = f / fps
        frame = np.ones((height, width, 3), dtype=np.uint8) * 238
        
        # Draw room walls and floor
        cv2.rectangle(frame, (0, floor_y), (width, height), (200, 205, 210), -1)
        cv2.line(frame, (0, floor_y), (width, floor_y), (160, 165, 170), 2)
        # Wall baseboard
        cv2.rectangle(frame, (0, floor_y - 15), (width, floor_y), (180, 185, 190), -1)
        
        # Calculate figure kinematics based on phase
        if t < 3.0:
            # Normal walking
            x = 180 + int(t * 35)
            y_feet = floor_y - 10
            angle_rad = math.sin(t * 8) * 0.05
            body_h = 190
            body_w = 40
            head_h = 24
            is_fallen = False
            
        elif t < 4.2:
            # Stumble & Fall phase (rapid drop & tilt)
            prog = (t - 3.0) / 1.2
            x = 285 + int(prog * 60)
            # Center of gravity drops rapidly
            body_h = int(190 * (1 - prog * 0.65))
            y_feet = floor_y - 10
            angle_rad = prog * (math.pi / 2.1) # tilts to ~85 degrees
            head_h = 24
            is_fallen = prog > 0.7
            
        elif t < 10.0:
            # Lying flat on floor (Stillness / Inactivity)
            x = 345
            angle_rad = math.pi / 2.15 # ~85 degrees
            body_h = 65
            y_feet = floor_y - 5
            head_h = 22
            # Small subtle breathing movement
            y_feet += int(math.sin(t * 3) * 1.5)
            is_fallen = True
            
        else:
            # Recovery phase (getting back up)
            prog = (t - 10.0) / 3.0
            prog = min(1.0, prog)
            x = 345 - int(prog * 20)
            angle_rad = (1.0 - prog) * (math.pi / 2.15)
            body_h = int(65 + prog * 125)
            y_feet = floor_y - 10
            head_h = 24
            is_fallen = prog < 0.3

        # Render humanoid mannequin
        # Hip center
        hip_y = y_feet - int(body_h * 0.45)
        hip_x = x
        
        # Spine vector
        spine_len = body_h * 0.45
        sh_x = int(hip_x + spine_len * math.sin(angle_rad))
        sh_y = int(hip_y - spine_len * math.cos(angle_rad))
        
        # Head
        neck_x = int(sh_x + (head_h * 0.4) * math.sin(angle_rad))
        neck_y = int(sh_y - (head_h * 0.4) * math.cos(angle_rad))
        head_cx = int(neck_x + head_h * math.sin(angle_rad))
        head_cy = int(neck_y - head_h * math.cos(angle_rad))
        
        # Legs
        leg_len = body_h * 0.5
        left_foot_x = int(hip_x - (leg_len * 0.3 if not is_fallen else leg_len * math.sin(angle_rad)))
        left_foot_y = y_feet
        right_foot_x = int(hip_x + (leg_len * 0.3 if not is_fallen else -leg_len * 0.7 * math.sin(angle_rad)))
        right_foot_y = y_feet

        # Arms
        arm_len = body_h * 0.35
        left_hand_x = int(sh_x - arm_len * 0.5)
        left_hand_y = int(sh_y + arm_len * 0.7)
        right_hand_x = int(sh_x + arm_len * 0.5)
        right_hand_y = int(sh_y + arm_len * 0.7)

        # Draw skin/clothing
        mannequin_color = (60, 60, 75)
        skin_color = (180, 205, 235)
        
        # Draw Head
        cv2.circle(frame, (head_cx, head_cy), int(head_h * 0.8), skin_color, -1)
        cv2.circle(frame, (head_cx, head_cy), int(head_h * 0.8), (40, 40, 50), 2)
        
        # Draw Torso
        cv2.line(frame, (hip_x, hip_y), (sh_x, sh_y), mannequin_color, 18)
        
        # Draw Legs
        cv2.line(frame, (hip_x - 6, hip_y), (left_foot_x, left_foot_y), (45, 55, 65), 10)
        cv2.line(frame, (hip_x + 6, hip_y), (right_foot_x, right_foot_y), (45, 55, 65), 10)
        
        # Draw Arms
        cv2.line(frame, (sh_x, sh_y), (left_hand_x, left_hand_y), mannequin_color, 8)
        cv2.line(frame, (sh_x, sh_y), (right_hand_x, right_hand_y), mannequin_color, 8)

        # Text banner on synthetic frame
        phase_text = "Phase 1: Normal Walking"
        if 3.0 <= t < 4.2:
            phase_text = "Phase 2: Sudden Stumble & Fall"
        elif 4.2 <= t < 10.0:
            phase_text = f"Phase 3: Fallen on Floor (Inactivity: {t-4.2:.1f}s)"
        elif t >= 10.0:
            phase_text = "Phase 4: Recovery & Standing Up"
            
        cv2.putText(frame, f"[Test Simulation] {phase_text}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 60), 2)
        out.write(frame)
        
    out.release()
    return output_path

if __name__ == "__main__":
    generate_synthetic_fall_video()
