#!/usr/bin/env python3
"""
Enhanced Cricket Bowling Biomechanics Analysis with Front Foot Contact Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.python.solutions.pose import PoseLandmark

def rename_columns(landmarks_df):
    """Map Mediapipe indices to human-readable names."""
    rename_map = {}
    for idx, lm in enumerate(PoseLandmark):
        rename_map[f"{idx}_x"] = f"{lm.name}_x"
        rename_map[f"{idx}_y"] = f"{lm.name}_y"
        rename_map[f"{idx}_z"] = f"{lm.name}_z"
        rename_map[f"{idx}_v"] = f"{lm.name}_v"
    return landmarks_df.rename(columns=rename_map)

class EnhancedCricketBiomechanicsExtractor:
    def __init__(self, landmarks_df):
        self.df = landmarks_df
        self.features = []
        self.ffc_frame = None
        self.ball_release_frame = None
        
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate angle between three points a, b, c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    
    def detect_front_foot_contact(self):
        """
        Detect Front Foot Contact (FFC) frame based on:
        1. Minimum distance between front foot and ground
        2. Sudden change in front ankle velocity
        3. Maximum hip-shoulder separation (usually occurs around FFC)
        """
        print("Detecting Front Foot Contact (FFC)...")
        
        # Calculate front ankle Y position (higher Y = closer to ground in normalized coords)
        front_ankle_y = []
        hip_shoulder_separation = []
        
        for idx, row in self.df.iterrows():
            try:
                # Assuming right foot is the front foot for right-handed bowler
                ankle_y = row.get("RIGHT_ANKLE_y", np.nan)
                front_ankle_y.append(ankle_y)
                
                # Calculate hip-shoulder separation for this frame
                separation = self.calculate_angle(
                    (row.get("LEFT_HIP_x", 0), row.get("LEFT_HIP_y", 0)),
                    (row.get("RIGHT_SHOULDER_x", 0), row.get("RIGHT_SHOULDER_y", 0)),
                    (row.get("RIGHT_HIP_x", 0), row.get("RIGHT_HIP_y", 0))
                )
                hip_shoulder_separation.append(separation)
                
            except Exception:
                front_ankle_y.append(np.nan)
                hip_shoulder_separation.append(np.nan)
        
        # Find FFC as the frame with maximum ankle Y (closest to ground)
        front_ankle_y = np.array(front_ankle_y)
        valid_indices = ~np.isnan(front_ankle_y)
        
        if valid_indices.any():
            # FFC is typically when front ankle is closest to ground (max Y value)
            ffc_candidates = np.where(valid_indices)[0]
            ffc_frame_idx = ffc_candidates[np.argmax(front_ankle_y[valid_indices])]
            self.ffc_frame = self.df.iloc[ffc_frame_idx]['frame']
            
            # Ball release is typically 3-8 frames after FFC
            release_offset = min(8, len(self.df) - ffc_frame_idx - 1)
            if release_offset > 0:
                self.ball_release_frame = self.df.iloc[ffc_frame_idx + release_offset]['frame']
            else:
                self.ball_release_frame = self.ffc_frame
                
            print(f"✅ FFC detected at frame {self.ffc_frame}")
            print(f"✅ Ball release estimated at frame {self.ball_release_frame}")
        else:
            print("❌ Could not detect FFC - using middle frame as approximation")
            mid_frame = len(self.df) // 2
            self.ffc_frame = self.df.iloc[mid_frame]['frame']
            self.ball_release_frame = self.df.iloc[min(mid_frame + 5, len(self.df)-1)]['frame']
    
    def extract_features_post_ffc(self):
        """Extract biomechanical features with proper timing considerations."""
        if self.ffc_frame is None:
            self.detect_front_foot_contact()
        
        print(f"Extracting features post-FFC (frame {self.ffc_frame})...")
        
        # Get frames from FFC onwards for analysis
        post_ffc_frames = self.df[self.df['frame'] >= self.ffc_frame]
        
        if post_ffc_frames.empty:
            print("❌ No frames found post-FFC")
            return pd.DataFrame()
        
        print(f"Analyzing {len(post_ffc_frames)} frames post-FFC...")
        
        # Calculate front knee angles for all post-FFC frames to find key phases
        front_knee_angles = []
        back_knee_angles = []
        frame_features = []
        
        for idx, row in post_ffc_frames.iterrows():
            try:
                # Front knee angle (right leg assumed to be front leg)
                front_knee_angle = self.calculate_angle(
                    (row["RIGHT_HIP_x"], row["RIGHT_HIP_y"]),
                    (row["RIGHT_KNEE_x"], row["RIGHT_KNEE_y"]),
                    (row["RIGHT_ANKLE_x"], row["RIGHT_ANKLE_y"])
                )
                front_knee_angles.append((row["frame"], front_knee_angle))
                
                # Other angles for comprehensive analysis
                elbow_angle = self.calculate_angle(
                    (row["RIGHT_SHOULDER_x"], row["RIGHT_SHOULDER_y"]),
                    (row["RIGHT_ELBOW_x"], row["RIGHT_ELBOW_y"]),
                    (row["RIGHT_WRIST_x"], row["RIGHT_WRIST_y"])
                )
                
                # Back knee angle (left leg - back leg flexion)
                back_knee_angle = self.calculate_angle(
                    (row["LEFT_HIP_x"], row["LEFT_HIP_y"]),
                    (row["LEFT_KNEE_x"], row["LEFT_KNEE_y"]),
                    (row["LEFT_ANKLE_x"], row["LEFT_ANKLE_y"])
                )
                back_knee_angles.append((row["frame"], back_knee_angle))
                
                trunk_angle = self.calculate_angle(
                    (row["LEFT_SHOULDER_x"], row["LEFT_SHOULDER_y"]),
                    (row["RIGHT_HIP_x"], row["RIGHT_HIP_y"]),
                    (row["RIGHT_KNEE_x"], row["RIGHT_KNEE_y"])
                )
                
                separation = self.calculate_angle(
                    (row["LEFT_HIP_x"], row["LEFT_HIP_y"]),
                    (row["RIGHT_SHOULDER_x"], row["RIGHT_SHOULDER_y"]),
                    (row["RIGHT_HIP_x"], row["RIGHT_HIP_y"])
                )
                
                # Additional comprehensive biomechanical parameters
                
                # Shoulder alignment (left-right shoulder level)
                shoulder_alignment = abs(row["LEFT_SHOULDER_y"] - row["RIGHT_SHOULDER_y"]) * 57.2958  # Convert to degrees approximation
                
                # Bowling arm angle (shoulder-elbow-wrist)
                bowling_arm_angle = self.calculate_angle(
                    (row["RIGHT_SHOULDER_x"], row["RIGHT_SHOULDER_y"]),
                    (row["RIGHT_ELBOW_x"], row["RIGHT_ELBOW_y"]),
                    (row["RIGHT_WRIST_x"], row["RIGHT_WRIST_y"])
                )
                
                # Non-bowling arm position (left shoulder-elbow-wrist)
                non_bowling_arm_angle = self.calculate_angle(
                    (row["LEFT_SHOULDER_x"], row["LEFT_SHOULDER_y"]),
                    (row["LEFT_ELBOW_x"], row["LEFT_ELBOW_y"]),
                    (row["LEFT_WRIST_x"], row["LEFT_WRIST_y"])
                )
                
                # Head position (relative to shoulder line)
                head_tilt = self.calculate_angle(
                    (row["LEFT_SHOULDER_x"], row["LEFT_SHOULDER_y"]),
                    (row["NOSE_x"], row["NOSE_y"]),
                    (row["RIGHT_SHOULDER_x"], row["RIGHT_SHOULDER_y"])
                )
                
                # Pelvis rotation (hip line angle)
                pelvis_rotation = self.calculate_angle(
                    (row["LEFT_HIP_x"], row["LEFT_HIP_y"]),
                    (row["RIGHT_HIP_x"], row["RIGHT_HIP_y"]),
                    (row["RIGHT_HIP_x"] + 10, row["RIGHT_HIP_y"])  # Reference horizontal line
                )
                
                # Front ankle angle (shin-foot angle)
                front_ankle_angle = self.calculate_angle(
                    (row["RIGHT_KNEE_x"], row["RIGHT_KNEE_y"]),
                    (row["RIGHT_ANKLE_x"], row["RIGHT_ANKLE_y"]),
                    (row["RIGHT_FOOT_INDEX_x"], row["RIGHT_FOOT_INDEX_y"])
                )
                
                # Back ankle angle
                back_ankle_angle = self.calculate_angle(
                    (row["LEFT_KNEE_x"], row["LEFT_KNEE_y"]),
                    (row["LEFT_ANKLE_x"], row["LEFT_ANKLE_y"]),
                    (row["LEFT_FOOT_INDEX_x"], row["LEFT_FOOT_INDEX_y"])
                )
                
                # Spine angle (hip-shoulder alignment)
                spine_angle = self.calculate_angle(
                    ((row["LEFT_HIP_x"] + row["RIGHT_HIP_x"])/2, (row["LEFT_HIP_y"] + row["RIGHT_HIP_y"])/2),
                    ((row["LEFT_SHOULDER_x"] + row["RIGHT_SHOULDER_x"])/2, (row["LEFT_SHOULDER_y"] + row["RIGHT_SHOULDER_y"])/2),
                    ((row["LEFT_SHOULDER_x"] + row["RIGHT_SHOULDER_x"])/2 + 10, (row["LEFT_SHOULDER_y"] + row["RIGHT_SHOULDER_y"])/2)
                )
                
                frame_features.append({
                    "frame": row["frame"],
                    "phase": "post_ffc",
                    "is_ball_release": row["frame"] == self.ball_release_frame,
                    "is_ffc": row["frame"] == self.ffc_frame,
                    "Elbow Angle": elbow_angle,
                    "Front Knee Angle": front_knee_angle,
                    "Back Knee Angle": back_knee_angle,
                    "Trunk Lean": trunk_angle,
                    "Hip-Shoulder Separation": separation,
                    "Shoulder Alignment": shoulder_alignment,
                    "Bowling Arm Angle": bowling_arm_angle,
                    "Non-Bowling Arm Angle": non_bowling_arm_angle,
                    "Head Position": head_tilt,
                    "Pelvis Rotation": pelvis_rotation,
                    "Front Ankle Angle": front_ankle_angle,
                    "Back Ankle Angle": back_ankle_angle,
                    "Spine Angle": spine_angle
                })
                
            except Exception as e:
                print(f"Error processing frame {row['frame']}: {e}")
                front_knee_angles.append((row["frame"], np.nan))
                back_knee_angles.append((row["frame"], np.nan))
                frame_features.append({
                    "frame": row["frame"],
                    "phase": "post_ffc",
                    "is_ball_release": False,
                    "is_ffc": False,
                    "Elbow Angle": np.nan,
                    "Front Knee Angle": np.nan,
                    "Back Knee Angle": np.nan,
                    "Trunk Lean": np.nan,
                    "Hip-Shoulder Separation": np.nan,
                    "Shoulder Alignment": np.nan,
                    "Bowling Arm Angle": np.nan,
                    "Non-Bowling Arm Angle": np.nan,
                    "Head Position": np.nan,
                    "Pelvis Rotation": np.nan,
                    "Front Ankle Angle": np.nan,
                    "Back Ankle Angle": np.nan,
                    "Spine Angle": np.nan
                })
        
        return pd.DataFrame(frame_features), front_knee_angles, back_knee_angles
    
    def analyze_front_knee_phases(self, front_knee_angles):
        """Analyze front knee angle at key phases: FFC, max flexion, max extension."""
        if not front_knee_angles:
            return None, None, None, None, None, None, None
        
        # Remove NaN values
        valid_angles = [(frame, angle) for frame, angle in front_knee_angles if not np.isnan(angle)]
        
        if not valid_angles:
            return None, None, None, None, None, None, None
        
        frames, angles = zip(*valid_angles)
        angles = np.array(angles)
        frames = np.array(frames)
        
        # Find key phases
        # 1. At FFC (first frame)
        ffc_angle = angles[0] if len(angles) > 0 else np.nan
        ffc_frame = frames[0] if len(frames) > 0 else np.nan
        
        # 2. Maximum flexion (minimum angle - most bent)
        max_flex_idx = np.argmin(angles)
        max_flexion_angle = angles[max_flex_idx]
        max_flexion_frame = frames[max_flex_idx]
        
        # 3. Maximum extension (maximum angle - most straight)
        max_ext_idx = np.argmax(angles)
        max_extension_angle = angles[max_ext_idx]
        max_extension_frame = frames[max_ext_idx]
        
        # 4. Calculate knee angle deviation (FFC to max extension)
        knee_angle_deviation = max_extension_angle - ffc_angle if not np.isnan(ffc_angle) and not np.isnan(max_extension_angle) else np.nan
        
        print(f"Front Knee Analysis:")
        print(f"  At FFC (frame {ffc_frame}): {ffc_angle:.2f}°")
        print(f"  Max Flexion (frame {max_flexion_frame}): {max_flexion_angle:.2f}°")
        print(f"  Max Extension (frame {max_extension_frame}): {max_extension_angle:.2f}°")
        print(f"  Knee Angle Deviation (FFC→Max Extension): {knee_angle_deviation:.2f}°")
        
        return ffc_angle, max_flexion_angle, max_extension_angle, ffc_frame, max_flexion_frame, max_extension_frame, knee_angle_deviation

    def analyze_back_knee_phases(self, back_knee_angles_list):
        """
        Analyze back knee at three key phases:
        a. at back foot contact (initial contact)
        b. at maximum flexion
        c. at maximum extension prior to back foot lift off
        """
        if not back_knee_angles_list:
            print("❌ No back knee angle data available")
            return None, None, None, None, None, None, None
        
        # Remove NaN values
        valid_angles = [(frame, angle) for frame, angle in back_knee_angles_list if not np.isnan(angle)]
        
        if not valid_angles:
            print("❌ All back knee angles are NaN")
            return None, None, None, None, None, None, None
        
        frames, angles = zip(*valid_angles)
        angles = np.array(angles)
        frames = np.array(frames)
        
        # 1. Back foot contact - assume it's at the beginning of the analysis window
        bfc_frame = frames[0]  # Back foot contact at start
        bfc_angle = angles[0]
        
        # 2. Maximum flexion (minimum angle - most bent)
        max_flex_idx = np.argmin(angles)
        max_flexion_angle = angles[max_flex_idx]
        max_flexion_frame = frames[max_flex_idx]
        
        # 3. Maximum extension (maximum angle before lift-off)
        # Find the maximum extension after flexion but before lift-off
        # Look for maximum extension in the latter part of the sequence
        latter_half_start = len(angles) // 2
        max_ext_idx_relative = np.argmax(angles[latter_half_start:])
        max_ext_idx = latter_half_start + max_ext_idx_relative
        max_extension_angle = angles[max_ext_idx]
        max_extension_frame = frames[max_ext_idx]
        
        # 4. Calculate back knee angle deviation (BFC to max extension)
        back_knee_deviation = max_extension_angle - bfc_angle if not np.isnan(bfc_angle) and not np.isnan(max_extension_angle) else np.nan
        
        print(f"Back Knee Analysis:")
        print(f"  At Back Foot Contact (frame {bfc_frame}): {bfc_angle:.2f}°")
        print(f"  Max Flexion (frame {max_flexion_frame}): {max_flexion_angle:.2f}°")
        print(f"  Max Extension (frame {max_extension_frame}): {max_extension_angle:.2f}°")
        print(f"  Back Knee Deviation (BFC→Max Extension): {back_knee_deviation:.2f}°")
        
        return bfc_angle, max_flexion_angle, max_extension_angle, bfc_frame, max_flexion_frame, max_extension_frame, back_knee_deviation

    def calculate_biomechanical_summary(self):
        """Calculate final biomechanical metrics with proper timing."""
        # Extract features from post-FFC frames
        result = self.extract_features_post_ffc()
        post_ffc_features, front_knee_angles, back_knee_angles_list = result
        
        if post_ffc_features is None or (hasattr(post_ffc_features, "empty") and post_ffc_features.empty):
            print("❌ No post-FFC features available")
            return pd.DataFrame()
        
        # Analyze front knee phases
        ffc_knee, max_flex_knee, max_ext_knee, ffc_frame, flex_frame, ext_frame, knee_deviation = self.analyze_front_knee_phases(front_knee_angles)
        
        # Analyze back knee phases using the collected back knee angles
        bfc_knee, back_max_flex_knee, back_max_ext_knee, bfc_frame, back_flex_frame, back_ext_frame, back_knee_deviation = self.analyze_back_knee_phases(back_knee_angles_list)
        
        # Focus on ball release frame for critical measurements
        release_frame_data = post_ffc_features[post_ffc_features['is_ball_release'] == True] if isinstance(post_ffc_features, pd.DataFrame) else post_ffc_features[0][post_ffc_features[0]['is_ball_release'] == True]
        
        if release_frame_data.empty:
            # If no specific release frame, use the last few frames post-FFC
            if isinstance(post_ffc_features, pd.DataFrame):
                release_frame_data = post_ffc_features.tail(3)
            else:
                release_frame_data = post_ffc_features.tail(3)
        
        print(f"Using {len(release_frame_data)} frames around ball release for analysis")
        
        # Calculate summary statistics
        summary_data = []
        
        # Elbow angle (at ball release)
        elbow_values = release_frame_data['Elbow Angle'].dropna()
        if not elbow_values.empty:
            summary_data.append({
                'Feature': 'Elbow Angle',
                'Average': elbow_values.mean(),
                'Min': elbow_values.min(),
                'Max': elbow_values.max(),
                'Frames': len(elbow_values),
                'Measurement_Phase': 'Ball Release',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Front Knee Angle - at FFC
        if ffc_knee is not None and not np.isnan(ffc_knee):
            summary_data.append({
                'Feature': 'Front Knee at FFC',
                'Average': ffc_knee,
                'Min': ffc_knee,
                'Max': ffc_knee,
                'Frames': 1,
                'Measurement_Phase': 'Front Foot Contact',
                'FFC_Frame': ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Front Knee Angle - at maximum flexion
        if max_flex_knee is not None and not np.isnan(max_flex_knee):
            summary_data.append({
                'Feature': 'Front Knee Max Flexion',
                'Average': max_flex_knee,
                'Min': max_flex_knee,
                'Max': max_flex_knee,
                'Frames': 1,
                'Measurement_Phase': 'Maximum Flexion',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Front Knee Angle - at maximum extension
        if max_ext_knee is not None and not np.isnan(max_ext_knee):
            summary_data.append({
                'Feature': 'Front Knee Max Extension',
                'Average': max_ext_knee,
                'Min': max_ext_knee,
                'Max': max_ext_knee,
                'Frames': 1,
                'Measurement_Phase': 'Maximum Extension',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Knee Angle Deviation (FFC to Max Extension) - Power generation metric
        if knee_deviation is not None and not np.isnan(knee_deviation):
            summary_data.append({
                'Feature': 'Knee Angle Deviation',
                'Average': knee_deviation,
                'Min': knee_deviation,
                'Max': knee_deviation,
                'Frames': 1,
                'Measurement_Phase': 'FFC to Max Extension',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Back Knee Angle - at back foot contact
        if bfc_knee is not None and not np.isnan(bfc_knee):
            summary_data.append({
                'Feature': 'Back Knee at BFC',
                'Average': bfc_knee,
                'Min': bfc_knee,
                'Max': bfc_knee,
                'Frames': 1,
                'Measurement_Phase': 'Back Foot Contact',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Back Knee Angle - at maximum flexion
        if back_max_flex_knee is not None and not np.isnan(back_max_flex_knee):
            summary_data.append({
                'Feature': 'Back Knee Max Flexion',
                'Average': back_max_flex_knee,
                'Min': back_max_flex_knee,
                'Max': back_max_flex_knee,
                'Frames': 1,
                'Measurement_Phase': 'Back Maximum Flexion',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Back Knee Angle - at maximum extension before lift off
        if back_max_ext_knee is not None and not np.isnan(back_max_ext_knee):
            summary_data.append({
                'Feature': 'Back Knee Max Extension',
                'Average': back_max_ext_knee,
                'Min': back_max_ext_knee,
                'Max': back_max_ext_knee,
                'Frames': 1,
                'Measurement_Phase': 'Back Maximum Extension',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Back Knee Angle Deviation (BFC to Max Extension) - Power generation metric
        if back_knee_deviation is not None and not np.isnan(back_knee_deviation):
            summary_data.append({
                'Feature': 'Back Knee Deviation',
                'Average': back_knee_deviation,
                'Min': back_knee_deviation,
                'Max': back_knee_deviation,
                'Frames': 1,
                'Measurement_Phase': 'BFC to Max Extension',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })

        # Back knee angle (for power generation)
        back_knee_values = release_frame_data['Back Knee Angle'].dropna()
        if not back_knee_values.empty:
            summary_data.append({
                'Feature': 'Back Knee Angle',
                'Average': back_knee_values.mean(),
                'Min': back_knee_values.min(),
                'Max': back_knee_values.max(),
                'Frames': len(back_knee_values),
                'Measurement_Phase': 'Ball Release',
                'FFC_Frame': self.ffc_frame,
                'Release_Frame': self.ball_release_frame
            })
        
        # Other features - Comprehensive biomechanical analysis
        comprehensive_features = [
            ('Trunk Lean', 'Trunk Lean'),
            ('Hip-Shoulder Separation', 'Hip-Shoulder Separation'), 
            ('Shoulder Alignment', 'Shoulder Alignment'),
            ('Bowling Arm Angle', 'Bowling Arm Angle'),
            ('Non-Bowling Arm Angle', 'Non-Bowling Arm Angle'),
            ('Head Position', 'Head Position'),
            ('Pelvis Rotation', 'Pelvis Rotation'),
            ('Front Ankle Angle', 'Front Ankle Angle'),
            ('Back Ankle Angle', 'Back Ankle Angle'),
            ('Spine Angle', 'Spine Angle')
        ]
        
        for feature, column in comprehensive_features:
            values = release_frame_data[column].dropna()
            
            if not values.empty:
                summary_data.append({
                    'Feature': feature,
                    'Average': values.mean(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Frames': len(values),
                    'Measurement_Phase': 'Ball Release',
                    'FFC_Frame': self.ffc_frame,
                    'Release_Frame': self.ball_release_frame
                })
            else:
                summary_data.append({
                    'Feature': feature,
                    'Average': np.nan,
                    'Min': np.nan,
                    'Max': np.nan,
                    'Frames': 0,
                    'Measurement_Phase': 'Ball Release',
                    'FFC_Frame': self.ffc_frame,
                    'Release_Frame': self.ball_release_frame
                })
        
        return pd.DataFrame(summary_data)

def main():
    print("Enhanced Cricket Bowling Biomechanics Analysis")
    print("=" * 50)
    
    # Load landmarks data
    landmarks_csv = "labeled_pose_video_landmarks.csv"
    try:
        landmarks_df = pd.read_csv(landmarks_csv)
        landmarks_df = rename_columns(landmarks_df)
        print(f"✅ Loaded landmarks: {landmarks_df.shape}")
    except Exception as e:
        print(f"❌ Error loading landmarks: {e}")
        return
    
    # Extract enhanced biomechanics
    extractor = EnhancedCricketBiomechanicsExtractor(landmarks_df)
    biomechanics_summary = extractor.calculate_biomechanical_summary()
    
    if not biomechanics_summary.empty:
        print("\n" + "="*60)
        print("ENHANCED BIOMECHANICAL ANALYSIS (POST-FFC)")
        print("="*60)
        
        for _, row in biomechanics_summary.iterrows():
            print(f"\n{row['Feature']}:")
            print(f"  Measurement: {row['Average']:.2f}° (Post-FFC)")
            print(f"  Range: {row['Min']:.2f}° - {row['Max']:.2f}°")
            print(f"  Phase: {row['Measurement_Phase']}")
            print(f"  FFC Frame: {row['FFC_Frame']}")
        
        # Save enhanced results
        output_csv = "enhanced_biomechanics_post_ffc.csv"
        biomechanics_summary.to_csv(output_csv, index=False)
        print(f"\n✅ Enhanced analysis saved to: {output_csv}")
        
        # Replace the old comparison report with the new timing-aware version
        comparison_output = "biomechanics_comparison_report.csv"
        biomechanics_summary[['Feature', 'Average', 'Min', 'Max', 'Frames']].to_csv(comparison_output, index=False)
        print(f"✅ Updated comparison report: {comparison_output}")
        
    else:
        print("❌ No biomechanical features could be extracted")

if __name__ == "__main__":
    main()