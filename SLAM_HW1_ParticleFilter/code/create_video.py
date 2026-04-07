'''
Script to create video from particle filter visualization images
Tries opencv first, falls back to ffmpeg if opencv is not available
'''

import argparse
import os
import glob
import re
import subprocess
import sys

def create_video_opencv(image_dir, output_video, fps=10, speed_multiplier=1):
    """Create video using opencv"""
    try:
        import cv2
    except ImportError:
        return False
    
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if len(image_files) == 0:
        return False
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        return False
    
    height, width, _ = first_img.shape
    actual_fps = fps * speed_multiplier
    
    # Try different codecs in order of preference
    codecs = ['avc1', 'H264', 'mp4v', 'XVID']
    out = None
    fourcc = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_video, fourcc, actual_fps, (width, height))
            if out.isOpened():
                break
            else:
                out.release()
                out = None
        except:
            if out is not None:
                out.release()
                out = None
            continue
    
    if out is None or not out.isOpened():
        return False
    
    success_count = 0
    failed_count = 0
    
    for idx, img_file in enumerate(image_files):
        img = cv2.imread(img_file)
        if img is None:
            failed_count += 1
            continue
        
        # Check if image dimensions match
        h, w, _ = img.shape
        if h != height or w != width:
            # Resize image to match expected dimensions
            img = cv2.resize(img, (width, height))
        
        # Write frame (write() returns None, so we can't check return value)
        # Just write and continue - we'll verify the output file at the end
        try:
            out.write(img)
            success_count += 1
        except Exception as e:
            failed_count += 1
            # If too many failures, abort and fall back to ffmpeg
            if failed_count > 10 and failed_count > success_count * 0.1:
                out.release()
                return False
    
    out.release()
    
    # Check if video file was created and has reasonable size
    if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        return True
    else:
        return False

def create_video_ffmpeg(image_dir, output_video, fps=10, speed_multiplier=1):
    """Create video using ffmpeg"""
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    if len(image_files) == 0:
        print(f"No PNG files found in {image_dir}")
        return False
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
    actual_fps = fps * speed_multiplier
    
    # Check if files are sequentially numbered starting from 0 or 1
    # This is needed for pattern-based input to work
    use_pattern = False
    if len(image_files) > 0:
        first_file = image_files[0]
        base_name = os.path.basename(first_file)
        match = re.search(r'(\d+)', base_name)
        if match:
            first_num = int(match.group(1))
            num_digits = len(match.group(1))
            # Check if files are sequential (0, 1, 2, ... or 1, 2, 3, ...)
            if first_num <= 1:
                # Check if all files follow the pattern
                expected_count = len(image_files)
                if first_num == 0:
                    expected_range = set(range(expected_count))
                else:
                    expected_range = set(range(1, expected_count + 1))
                
                actual_numbers = set()
                for img_file in image_files:
                    base = os.path.basename(img_file)
                    num_match = re.search(r'(\d+)', base)
                    if num_match:
                        actual_numbers.add(int(num_match.group(1)))
                
                # If numbers match expected sequence, use pattern
                if actual_numbers == expected_range:
                    pattern = re.sub(r'\d+', f'%0{num_digits}d', base_name)
                    use_pattern = True
    
    if use_pattern:
        # Use pattern-based input
        input_pattern = os.path.join(image_dir, pattern)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(actual_fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            output_video
        ]
    else:
        # Use concat demuxer (more reliable for non-sequential files)
        list_file = os.path.join(image_dir, 'filelist.txt')
        try:
            with open(list_file, 'w') as f:
                for img_file in image_files:
                    # Use absolute path to avoid path issues
                    abs_path = os.path.abspath(img_file)
                    # Escape single quotes in path
                    abs_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{abs_path}'\n")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-r', str(actual_fps),
                '-i', list_file,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'medium',
                output_video
            ]
        except Exception as e:
            print(f"Error creating file list: {e}")
            return False
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up
    list_file = os.path.join(image_dir, 'filelist.txt')
    if os.path.exists(list_file):
        try:
            os.remove(list_file)
        except:
            pass
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False
    
    # Verify output file was created
    if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        return True
    else:
        return False

def create_video_from_images(image_dir, output_video, fps=10, speed_multiplier=1):
    """
    Create video from sequence of PNG images
    
    param[in] image_dir : directory containing PNG images
    param[in] output_video : output video file path
    param[in] fps : frames per second
    param[in] speed_multiplier : speed multiplier for video (e.g., 2 = 2x speed)
    """
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    if len(image_files) == 0:
        print(f"No PNG files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Creating video: {output_video}")
    
    actual_fps = fps * speed_multiplier
    print(f"FPS: {actual_fps} (original {fps} x {speed_multiplier} speed)")
    
    # Try opencv first
    print("Trying opencv...")
    if create_video_opencv(image_dir, output_video, fps, speed_multiplier):
        print(f"Video created using opencv: {output_video}")
        print(f"Total frames: {len(image_files)}")
        print(f"Video duration: {len(image_files) / actual_fps:.2f} seconds")
        return
    
    # Fall back to ffmpeg
    print("Trying ffmpeg...")
    if create_video_ffmpeg(image_dir, output_video, fps, speed_multiplier):
        print(f"Video created using ffmpeg: {output_video}")
        print(f"Total frames: {len(image_files)}")
        print(f"Video duration: {len(image_files) / actual_fps:.2f} seconds")
        return
    
    # If both fail
    print("Error: Could not create video. Please install one of:")
    print("  - opencv-python: pip install opencv-python")
    print("  - ffmpeg: brew install ffmpeg (macOS) or sudo apt-get install ffmpeg (Linux)")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create video from particle filter visualization images')
    parser.add_argument('--image_dir', default='results/robotdata1', help='Directory containing PNG images')
    parser.add_argument('--output', default='localization_video.mp4', help='Output video file')
    parser.add_argument('--fps', default=10, type=int, help='Frames per second (before speed multiplier)')
    parser.add_argument('--speed', default=5, type=int, help='Speed multiplier (e.g., 5 = 5x speed)')
    args = parser.parse_args()
    
    create_video_from_images(args.image_dir, args.output, args.fps, args.speed)

