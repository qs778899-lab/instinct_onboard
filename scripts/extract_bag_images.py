#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2

# import ROS2 packages
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options

def extract_images_from_bag(bag_path, output_dir):
    """
    Extract images from a ROS 2 bag file and save them as PNGs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    raw_depth_dir = os.path.join(output_dir, "raw_depth")
    processed_depth_dir = os.path.join(output_dir, "processed_depth")
    
    os.makedirs(raw_depth_dir, exist_ok=True)
    os.makedirs(processed_depth_dir, exist_ok=True)

    storage_options, converter_options = get_rosbag_options(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    topics_to_extract = ['/debug/raw_depth_image', '/debug/depth_image']
    
    # Filter connections by topics we want
    storage_filter = rosbag2_py.StorageFilter(topics=topics_to_extract)
    reader.set_filter(storage_filter)

    print(f"Opening bag: {bag_path}")
    print(f"Extracting to: {output_dir}")
    
    count_raw = 0
    count_processed = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        # Deserialize message
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if msg.encoding == '16UC1':
            # Convert ROS Image message to numpy array (16-bit uint)
            img_array = np.ndarray(
                shape=(msg.height, msg.width), 
                dtype=np.uint16, 
                buffer=msg.data
            )
            
            # Save the image
            timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            if topic == '/debug/raw_depth_image':
                # Raw depth is in mm (from our scaling). 
                # Saving as 16-bit PNG preserves the exact mm values
                filename = os.path.join(raw_depth_dir, f"raw_{timestamp_sec:.6f}.png")
                cv2.imwrite(filename, img_array)
                count_raw += 1
                
            elif topic == '/debug/depth_image':
                # Processed depth is 16-bit scaled image (scaled by 255*2 during publishing)
                # Saving as 16-bit PNG
                filename = os.path.join(processed_depth_dir, f"proc_{timestamp_sec:.6f}.png")
                cv2.imwrite(filename, img_array)
                count_processed += 1
                
            # ---------------- ADDED: Normalize and save an 8-bit viewable version ----------------
            if msg.encoding == '16UC1':
                # Convert 16-bit to 8-bit for easy viewing
                # Normalize values to 0-255 based on the min/max in the image
                if img_array.max() > 0:
                    viewable_img = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
                else:
                    viewable_img = np.zeros_like(img_array, dtype=np.uint8)
                viewable_img = viewable_img.astype(np.uint8)
                
                # Apply colormap (JET is commonly used for depth maps: red is near, blue is far or vice versa depending on depth setup)
                # Or just keep it grayscale
                viewable_img_colored = cv2.applyColorMap(viewable_img, cv2.COLORMAP_JET)
                
                viewable_dir = os.path.join(output_dir, "viewable_depth")
                os.makedirs(viewable_dir, exist_ok=True)
                
                prefix = "raw" if topic == '/debug/raw_depth_image' else "proc"
                viewable_filename = os.path.join(viewable_dir, f"{prefix}_{timestamp_sec:.6f}.png")
                cv2.imwrite(viewable_filename, viewable_img_colored)
            # -------------------------------------------------------------------------------------

        else:
            print(f"Warning: Unexpected encoding '{msg.encoding}' for topic {topic}")

    print("Extraction complete!")
    print(f"Extracted {count_raw} raw depth images to {raw_depth_dir}")
    print(f"Extracted {count_processed} processed depth images to {processed_depth_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract depth images from a ROS 2 bag file.')
    # Instead of full path, just take the bag folder name and assume it's in rosbag_images
    parser.add_argument('bag_name', help='Name of the bag folder inside rosbag_images (e.g. bag_20260324_203015)')
    
    args = parser.parse_args()
    
    # Base directory for the script (assumes it's run from instinct_onboard or one level down)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Construct input path
    bag_path = os.path.join(project_root, "rosbag_images", args.bag_name)
    if not os.path.exists(bag_path):
        # Fallback in case user provides full path anyway
        if os.path.exists(args.bag_name):
            bag_path = args.bag_name
            args.bag_name = os.path.basename(os.path.normpath(args.bag_name))
        else:
            print(f"Error: Could not find bag at {bag_path}")
            exit(1)
            
    # Construct output path
    output_dir = os.path.join(project_root, "rosbag_viewable_images", args.bag_name)
    
    extract_images_from_bag(bag_path, output_dir)