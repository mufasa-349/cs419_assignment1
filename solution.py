#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS419 - Digital Image Analysis Assignment 1
Road Intersection and Roundabout Detection

Approach:
1. Binary segmentation using Otsu thresholding
2. Morphological skeleton to find road centerlines
3. Junction detection via 8-neighborhood degree analysis
4. Roundabout detection via circular contour analysis
"""

import cv2
import numpy as np
import argparse
import math
from typing import List, Tuple


def otsu_threshold(img_gray: np.ndarray) -> np.ndarray:
    """Convert to binary using Otsu's method with auto-inversion."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert (roads should be white)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)
    
    return binary


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Remove noise using morphological opening and closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    """Compute morphological skeleton using iterative erosion."""
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(binary, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        
        if not np.any(eroded):
            break
        binary = eroded
    
    return skeleton


def prune_skeleton(skeleton: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Remove small spurs from skeleton."""
    pruned = skeleton.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    for _ in range(iterations):
        # Identify endpoints
        endpoints = get_endpoints(pruned)
        pruned[endpoints > 0] = 0
    
    return pruned


def get_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Get endpoint pixels (8-neighbors == 1)."""
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
    degree = neighbors - (skeleton > 0).astype(np.uint8)
    endpoints = (skeleton > 0) & (degree == 1)
    return endpoints.astype(np.uint8) * 255


def get_junctions(skeleton: np.ndarray) -> np.ndarray:
    """Get junction pixels (8-neighbors >= 4)."""
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
    degree = neighbors - (skeleton > 0).astype(np.uint8)
    junctions = (skeleton > 0) & (degree >= 4)
    return junctions.astype(np.uint8) * 255


def connected_components_to_points(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert connected components to center points."""
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    points = []
    
    for label_id in range(1, num_labels):
        component = (labels == label_id)
        if np.sum(component) == 0:
            continue
        
        # Get center of component
        y_coords, x_coords = np.where(component)
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
        points.append((cx, cy))
    
    return points


def merge_nearby_points(points: List[Tuple[int, int]], threshold: int = 20) -> List[Tuple[int, int]]:
    """Merge points that are very close to each other."""
    if not points:
        return []
    
    # Convert to numpy array
    points_array = np.array(points, dtype=np.float32)
    
    # Simple clustering approach
    merged = []
    used = np.zeros(len(points), dtype=bool)
    
    for i in range(len(points)):
        if used[i]:
            continue
        
        # Find all points within threshold
        distances = np.linalg.norm(points_array - points_array[i], axis=1)
        cluster = np.where(distances <= threshold)[0]
        
        # Average cluster points
        avg_x = int(np.mean(points_array[cluster, 0]))
        avg_y = int(np.mean(points_array[cluster, 1]))
        merged.append((avg_x, avg_y))
        
        # Mark as used
        used[cluster] = True
    
    return merged


def find_roundabouts_hough(binary: np.ndarray) -> List[Tuple[int, int]]:
    """Find roundabouts using Circle Hough Transform."""
    roundabouts = []
    
    # Use Circle Hough Transform to detect circular structures
    # Blur slightly to reduce noise
    blurred = cv2.GaussianBlur(binary, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,               # Inverse ratio of accumulator resolution
        minDist=150,        # Minimum distance between circle centers
        param1=50,          # Upper threshold for edge detection
        param2=35,          # Accumulator threshold for center detection (higher = fewer false positives)
        minRadius=15,       # Minimum circle radius
        maxRadius=80        # Maximum circle radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            roundabouts.append((x, y))
    
    return merge_nearby_points(roundabouts, threshold=50)


def find_loops(skeleton: np.ndarray) -> np.ndarray:
    """Find closed loops in skeleton."""
    # Find connected components without endpoints
    endpoints = get_endpoints(skeleton)
    
    # Remove endpoints temporarily
    no_endpoints = skeleton.copy()
    no_endpoints[endpoints > 0] = 0
    
    # Find components
    num_labels, labels = cv2.connectedComponents(no_endpoints, connectivity=8)
    loops = np.zeros_like(skeleton)
    
    for label_id in range(1, num_labels):
        component = (labels == label_id)
        pixel_count = np.sum(component)
        
        # Must have reasonable size (more lenient)
        if pixel_count > 20:
            loops[component] = 255
    
    return loops


def detect_intersections_and_roundabouts(input_path: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Main detection function.
    Returns: (junction_points, roundabout_points)
    """
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    
    # Binarization
    binary = otsu_threshold(img)
    
    # Morphological cleanup
    binary_clean = morphological_cleanup(binary)
    
    # Skeleton
    skeleton = morphological_skeleton(binary_clean)
    skeleton = prune_skeleton(skeleton, iterations=2)
    
    # Junction detection
    junctions_mask = get_junctions(skeleton)
    junction_points = connected_components_to_points(junctions_mask)
    junction_points = merge_nearby_points(junction_points, threshold=60)
    
    # Roundabout detection - using Circle Hough Transform
    roundabout_points = find_roundabouts_hough(binary_clean)
    
    return junction_points, roundabout_points


def main():
    parser = argparse.ArgumentParser(description="Detect road intersections and roundabouts")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--debug", help="Save debug images", action="store_true")
    
    args = parser.parse_args()
    
    junctions, roundabouts = detect_intersections_and_roundabouts(args.input)
    
    # Print results
    print(f"# Intersections (junctions): {len(junctions)}")
    for x, y in junctions:
        print(f"intersection: ({x}, {y})")
    
    print(f"\n# Roundabouts: {len(roundabouts)}")
    for x, y in roundabouts:
        print(f"roundabout: ({x}, {y})")


if __name__ == "__main__":
    main()

