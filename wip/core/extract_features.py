"""
Extracts features assuming no shared cameras and no shared focal distances. 
 

"""



def extract_features(image_path, database_path) -> None:

    print("Starting feature matching.")

    cmd = [
        "colmap",
        "feature_extractor",
        "--image_path", str(image_path),
        "--database_path", str(database_path),
        "--camera_mode", "3",  # Different model for each Image
        "--ImageReader.single_camera", "0",  # No shared intrinsics
        "--ImageReader.single_camera_per_image", "1",  # Use a different camera for each image 
        "--FeatureExtraction.use_gpu", "1",  # Use GPU for extraction
        "--FeatureExtraction.num_threads", "-1"  # Use all Cores
    ]
