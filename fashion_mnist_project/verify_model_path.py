import os

def verify_model_path(model_path):
    if os.path.exists(model_path) and os.path.isfile(model_path):
        print(f"Found: {model_path}")
        return True
    else:
        print(f"Error: {model_path} not found or is not a file")
        return False

if __name__ == "__main__":
    model_path = "correct/path/to/object_detection_model.keras"
    if verify_model_path(model_path):
        print("Model path verification successful.")
    else:
        print("Model path verification failed.")
