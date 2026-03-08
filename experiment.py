import time
import os
from PIL import Image
from transformers import pipeline

# 1. Setup Models
print("Loading models...")
model_1 = pipeline("image-classification", model="hanslab37/architectural-classifier-resnet-50")
model_2 = pipeline("image-classification", model="google/vit-base-patch16-224")

# 2. Define your Ground Truth (The "Answer Key")
# Replace these with your actual test image filenames and their true labels
ground_truth = {
    "page_1.png": "Drawing",
    "page_2.png": "Non-Drawing",
    "page_3.png": "Drawing",
    # add all your test images here...
}

image_folder = "./test_images" # Folder where your test images are stored

def run_experiment(model_pipeline, model_name):
    correct_predictions = 0
    total_images = len(ground_truth)
    
    print(f"\n--- Starting Evaluation for {model_name} ---")
    
    # Start the timer
    start_time = time.time()
    
    for filename, true_label in ground_truth.items():
        img_path = os.path.join(image_folder, filename)
        
        try:
            image = Image.open(img_path)
            results = model_pipeline(image)
            top_prediction = str(results[0]['label']).lower()
            
            # Map the model's specific output to your "Drawing" or "Non-Drawing" categories
            # Note: You may need to adjust these keywords based on what each model actually outputs!
            if "drawing" in top_prediction or "architectural" in top_prediction or "plan" in top_prediction:
                predicted_label = "Drawing"
            else:
                predicted_label = "Non-Drawing"
                
            if predicted_label == true_label:
                correct_predictions += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # End the timer
    end_time = time.time()
    
    # Calculate Metrics
    runtime = end_time - start_time
    accuracy = (correct_predictions / total_images) * 100
    
    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    print(f"Total Runtime: {runtime:.2f} seconds")
    print(f"Average Time per Image: {(runtime/total_images):.4f} seconds")
    
    return accuracy, runtime

# 3. Run the experiment on both models
acc_1, time_1 = run_experiment(model_1, "ResNet-50 (hanslab37)")
acc_2, time_2 = run_experiment(model_2, "google")
