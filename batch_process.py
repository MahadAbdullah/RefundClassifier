import os
import csv
from datetime import datetime
from preprocessing import preprocess_image_bytes
from model import predict

def batch_process_images(image_folder="images", output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_array = preprocess_image_bytes(image_bytes)
                predictions = predict(image_array)

                top_preds = list(predictions.items())[:3]
                row = {
                    "filename": filename,
                    "prediction_1": top_preds[0][0],
                    "prob_1": round(top_preds[0][1], 3),
                    "prediction_2": top_preds[1][0],
                    "prob_2": round(top_preds[1][1], 3),
                    "prediction_3": top_preds[2][0],
                    "prob_3": round(top_preds[2][1], 3)
                }
                results.append(row)

    # Save output with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(output_folder, f"batch_results_{timestamp}.csv")
    fieldnames = ["filename", "prediction_1", "prob_1", "prediction_2", "prob_2", "prediction_3", "prob_3"]

    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n📝 Batch predictions saved to {output_path}")

if __name__ == "__main__":
    batch_process_images()
