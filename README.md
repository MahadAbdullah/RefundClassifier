# Refund Classification Prototype

This is a minimal image classification system that helps automate refund item identification using ResNet50 pretrained on ImageNet. It supports both manual prediction via a REST API and automated nightly batch processing.

## Project Structure

- `app.py` — REST API endpoint for image predictions
- `batch_process.py` — Processes images nightly in bulk
- `model.py` — Uses ResNet50 to classify images
- `preprocessing.py` — Resizes and normalizes image data
- `requirements.txt` — All required Python packages
- `images/` — Folder with refund item images
- `results/` — Folder where CSV prediction files are saved

## Setting up the project

1. Clone the repository
2. Create a virtual environment
    ```bash
    # Linux:
    source venv/bin/activate
    # Windows:
    venv\Scripts\activate
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Using the API
- Start the Flask server:
    ```
    python app.py
    ```
- Send a single image:
    ```
    curl -X POST -F "file=@images/Towels.jpg" http://localhost:5000/predict
    ```  
- Send multiple images in one request:
    ```
    curl -X POST -F "file=@images/Towels.jpg" -F "file=@images/Couch.jpg" http://localhost:5000/predict
    ```  

## Running the batch processor

- Run it manually:
    ```
    python batch_process.py
    ```

Results will be saved in a timestamped CSV file inside the `results/` folder.

## Automating Daily Predictions

Set up a cron job or use Windows Task Scheduler to run `batch_process.py` every night.

Example cron entry (Linux/macOS):
```
0 1 * * * /path/to/venv/bin/python /path/to/batch_process.py
```

## Notes

- This system uses ResNet50 trained on ImageNet, so no custom dataset is required.
- Preprocessing is done in-memory (no temp files).
- Results are saved in CSV format for easy review by refund staff.
