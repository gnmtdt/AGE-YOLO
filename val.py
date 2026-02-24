from ultralytics import YOLO

if __name__ == '__main__':
    # Load trained AGE-YOLO weights
    model = YOLO('weights/age_yolo_gc10.pt')

    # Evaluate on the validation split defined in dataset/gc10.yaml
    metrics = model.val(
        data='dataset/gc10_data.yaml',
        split='val',
        imgsz=640,
        device=0
    )

    # Output key evaluation metrics
    print(f"Val mAP@50:     {metrics.results_dict['metrics/mAP50(B)']:.4f}")

