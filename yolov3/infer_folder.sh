python detect_car_batch.py \
    --cfg cfg/yolov3.cfg \
    --weights weights/yolov3.weights \
    --conf-thres 0.7 \
    --source /opt/data/private/DATASETS/model_cars/ \
    --output /opt/data/private/DATASETS/output/ \
    --output_vehicle  /opt/data/private/DATASETS/output_car/
