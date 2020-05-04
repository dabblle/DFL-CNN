python detect_car.py \
	--cfg cfg/yolov3.cfg \
	--weights weights/yolov3.weights \
	--conf-thres 0.7 \
	--source /opt/data/private/code/DFL-CNN/tmp \
	--output ./output \
	--output_vehicle ./output_car 
