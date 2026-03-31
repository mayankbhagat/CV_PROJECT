

DATA_YAML=$1
MODEL="yolov8n.pt"   # start with nano, switch to yolov8s/m/l/x later
EPOCHS=30
IMGSZ=1024
BATCH=4   # change to fit GPU memory
PROJECT="runs/detect"
NAME="vindr_v1"

yolo detect train model=${MODEL} data=${DATA_YAML} epochs=${EPOCHS} imgsz=${IMGSZ} batch=${BATCH} project=${PROJECT} name=${NAME}
