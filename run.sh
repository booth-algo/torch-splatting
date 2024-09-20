
blender -b -P gaussian_splatting/utils/blender_script.py -- --input_path ./training-data/truck-col/Truck.ply --output_path ./training-data/truck-col/ 

python gaussian_splatting/utils/blender2json.py --input_path ./training-data/truck-col/ --output_path ./training-data/processed/
 
