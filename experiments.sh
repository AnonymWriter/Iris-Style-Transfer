device=0

################################## train iris classification model ##################################
# no variation
python iris_classification.py -device $device -rp 0 -rd 0 -pp 0 -pd 0

# random rotation
for rd in 5 10 20 30 45 60 90 120 150 180
do
    python iris_classification.py -device $device -rp 1 -rd $rd -pp 0 -pd 0
done

# random perspective transformation
for pd in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python iris_classification.py -device $device -rp 0 -rd 0 -pp 1 -pd $pd
done

################################## train gaze estimation model ##################################
python gaze_estimation.py -device $device -estimator 1 --save_period 10 -E 250
python gaze_estimation.py -device $device -estimator 2 --save_period 50 -E 500

##################################  iris style transfer on OpenEDS2019 ##################################
python iris_style_transfer_openeds2019.py -device $device

##################################  iris style transfer on OpenEDS2020 ##################################
python iris_style_transfer_openeds2020.py -device $device