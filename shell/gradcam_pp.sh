cd ..
CUDA_VISIBLE_DEVICES=3  python3 main_gradcam.py --nGPU 1 \
--model AlexNet \
--module MGN  --slice_p2 2 --slice_p3 3 \
--datadir ./dataset \
--num_classes 3604 --gradcam_loss 1*CrossEntropy \
--margin 1.2 --save demo__   \
--center yes --width 28 --height 28 --layer_name conv5
