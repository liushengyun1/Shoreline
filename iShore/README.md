

## Train your own model

### Train FPN
For better training efficiency and convergence performance, we train the FPN and the agent network separately. A saved checkpoint of FPN is provided, which could be directly used for later tasks. FPN is trained with the pretrain set.
运行
```
./run_train_seg.bash
```


### Train the agent network
运行
```
./run_train.bash
```

All the training parameters could be modified in the ```config.yml```. You can choose the number of rounds of the restricted exploration (```r_exp```) and the free exploration (```f_exp```) in this file, but it is suggested to set ```r_exp=1``` and ```f_exp=1``` for the trade-off between efficiency and effectiveness. More rounds of free explorations will not effectively enhance the final performance due to more time consumption.

If your need the visualization of the trajectories generated during the training period, set ```visualization``` in ```config.yml``` to **True**, and check the visualizations in folder ```./records/train/vis```. 

### Tensorboard
Open another terminal and run 
```
docker exec -it topo_iCurb bash
``` 
Then run 
```
./run_tensorboard_seg.bash # if you train segmentation net
./run_tensorboard.bash # if you train agent net
``` 
The port number of above two bashes is **5008**. 

### Inference
运行
```
./run_eval.bash
```

推理可以通过保存的检查点来完成。

生成的图的二值图保存在```中。生成的顶点保存在```./records/test/vertices_record``中。推理后，触发一个脚本来计算推理的评估结果。

## Visualization in aerial images
如果您想将生成的道路边界图投影到航空图像上以获得更好的可视化，请运行
```
python utils/visualize.py
```
Generated visualizations are saved in ```./records/test/final_vis```.



