# Hypernetworks_NNweights_TrainingDataset
The dataset contains 1,000 instances of neural networks, divided into a total
of 10 classes. Each neural network is a two-way, one-versus-all image classifier,
and each class contains 101 neural networks that can identify the images of that
class. The different classes are taken from the imagenette dataset, specifically
the Imagenette 320px V2 dataset with classes 0: Tench, 1: English Springer, 2:
Cassette Player, 3: Chain Saw, 4: Church, 5:French Horn, 6: Garbage Truck,
7: Gas Pump, 8: Golf Ball, and 9: Parachute

 | class          |          accuracy          |          precision        |           recall          |         f1score          |
 | -------------- |  min   |  max  |  average  |  min  |  max  |  average  |  min  |  max  |  average  |  min  |  max  |  average |
 | -------------- | ------ | ----- | --------- |-------|-------|-----------|-------|-------|-----------|------|-------|-----------|
 tench            | 0.935  | 0.949 |   0.943   | 0.702 | 0.848 |   0.780   | 0.488 | 0.680 |   0.592   | 0.606 | 0.704 |   0.672  | 
 english springer | 0.901  | 0.920 |   0.912   | 0.510 | 0.720 |   0.611   | 0.228 | 0.496 |   0.365   | 0.346 | 0.527 |   0.452  | 
 cassette player  | 0.899  | 0.934 |   0.926   | 0.455 | 0.797 |   0.648   | 0.342 | 0.591 |   0.439   | 0.460 | 0.551 |   0.519  | 
 chain saw        | 0.898  | 0.906 |   0.902   | 0.418 | 0.611 |   0.512   | 0.047 | 0.148 |   0.099   | 0.086 | 0.225 |   0.164  | 
 church           | 0.900  | 0.916 |   0.910   | 0.524 | 0.718 |   0.631   | 0.196 | 0.484 |   0.343   | 0.299 | 0.520 |   0.441  | 
 french horn      | 0.891  | 0.906 |   0.900   | 0.431 | 0.597 |   0.505   | 0.086 | 0.376 |   0.231   | 0.146 | 0.420 |   0.312  | 
 garbage truck    | 0.905  | 0.925 |   0.918   | 0.520 | 0.762 |   0.644   | 0.198 | 0.586 |   0.413   | 0.314 | 0.560 |   0.498  | 
 gas pump         | 0.882  | 0.898 |   0.891   | 0.376 | 0.610 |   0.478   | 0.086 | 0.246 |   0.175   | 0.151 | 0.331 |   0.254  | 
 golf ball        | 0.900  | 0.919 |   0.912   | 0.513 | 0.763 |   0.638   | 0.228 | 0.421 |   0.314   | 0.347 | 0.485 |   0.418  | 
 parachute        | 0.925  | 0.943 |   0.938   | 0.617 | 0.870 |   0.755   | 0.451 | 0.641 |   0.557   | 0.582 | 0.675 |   0.639  | 
 all classes      | .882   | 0.949 |   0.915   | 0.376 | 0.870 |   0.620   | 0.047 | 0.680 |   0.353   | 0.086 | 0.704 |   0.437  | 
