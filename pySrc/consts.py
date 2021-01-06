AVG_HEIGHT = {
    0 : 1.0,
    1 :0.87331,
    2 :1.43447,
    3 :2.04129,
    4 :2.16531,
    5 :1.22675,
    6 :1.04758,
    7 :1.16441,
    9 :0.53083,
    8 :1.23136,
    10 :0.71563,
}


CLASS_NAME = {
    #   'barrier': 0,
      'car': 1, 
      'truck': 2, 
      'bus': 3, 
      'trailer': 4, 
      'construction_vehicle': 5, 
      'pedestrian': 6, 
      'motorcycle': 7, 
      'bicycle': 8,
      'traffic_cone': 9, 
      'barrier': 10
}

class_reverse = {index: name for name, index in CLASS_NAME.items()}