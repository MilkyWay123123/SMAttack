# SMAttack

# Prerequisites
> python >= 3.9  
  torch >= 2.3.0

# Data Preparation
## NTU RGB+D 60 and 120
- Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
- Download the skeleton-only datasets:  
  nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)  
  nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120)

# Training
```python
python Attack.py --config ./config/stgcn-ntu60-cs.yaml
```

