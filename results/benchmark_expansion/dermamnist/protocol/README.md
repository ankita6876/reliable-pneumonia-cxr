
# DermaMNIST Spatial Shortcut Protocol

Dataset:
- Official DermaMNIST+ 64x64
- Seven-class classification

Marker:
- 6x6 RGB square
- Seven distinct marker colors
- Identical area across all marker codes

Positions:
- Center
- Bottom-right
- Random

Reliability:
- Independent control
- 50%
- 75%
- 100%

Definition:
P(marker_code = true_class)

Assignment:
- Stratified by true class
- Incorrect marker code uniformly selected
  among the other six classes

Independent control:
- Marker code sampled independently of class

Training:
- Not performed during this phase

Evaluation planned:
- Clean
- Matching
- Reversed / alternative code
- Neutral marker
- Counterfactual probability sensitivity
- Decision flip rate
- Accuracy
- Macro-F1
- Multiclass AUROC where defined

Important:
- Multiclass 50% reliability is not chance level.
- Independent control is the appropriate null condition.
- Dataset-size effects require within-dataset ablations.
