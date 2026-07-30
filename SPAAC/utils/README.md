# utils

Helper package imported by the two scripts in `../`. Most of it is adapted from the mini-Xception face-classification work. The SPAAC-specific part is the label mapping.

| File | Holds |
|---|---|
| `datasets.py` | `DataManager` for FER2013 (CSV), IMDB gender (`.mat`), or KDEF (folder), plus `get_labels`, `get_class_to_arg`, and split helpers |
| `preprocessor.py` | `preprocess_input` (scale to `[-1, 1]`), `_imread`/`_imresize` over OpenCV, `to_categorical` |
| `inference.py` | Runtime drawing and detection: Haar cascade loading, `detectMultiScale`, boxes, text, bounding-box offsets |
| `data_augmentation.py` | `ImageGenerator` batch generator: saturation, brightness, contrast, lighting noise, random crop, flips |
| `visualizer.py` | Matplotlib helpers to tile faces and plot conv-layer weights |
| `grad_cam.py` | Grad-CAM and guided backprop heatmaps |

Only `datasets.py`, `preprocessor.py`, and `inference.py` are used by the two run scripts. The other three are training and debugging tools.

## The SPAAC twist

`get_labels("fer2013")` doesn't return the seven raw emotions. It collapses them into an attention signal:

```python
{
    0: "NOT ATTENTIVE",  # angry
    1: "NOT ATTENTIVE",  # disgust
    2: "NOT ATTENTIVE",  # fear
    3: "Attentive",      # happy
    4: "NOT ATTENTIVE",  # sad
    5: "Attentive",      # surprise
    6: "Attentive",      # neutral
}
```

The emotion classifier is repurposed as a proxy for engagement. That mapping is the project's core idea, not a general label set.

## Notes

- Dataset paths default to `../datasets/...` and models to `../trained_models/...`, so run from the SPAAC root with those folders present.
- `grad_cam.py` and `visualizer.py` have `__main__` blocks expecting pickled `faces.pkl` / `emotions.pkl`. Debugging entry points, not part of the flow.
- `_load_fer2013` calls `pd.get_dummies(...).as_matrix()`, removed in modern pandas. This targets an older pandas, Keras, and TF1 stack, and `grad_cam.py` uses TF1-era APIs throughout.
