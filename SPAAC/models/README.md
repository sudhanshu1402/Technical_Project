# models/

`emotion_model.hdf5` (~852 KB), a Keras CNN trained on FER2013 and saved with architecture plus weights. Third-party pretrained weights, not trained in this repo. No training code here.

Seven-class classifier over the FER2013 set (angry, disgust, fear, happy, sad, surprise, neutral). The small file size fits a compact mini-Xception style architecture.

## Contract

**In:** a grayscale face crop resized to the model's `input_shape` (48x48), normalized to `[-1, 1]`, shaped `(1, H, W, 1)`.
**Out:** a probability vector over the 7 classes. Calling code takes `argmax` for the label and `max` for confidence.

Normalization lives in `../utils/preprocessor.py`:

```python
x = x.astype("float32") / 255.0
x = (x - 0.5) * 2.0   # -> [-1, 1]
```

Both scripts load it by relative path, so run them from the SPAAC root, not from in here:

```python
emotion_classifier = load_model("./models/emotion_model.hdf5")
emotion_target_size = emotion_classifier.input_shape[1:3]
```

## Note on labels

SPAAC never displays the raw emotion names. `get_labels("fer2013")` remaps the 7 class indices onto Attentive and NOT ATTENTIVE after `argmax`, and that's what gets drawn and logged.

No evaluation or benchmark data is checked in. Treat accuracy as whatever FER2013 mini-Xception gives you: fine for a demo, not validated for real classroom use.
