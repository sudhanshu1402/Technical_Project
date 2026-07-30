# SPAAC

The code. Run both entry points from this directory, since all paths are relative to it.

```bash
python face-rec-emotion.py     # recognition + attention + CSV logging (webcam)
python emotions.py             # emotion only, plays test/testvdo.mp4
```

`q` quits. `USE_WEBCAM` at the top of each script switches between camera index 0 and the video file.

## The attention idea

FER2013's seven emotions are collapsed into two classes in `utils/datasets.py`: happy, surprise, and neutral count as **Attentive**, and angry, disgust, fear, and sad as **NOT ATTENTIVE**. That remap is the project's actual contribution. The model underneath still predicts all seven.

## Layout

| File | Role |
|---|---|
| `face-rec-emotion.py` | Main script: dlib face detection, `face_recognition` matching against `images/`, emotion CNN, per-frame CSV rows |
| `emotions.py` | Simpler variant, no identity and no CSV |
| `utils/` | Dataset loaders, the label map, drawing helpers, preprocessing |
| `models/` | The pretrained Keras CNN |
| `images/` | Enrolled student gallery, one JPG each |
| `test/` | Sample videos, demo GIFs, project report |

## Rough edges

- **Hardcoded CSV path** to `C:\Users\admin\PycharmProjects\SPAAC\file.csv`. Logging fails anywhere else.
- **Enrollment is hardcoded**, loaded by name in the script rather than by scanning `images/`.
- **The color-by-emotion branches never fire.** `get_labels("fer2013")` returns the attention labels, but the `if emotion_text == "ANGRY"` checks compare against raw emotion names, so box color always falls through to green. Cosmetic.
- The bottom display loop in `face-rec-emotion.py` reads `face_names`, which is empty, so that second label pass draws nothing.

Setup and dependencies are in the [parent README](../README.md).
