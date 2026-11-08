# Plant Disease Detection

A small web app that looks at a photo of a leaf and tells you which disease it has, or whether it's healthy. It works for pepper (capsicum), potato and tomato leaves, 15 classes in total.

I built this to learn how image classification works end to end: exploring the data, training a model from scratch, trying transfer learning, and then putting the result in front of people through a Streamlit app.

Live demo: <add your Streamlit link here>

## What it can do

You upload a leaf photo and the app shows the top 3 predictions with their confidence. If the best guess is under 50%, it says so instead of acting sure about it.

Supported classes:

- Pepper (bell): bacterial spot, healthy
- Potato: early blight, late blight, healthy
- Tomato: bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, yellow leaf curl virus, mosaic virus, healthy

## Dataset

I used the PlantVillage dataset, only the pepper, potato and tomato folders (15 classes, 20,638 images). I checked every image for corruption and looked at the image sizes before doing anything else. The classes are not balanced, some tomato diseases have far more images than potato healthy, so I used a stratified split to keep the same proportions everywhere.

Split: 70% train, 15% validation, 15% test (random_state=42).

## Models

I trained two models and compared them on the same test set.

**Custom CNN.** Two conv layers (32 and 64 filters) with max pooling, global average pooling, a 128-unit dense layer with 0.5 dropout, then softmax. Trained for 10 epochs on images scaled to 0-1. Test accuracy: 79.91%.

**MobileNetV2 (transfer learning).** ImageNet weights with the base frozen, global average pooling, 0.3 dropout and a softmax layer on top. Adam with a learning rate of 0.001, 5 epochs. Test accuracy: 91.89%.

One thing that tripped me up: the two models need different preprocessing. The CNN gets pixels divided by 255, while MobileNetV2 needs `mobilenet_v2.preprocess_input` (values between -1 and 1). The model file doesn't include this step, so the app has to do it itself.

## Where it fails

This is the part I'd want to know about if I were using it.

PlantVillage photos are taken in a lab, one leaf on a plain background. Photos taken in a field, with soil, hands and other leaves in the frame, are harder, and I expect the accuracy to be lower than the number above. I haven't measured this properly yet.

The model only knows 15 classes and has no way to say "this isn't one of them". When I tested it with a healthy brinjal (eggplant) leaf, it called it tomato late blight with 84% confidence. Brinjal and tomato are related plants and their leaves look alike, but the real problem is that the model has to pick something. So a high confidence score doesn't mean the leaf is from a supported plant.

Also, a low score isn't always a wrong answer. One correct tomato early blight prediction came out at 48.8%, which is why the warning threshold is 50% and not higher.

Please don't use this to decide how to treat a real crop. Check with someone who knows plant diseases.

## Run it yourself

```
git clone https://github.com/MusabAmin1/Plant-disease-detection.git
cd Plant-disease-detection
pip install -r requirements.txt
streamlit run app.py
```

Files you need in the same folder: `app.py`, `class_names.json` and `plant_disease_mobilenetv2.keras`.

`class_names.json` is the list of class names in the order used during training. The model only outputs numbers, so this file is what turns index 3 into "Potato___Late_blight". If the order changes, the labels will be wrong even though the predictions are right.


## What I want to do next

- Add an "Other / not supported" class using leaves from other plants, so the app can refuse instead of guessing
- Test with real phone photos and report the accuracy honestly
- Unfreeze the top MobileNetV2 layers and fine-tune with a small learning rate

Built by Musab Amin. https://www.linkedin.com/in/musab-amin-96b418356/
