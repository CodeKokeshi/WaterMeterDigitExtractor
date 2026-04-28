# LeNet Inference Guide For This Project

This guide explains how **our code** uses the trained LeNet model to read a 5-digit meter strip.

This is **not** a generic explanation of LeNet-5.
This is the exact practical flow you would want to replicate later in Android Studio.

## Goal

The classifier does **not** find the meter by itself.
It only reads a strip **after** the correct region has already been extracted.

So the full high-level flow is:

1. Start from the real image.
2. Find a correct 4-point ROI around the 5-digit strip.
3. Warp that ROI into a flat strip.
4. Binarize the strip.
5. Resize the strip to `140x28`.
6. Split it into five `28x28` digit cells.
7. Invert each digit cell before inference.
8. Normalize pixels to `0.0..1.0`.
9. Run the model.
10. Take the highest-probability class for each of the 5 cells.
11. Join the 5 predicted digits into one final label.

## Important Training Assumption

In this project, the LeNet model was trained on **inverted** digit images.

That means:

- background is mostly black
- digit strokes are mostly white

Because of that, the inference path also inverts the digit cells before prediction.

If you skip inversion in Android, the model will still produce an answer, but it may be confidently wrong.

## Step 1: Start From A Correct ROI

LeNet only works well if the ROI is already correct.

For this project, a "correct ROI" means:

- it contains only the 5-digit strip area
- the strip is straight enough after perspective warp
- the five digits occupy the strip evenly from left to right
- the top and bottom edges do not cut off the digits
- the left and right borders do not include too much extra background

If the ROI is wrong, LeNet will still read something, because it is a classifier.
It does not know when the input is nonsense.

## Step 2: Perspective Warp

Our app uses 4 points and warps the selected quadrilateral into a fixed rectangle.

Code reference:
- [D:\OS2025Dev\DigitExtractor\main.py](D:/OS2025Dev/DigitExtractor/main.py)

The warp target is:

- high-resolution buffer: `500x100`
- final strip size: `140x28`

Practical meaning:

- first the ROI is flattened into a rectangular strip
- then it is resized into the exact layout expected by our digit segmentation logic

## Step 3: Convert To Grayscale

After warping, the strip is converted to grayscale if needed.

This removes color and keeps only intensity information.

That matters because the model was trained on grayscale-like digit images, not RGB color images.

## Step 4: Binarization

Our app binarizes the warped strip before digit reading.

Current preprocessing in code:

- adaptive Gaussian threshold
- threshold mode: `THRESH_BINARY`
- block size: `11`
- `C = 2`
- then median blur with kernel `3`

Code reference:
- [D:\OS2025Dev\DigitExtractor\main.py](D:/OS2025Dev/DigitExtractor/main.py)

What this means in simple English:

- each small local region gets its own threshold
- this helps when lighting is uneven
- after thresholding, noise is reduced with a small median blur

This step is one reason the model works reasonably well even on real photos.

## Step 5: Resize The Whole Strip To 140x28

After binarization, the strip is resized to:

- width `140`
- height `28`

This is important because our segmentation logic assumes:

- 5 digits total
- each digit gets exactly `28` pixels in width

So the strip is treated as:

- digit 1: columns `0..27`
- digit 2: columns `28..55`
- digit 3: columns `56..83`
- digit 4: columns `84..111`
- digit 5: columns `112..139`

This is why the ROI must be correct.
If the strip is not evenly aligned, the 5-way split becomes wrong even if the model itself is fine.

## Step 6: Split Into Five 28x28 Cells

The `140x28` strip is split into five separate `28x28` images.

Each cell is treated as one digit.

This is the exact shape expected by the classifier.

If one cell contains part of its neighbor, or if a digit is too far left/right inside the strip, the final reading becomes unstable.

## Step 7: Invert Before Inference

This is the key project-specific rule.

Before the model reads the digits, each segmented cell is inverted.

Why:

- your dataset preparation flow used the `Invert Color` tool before training
- so the model learned from inverted digit images

In simple English:

- app extraction gives a binarized strip
- inference flips black and white
- then the model receives the same visual style it saw during training

## Step 8: Prepare Each Cell For The Model

Each digit cell is then prepared like this:

1. Ensure grayscale.
2. Resize to `28x28`.
3. Convert pixel values from `0..255` to `0.0..1.0`.
4. Add the channel dimension so the shape becomes `28x28x1`.

For a batch of 5 digits, the model input shape becomes:

- `5 x 28 x 28 x 1`

If you do this in Android Studio with TensorFlow Lite, this is the part that must match exactly.

## Step 9: Model Prediction

The model outputs 10 probabilities per cell:

- one probability for each digit class `0..9`

For each of the 5 cells:

1. Get the output vector of length 10.
2. Pick the index with the highest probability.
3. That index is the predicted digit.

Then the app joins the 5 predicted digits into one string.

Example:

- cell predictions: `0`, `0`, `0`, `5`, `8`
- final label: `00058`

## Step 10: Candidate Search In The Current App

The current app does not rely on only one warped strip anymore.

It creates several small variants of the extracted strip, such as:

- small rotation changes
- small shear changes
- slightly tighter or wider vertical scaling
- slight up/down shifts

Each candidate is read by LeNet.
Then the app compares the candidate scores and keeps the best one.

Why this exists:

- a slightly wrong ROI can still distort one or more digits
- trying a few nearby variants helps recover some of those cases

But this is still not a replacement for a correct ROI.
It only helps around the edges.

## How The App Scores Candidate Readings

The current app uses the model confidences to score a 5-digit prediction.

It combines:

- average digit confidence
- minimum digit confidence

This is used to avoid trusting a candidate just because 4 digits are confident while 1 digit is clearly weak.

Still, confidence alone is not proof that the reading is correct.
A classifier can be confidently wrong if the ROI is bad.

## Why Wrong Readings Still Happen

These are the main reasons:

1. The ROI is not the real strip.
2. The 4-point rectangle is too large or too small.
3. The strip is slanted but was forced into the wrong quadrilateral.
4. The 5-digit split is uneven.
5. A border, reflection, frame edge, or neighboring symbol leaks into one cell.
6. The strip was binarized in a way that damaged the digit strokes.
7. The model sees a valid-looking shape but it belongs to the wrong class.

The screenshots you shared show that this project's biggest problem right now is usually **ROI quality**, not the basic LeNet input pipeline.

## The Exact Practical Android Flow To Replicate

If you already have a perfect finder in Android Studio, the LeNet part should be:

1. Capture the real image.
2. Detect the 4 corners of the digit strip.
3. Perspective-warp the strip to a rectangle.
4. Convert to grayscale.
5. Apply adaptive thresholding.
6. Apply a small median blur.
7. Resize the full strip to `140x28`.
8. Split into five `28x28` cells.
9. Invert each cell.
10. Convert pixels to `float32` in the range `0.0..1.0`.
11. Feed the 5 cells to the TFLite model.
12. For each cell, choose the highest-probability class.
13. Join the 5 classes into one final reading.

Optional but recommended:

14. Generate a few nearby candidate warps.
15. Read all candidates.
16. Keep the strongest result instead of trusting only one warp.

## Android Consistency Rules

To match this desktop app, Android should keep these rules identical:

- same strip size: `140x28`
- same digit cell size: `28x28`
- same grayscale workflow
- same thresholding logic or as close as possible
- same inversion before inference
- same normalization to `0.0..1.0`
- same digit order from left to right

If these are changed, the model may still run but accuracy can drop.

## What To Tweak First If You Want Better Results

If the reading is wrong, tweak these in this order:

1. ROI quality
2. Perspective correction
3. Digit alignment across the 5 equal cells
4. Thresholding parameters
5. Candidate search around the ROI
6. Model retraining

This order matters because a bad ROI will break everything after it.

## Short Version

The classifier path for this project is:

`Perfect 4-point ROI -> warp -> grayscale -> adaptive threshold -> median blur -> resize to 140x28 -> split into 5 cells -> invert -> normalize -> LeNet/TFLite -> argmax per cell -> join into 5-digit result`

That is the pipeline you should mirror later in Android Studio.
