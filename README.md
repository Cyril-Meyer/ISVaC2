# ISVaC2
Image Sensor Vacuum Cleaner 2

## The idea
ISVaC2 is a new version of [ISVaC](https://github.com/Yt-trium/ISVaC), with new detection and reconstruction methods. ISVaC2 uses images and/or videos instead as dataset.

The idea was that, since the smudges always appear in the same places, it is possible to detect their position automatically from a pool of images / videos.
Once the mask is calculated, it should be possible to train a neural network on the correct part of the image to reconstruct missing pixels.
