# Insta360-ONE-X2-WebcamReframeTool
The `webcam.py` script allows users to reframe the Insta360 ONE X2's webcam feed into a cropped, single camera view, and it forwards this new view to a virtual webcam using pyvirtualcam.

# Prerequisites
This script requires Windows, because it relies on the [obs-virtual-cam module](https://github.com/Fenrirthviti/obs-virtual-cam) to create and stream to virtual video devices.

1. [Install obs-virtual-cam](https://github.com/Fenrirthviti/obs-virtual-cam/releases). The directions assume you install it at `C:\Program Files\obs-studio`.
2. Install pyvirtualcam
```
> pip install pyvirtualcam
```

# Directions
Before running the plugin, the virtual web camera device needs to be created using obs-virtual-cam:
- *YOU MUST run this CMD as Administrator*
```
regsvr32 /u "C:\Program Files\obs-studio\bin\64bit\obs-virtualsource.dll" 
regsvr32 /n /i:"1" "C:\Program Files\obs-studio\bin\64bit\obs-virtualsource.dll"
```
The above command should create a single virtual webcam.
Finally, you can run
```
> python webcam.py --hide-fps
```
in order to start feeding frames to the virtual webcam. A preview window will be displayed for you to see how the webcam is currently framed. While using the preview window, you can press the 'A' key to rotate the frame to the left, and the 'D' key to rotate the frame to the right. Press the 'Q' key to quit the script.

Your Insta360 ONE X2 camera must be connected to your PC, and it must have the USB mode set to Webcam mode.