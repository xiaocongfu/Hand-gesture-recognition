# Hand-gesture-recognition
Step 1, segmentation:
-Hand segmentation using skin color;
-Arm part above wrist is segmented and only pale is keeped.
Step 2, recognition:
-Hu moments is extracted as features of hand gestures;
-ANN is used to train features and do the recognition.

Hand gestures supported:
fist, 1-5, paper, lizaed, spock.

g++ gestRecog.cpp -o gestRecog `pkg-config --cflags --libs opencv`
./gestRecog
