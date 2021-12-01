# :eye_speech_bubble: Computer Vision Sudoku Project 
:snake:	```python 3.8.10 ``` :snake:	
## Librarii utilizate
```
opencv-python==4.5.4.60
numpy==1.21.4
scipy==1.7.2
imutils==0.5.4
```

## Instructiuni de compilare

### :star: Task 1
script: ```task1.py```

----> run without any arguments

output: the output files are in ```Dima_Oana_341/clasic/01_predicted.txt```

-----> if you need to change take a look to the path constants:
```
A_INPUT_FOLDER_PATH = "input folder path for train"
T_INPUT_FOLDER_PATH = "input folder for test"
 
[...]
 
apply_for_all(OUTPUT_FOLDER_PATH,
              A_INPUT_FOLDER_PATH, 
              T_OUTPUT_NAME_PATTERN) # for _preticted.txt
```

### :star: Task 2
script: ```task2.py```

----> run without any arguments

output: the output files are in ```Dima_Oana_341/jigsaw/01_predicted.txt```

-----> if you need to change take a look to the path constants:
```
A_INPUT_FOLDER_PATH =  "input folder path"
T_INPUT_FOLDER_PATH = "input folder for test"
 
[...]
 
apply_for_all(OUTPUT_FOLDER_PATH,
              A_INPUT_FOLDER_PATH,
              T_OUTPUT_NAME_PATTERN) # for _preticted.txt
```
