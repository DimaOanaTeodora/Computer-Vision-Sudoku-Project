# :eye_speech_bubble: Computer Vision Sudoku Project 

Computer vision project (third year uni course)

Check the documentation [here](https://github.com/DimaOanaTeodora/Sudoku-recognition/blob/main/Documentatie.pdf)

:snake:	```python 3.8.10 ```

<p align="center">
<img src="https://user-images.githubusercontent.com/61749814/167293534-b555b97a-a8fa-4dcc-bb43-f41ccd0f32dd.png" width="230" height="300" />
 <img src="https://user-images.githubusercontent.com/61749814/167293543-f1a5effd-7323-4c75-b223-0380bf60045e.png" width="270" height="300" />
</p>

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
