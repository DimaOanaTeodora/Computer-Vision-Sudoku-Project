# Computer-Vision-Sudoku-Project
Computer Vision third year uni course 

### Link de DropBox [aici](https://www.dropbox.com/sh/d54q2uq3nk8q3l7/AABuMSrZL7D1CACa_lo1_jdla?dl=0)

### Deadline 
- ```1 dec 2021, 23:59```
  - [incarcare cod(proiect) aici](https://tinyurl.com/CAVA-2021-TEMA1-SOLUTII)
- ```5 dec 2021, 23:59```
  - [datele de test se iau de aici](https://tinyurl.com/CAVA-2021-TEMA1)
  - [de incarcat arhiva zip cu solutiile aici](https://tinyurl.com/CAVA-2021-TEMA1-REZULTATE)
### README.md
Proiectul trebuie sa includa si un fisier ```README.md``` cu urmatoarele:

- librariile de care este nevoie pentru rularea proiectului
```numpy==1.15.4
opencv_python==4.1.1.26
scikit_image==0.15.0
tensorflow_gpu==1.12.0
Pillow==7.0.0
scikit_learn==0.22.1
skimage==0.0
tensorflow==2.1.0
```
- cum sa fie rulat fiecare fisier pentru ficare task
```
Task 1: 
script: task_1.py
function: run_task1(input_folder_name), where input_folder_name is the path to the folder containing the images for task1
output: the output file is results/task1.txt
```

### Scop
Implementarea unui sistem automat de extragere a informatiei vizuale din imagini ce contin careuri Sudoku de tip Clasic sau Jigsaw.

### Rezolutie imagini
- inaltime: 4032 px
- latime: 3024 px

### Arhiva materiale [here](https://tinyurl.com/CAVA-2021-TEMA1)
- ```testare``` - disponibile dupa termenul limita de trimitere a codului
- ```antrenare``` - ```clasic ```si ```jigsaw``` 
- ```evaluare``` - indica cum sa scriem codul pt faza de evaluare pe datele de test
  - ```fake_test``` - exemplifica cum vor arata datele de testare (structura ca la antrenare) si similar cu directorul de ```test``` in care vor fi imaginile de testare
  - ```fisiere_solutie``` - exemplifica formatul fisierelor de trimis in faza a doua (director ```Alexe_Bogdan_331```)
  - ```cod_evaluare``` - cod folosit pentru evaluarea automata a rezultatelor --- trebuie sa ne asiguram ca acest cod ruleaza pe fisierele noastre 

# Task 1 - Sudoku Clasic
- de determinat daca fiecare celula contine sau nu o cifra
- celulele libere cu litera ```o``` mic
- celulele ocupate cu litera ```x``` mic
- datele de antrenare: ```20``` de imagini (centrare si aliniate cu axele Ox, Oy)

![image](https://user-images.githubusercontent.com/61749814/142925428-4aa097c0-3062-417c-ad9e-b6233479e055.png)

# Task 2 - Sudoku Jigsaw
1) determinare regiune de forma neregulata din careu
2) determinare daca fiecare celula contine sau nu o cifra
- string de lungime 2 pe fiecare celula 
  - cifra de la 1 la 9 care reprezeinta numarul regiunii de la stanga la dreapta si de sus in jos
    - de aceeasi culoare
    - sau 
    - delimitate prin linii ingrosate  
  - celulele libere cu litera ```o``` mic
  - celulele ocupate cu litera ```x``` mic
- determinare cifra corespunzatoare celula dintr-o regiune cu forma neregulata
  - numaram celule/regiunile de la stanga la dreapta si de sus in jos
  - celula din coltul din stanga sus primeste cifra 1 intrucat face parte din prima regiune de forma neregulata
  - asignam aceeasi cifra tuturor celulelor din aceeasi regiune
  - pentru prima celula din regiunea urmatoare asociem o cifra mai mare cu 1 (am trecut la regiunea urmatoare)
- datele de antrenare: ```40``` de imagini (20 alb negru si 20 color) (centrate si aliniate cu axele Ox si Oy)
- cele 3 culori posibile regiuni: albastru, galben, rosu

![image](https://user-images.githubusercontent.com/61749814/142925523-6abeb41d-4d94-4fd7-917f-46a72aa23abc.png)

### Notare
- Task 1 (4p) 0.2 de fiecare
- Task 2 (4p) 0.1 de fiecare
- documentatie (1p) - minim o pagina && ```PDF```
- oficiu (1p)
- BONUS (1,5p) - recunoastere cifre
