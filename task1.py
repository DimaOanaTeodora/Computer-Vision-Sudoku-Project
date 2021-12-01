import cv2 as cv
import numpy as np
import imutils 
import os 

A_INPUT_FOLDER_PATH =  ".\\antrenare\\clasic\\"
A_EVALUATE_FOLDER_PATH = ".\\antrenare\\results\\clasic\\"
A_OUTPUT_NAME_PATTERN = "_gt.txt" # 01 # 10 .... 

OUTPUT_FOLDER_PATH  = ".\\Dima_Oana_341\\clasic\\"

# test images for 5 dec
T_INPUT_FOLDER_PATH = ".\\testare\\clasic\\"
T_OUTPUT_NAME_PATTERN = "_predicted.txt" # 01 # 10 ....

class SudokuClassic:
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.thresholded_image = None
        self.contours = None
        self.sudoku_contour = None
        self.sudoku_predicted = []
        self.extracted_sudoku = None

    @staticmethod
    def show_an_image(window_name, image):
        '''
        function using for debug
        show an image 
        '''
        scale_percent = 20 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        cv.imshow(window_name, resized)
        cv.waitKey(0)

    def save_to_txt(self, output_path):
        '''
        save the solution 
        '''
        f =  open(output_path, "w+")
        for i in range(9):
            line = self.sudoku_predicted[i]
            for cell in line:
                f.write(cell)
            if i < 8: 
                f.write('\n')
        f.close()

    def process_image(self):
        '''
        40px border for safety - the sudoko contour incomplete case
        convert the image to black and white
        remove the noise for a better recognition of the contours
        I need the contours to be white to detect them
        '''
        color = self.image[20,20]
        border_color = (int(color[0]), int(color[1]), int(color[2]))
        border = 40
        top_left = (border // 2, border // 2)
        bottom_right = (self.image.shape[1] - border // 2,
                        self.image.shape[0] - border // 2)
        self.image = cv.rectangle(img=self.image,
                                 pt1=top_left,
                                 pt2=bottom_right,
                                 color=border_color, 
                                 thickness=border)

        grayed = cv.cvtColor(src=self.image, code=cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(src=grayed, ksize=(15, 15), sigmaX= 6)
        thresholded = cv.adaptiveThreshold(src=blurred, 
                                                maxValue=255,
                                                adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                thresholdType=cv.THRESH_BINARY,
                                                blockSize=33,
                                                C= 4)
        self.thresholded_image = cv.bitwise_not(src=thresholded)

        # Debugging
        # self.show_an_image("process_image", self.thresholded_image)

    def get_contours(self):
        '''
        detect the contours
        save them as an np.array of 2 points [x,y] (the begin point and the end point)
        '''
        contours_found = cv.findContours(image=self.thresholded_image,
                                        mode=cv.RETR_EXTERNAL,
                                        method=cv.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(cnts=contours_found)
        self.contours = sorted(self.contours, key=cv.contourArea, reverse=True)

    def iterate_through_contours(self):
        '''
        the get_contours() function can detect also the numbers in the sudoku
        I need a detail rate for getting just the squares
        '''
        for contour in self.contours:
            detail_rate = 0.02
            epsilon = detail_rate * cv.arcLength(curve=contour, closed=True)

            # I need to be 4 details because a square has 4 corners
            resulted_details = cv.approxPolyDP(curve=contour,
                                               epsilon=epsilon,
                                               closed=True)

            if len(resulted_details) == 4:
                # square dimensions
                _, _, width, height = cv.boundingRect(contour)
                square_surface = width * height

                # print("Square surface: ", square_surface)

                if square_surface > 500000:
                    # I find a sudoku 
                    # save the sudoku contour 
                    self.sudoku_contour = resulted_details

                    # Debug
                    # cv.drawContours(image=self.image, 
                    #                 contours=[resulted_details],
                    #                 contourIdx=-1,
                    #                 color=(0, 0, 255),  #red
                    #                 thickness=4)
                    # self.show_an_image("Contours: ", self.image)

                    break
    
    def get_cells_coordinates(self, cell_width, cell_height):
        '''
        calculates the upper left corner for each cell
        '''
        # [(x,y)], where (x,y) = the top left corner
        cells = []

        for i in range(0, 81):
            cells.append(
                        (cell_width * (i % 9),
                         cell_height * (i // 9))
                        )

            # Debug
            # self.extracted_sudoku = cv.circle(img=self.extracted_sudoku,
            #                                  center=(cell_width * (i % 9), cell_height * (i // 9)),
            #                                  radius=20,
            #                                  color=(0, 0, 255), #red
            #                                  thickness=-1)
        
        # Debug
        # self.show_an_image("Cells", self.extracted_sudoku)
        return cells

    @staticmethod
    def get_cells_with_numbers(cells, extracted_sudoku, cell_height, cell_width):
        '''
        get the IDs of the cells that contain a number
        '''
        cells_with_numbers = []

        # Debug
        if len(cells) != 81: 
            print("Something went wrong!")

        for i in range(len(cells)):

            # how much I ignore from the cells to avoid the margins
            padding = 40
            coordinate = cells[i]

            cell = extracted_sudoku[coordinate[1] + padding: coordinate[1] + cell_height - padding,
                                        coordinate[0] + padding: coordinate[0] + cell_width - padding].copy()
                
            # RGB image, I need to make them black and white again
            cell_grayed = cv.cvtColor(src=cell, code=cv.COLOR_BGR2GRAY)
            threshold = cv.threshold(src=cell_grayed,
                                         thresh=145,
                                         maxval=255,
                                         type=cv.THRESH_BINARY_INV)

            mean_value = threshold[1].mean()

            # Debug
            # print("Mean value: ", mean_value)

            if mean_value > 10: # the mean bias -> 10
                # i found a cell with a number
                cells_with_numbers.append(i)

        return cells_with_numbers

    def extract(self):
        '''
        calculate the corners of the sudoku matrix
        cut the image with the sudoku
        transforms a possible image rotated by translation
        process the cells
        extract the cells which contain numbers
        '''
        if self.sudoku_predicted is None:
            raise Exception("Sufoku predictec is none")
        elif self.sudoku_contour is None:
            print("Sudoku contour is none")
        else:
            # 4 points (x,y)
            corners = np.zeros((4, 2), dtype='float32')

            sudoku_contour_reshaped = self.sudoku_contour.reshape(4, 2)

            # calculate each of the 4 corners
            sum = sudoku_contour_reshaped.sum(axis=1)
            diff = np.diff(sudoku_contour_reshaped, axis=1)

            # top left corner -> min sum
            corners[0] = sudoku_contour_reshaped[np.argmin(sum)]
            # top right -> min diff
            corners[1] = sudoku_contour_reshaped[np.argmin(diff)]
            # bottom right corner -> max sum
            corners[2] = sudoku_contour_reshaped[np.argmax(sum)]
            # bottom left -> max diff
            corners[3] = sudoku_contour_reshaped[np.argmax(diff)]

            # Euclidian distance: sqrt((x2-x1)^2 + (y2 - y1)^2)
            right = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) 
                    + ((corners[1][1] - corners[2][1]) ** 2))
            left = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) 
                    + ((corners[0][1] - corners[3][1]) ** 2))
            bottom = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) 
                    + ((corners[2][1] - corners[3][1]) ** 2))
            top = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) 
                    + ((corners[1][1] - corners[0][1]) ** 2))

            # calculate the dimensions of the new sudoku image
            sudoku_width = max(int(top), int(bottom))
            sudoku_height = max(int(left), int(right))

            sudoku_matrix_template = np.array([
                                             [0, 0],
                                             [sudoku_width - 1, 0],
                                             [sudoku_width - 1, sudoku_height - 1],
                                             [0, sudoku_height - 1]
                                             ], 
                                             dtype='float32')
            
            # transforms the possibly rotated image by translating the corners
            perspective_transform = cv.getPerspectiveTransform(src=corners,
                                                               dst=sudoku_matrix_template)
            # self.image is RGB => extracted_sudoku is RGB
            self.extracted_sudoku = cv.warpPerspective(src=self.image,
                                                  M=perspective_transform,
                                                  dsize=(sudoku_width, sudoku_height))

            # Debug
            # self.show_an_image("The cut", extracted_sudoku)

            # calculate the dimensions for each cell (9x9 matrix)
            cell_width = self.extracted_sudoku.shape[1] // 9
            cell_height = self.extracted_sudoku.shape[0] // 9

            cells = self.get_cells_coordinates(cell_width, cell_height)

            cells_with_numbers = self.get_cells_with_numbers(cells,
                                                             self.extracted_sudoku,
                                                             cell_height,
                                                             cell_width)

            self.make_sudoku(cells_with_numbers)

    def make_sudoku(self, cells_with_numbers):
        '''
        generate the matrix of the sudoku => list [[], []]
        '''
        cell_number = 0
        for i in range(9):
            l = []
            for j in range(9):
                if cell_number in cells_with_numbers:
                    l.append('x')
                else:
                    l.append('o')
                cell_number +=1
            self.sudoku_predicted.append(l)

def find_differences(path1, path2, name_pattern):
    errors = 0
    error_files = []
    for i in range(1, 21):
        if i < 10:
            p1 = path1 + "0" + str(i) + "_gt.txt"
            p2 = path2 + "0" + str(i) + name_pattern
        else:
            p1 = path1 + str(i) + "_gt.txt"
            p2 = path2 + str(i) + name_pattern

        f1 = open(p1, "r")
        f2 = open(p2, "r")
        mistakes = 0
        for x in range (9):
            l1 = f1.readline()
            l2 = f2.readline()

            # Debug
            if len(l1)==0 or len(l2)==0:
                print("Error incomplete/empty file")
                break
            
            for j in range (9):
                if l1[j] != l2[j]:
                    mistakes +=1
                    if i not in error_files:
                        error_files.append(i)
                    break
        if mistakes != 0:
            errors +=1

    for file in error_files:
        print("File number ", file, " has mistakes")

    return errors 

def apply_for_all(output_folder_path, input_folder_path, name_pattern):
    '''
    Iterate through the folder and process every image
    '''
    for i in range (1, 21):
        if i < 10:
            output_path = output_folder_path + "0" + str(i) + name_pattern
            image_path =  input_folder_path + "0" + str(i) + ".jpg"
        else:
            output_path = output_folder_path + str(i) +  name_pattern
            image_path =  input_folder_path + str(i) + ".jpg"

        new_sudoku_classic = SudokuClassic(image_path)
        new_sudoku_classic.process_image()
        new_sudoku_classic.get_contours()
        new_sudoku_classic.iterate_through_contours()
        new_sudoku_classic.extract()
        new_sudoku_classic.save_to_txt(output_path)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    apply_for_all(OUTPUT_FOLDER_PATH,
                  A_INPUT_FOLDER_PATH,
                  T_OUTPUT_NAME_PATTERN)
    
    # errors = find_differences(A_EVALUATE_FOLDER_PATH,
    #                           OUTPUT_FOLDER_PATH,
    #                           T_OUTPUT_NAME_PATTERN)
    # print("There was " + str(errors) + " wrong files")
    
    