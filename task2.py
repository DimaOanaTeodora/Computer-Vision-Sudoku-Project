import cv2 as cv
import numpy as np
import imutils 
import os 

A_INPUT_FOLDER_PATH =  ".\\antrenare\\jigsaw\\"
A_EVALUATE_FOLDER_PATH = ".\\antrenare\\results\\jigsaw\\"
A_OUTPUT_NAME_PATTERN = "_gt.txt" # 01 # 10 .... 

OUTPUT_FOLDER_PATH  = ".\\Dima_Oana_341\\jigsaw\\"

# test images for 5 dec
T_INPUT_FOLDER_PATH = ".\\testare\\jigsaw\\"
T_OUTPUT_NAME_PATTERN = "_predicted.txt" # 01 # 10 .... 

class SudokuJigsaw:
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.thresholded_image = None
        self.contours = None
        self.sudoku_contour = None
        self.sudoku_predicted = []
        self.extracted_sudoku = None
        self.sudoku_without_thin_lines = None
        self.zones_for_cells = []
        self.threshold = None
        self.COLORS = {
            '0': (255, 255, 255), # white because the initial canva is fill with white
            '1': (0, 0, 0), # black 
            '2': (100, 100, 100), # gray
            '3': (255, 0, 0), # red
            '4': (0, 0, 255), # blue 
            '5': (0, 255, 0), # green
            '6': (0, 0, 100), # dark blue
            '7': (0, 255, 255), # light blue
            '8': (100, 0, 100), # purple
            '9': (255, 255, 0) # yellow
        }

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
        if len(self.sudoku_predicted) == 9: 
            for i in range(9):
                line = self.sudoku_predicted[i]
                for cell in line:
                    f.write(cell)
                if i < 8: 
                    f.write('\n')
        f.close()

    def process_raw_image(self):
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
        # self.show_an_image("process_raw_image", self.thresholded_image)

    def process_extracted_sudoku(self):
        '''
        convert the image to black and white
        remove the noise for a better recognition of the contours
        remove the gradient
        '''
        grayed = cv.cvtColor(src=self.extracted_sudoku, code=cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(src=grayed,
                                  ksize=(5, 5),
                                  sigmaX=3)
        _, self.threshold = cv.threshold(src=blurred,
                                        thresh=80,
                                        maxval=255, 
                                        type=cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        # Debugging
        # self.show_an_image("process_extracted_sudoku", self.threshold)
    
    def remove_thin_lines(self):
        '''
        erode then dilate the image
        remove thin lines and keep the thick ones
        convert black to white and white to black 
        '''
       
        erode_kernel = np.ones(shape=(19, 19), dtype=np.uint8)    
        # MORPH_OPEN: erode then dilate 
        self.sudoku_without_thin_lines = cv.morphologyEx(src=self.threshold,
                                        op=cv.MORPH_OPEN,
                                        kernel=erode_kernel)
        self.sudoku_without_thin_lines = cv.bitwise_not(self.sudoku_without_thin_lines)

        # Debugging
        # self.show_an_image("without thin lines", self.sudoku_without_thin_lines)

    def draw_extra_border(self):
        '''
        30px margin around the table for safety
        '''
        border = 30
        top_left = (border // 2, border // 2)
        bottom_right = (self.sudoku_without_thin_lines.shape[1] - border // 2,
                        self.sudoku_without_thin_lines.shape[0] - border // 2)
        self.sudoku_without_thin_lines = cv.rectangle(img=self.sudoku_without_thin_lines,
                                                     pt1=top_left,
                                                     pt2=bottom_right,
                                                     color=(0, 0, 0), # black
                                                     thickness=border)
        # Debugging
        # self.show_an_image("with extra border", self.sudoku_without_thin_lines)
    
    def get_contours_f1(self):
        '''
        detect the contours for tresholded raw image
        save them as an np.array of 2 points [x,y] (the begin point and the end point)
        '''
        contours_found = cv.findContours(image=self.thresholded_image,
                                        mode=cv.RETR_EXTERNAL,
                                        method=cv.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(cnts=contours_found)
        self.contours = sorted(self.contours, key=cv.contourArea, reverse=True)
    
    def get_contours_f2(self):
        '''
        detect the contours for the processed imagine (without thin lines)
        save them as an np.array of 2 points [x,y] (the begin point and the end point)
        '''
        contours = cv.findContours(image=self.sudoku_without_thin_lines,
                                   mode=cv.RETR_EXTERNAL,
                                   method=cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        return contours

    def iterate_through_contours_f1(self):
        '''
        the get_contours_f1() function can detect also the numbers in the sudoku
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
                    #                 color=colors[str(number + 1)],  #red
                    #                 thickness=cv.FILLED)
                    # self.show_an_image("Contours: ", self.image)

                    break
    
    def iterate_through_contours_f2(self, contours):
            '''
            the get_contours_f2() function detect the contours of the zones
            I need a small detail rate for getting all the details
            '''
            # Back to RGB for coloring
            self.sudoku_without_thin_lines = cv.cvtColor(src=self.sudoku_without_thin_lines,
                                                     code=cv.COLOR_GRAY2RGB)
            for i in range(len(contours)):
                contour = contours[i]
                
                detail_rate = 0.00002
                epsilon = detail_rate * cv.arcLength(curve=contour, closed=True)

                resulted_details = cv.approxPolyDP(curve=contour,
                                                   epsilon=epsilon,
                                                   closed=True)

                # white canva (just the black contours)
                cv.drawContours(image=self.sudoku_without_thin_lines,
                                contours=[resulted_details],
                                contourIdx=-1,
                                color=self.COLORS['0'],
                                thickness=cv.FILLED)
            
            # Debug 
            # self.show_an_image("white zones", self.sudoku_without_thin_lines)
               
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

    def remove_mistakes(self, cells_coordinates, contours, cell_width, cell_height):
            '''
            removing possible mistakes
            test if a cell is in a zone
            color of the cells => color of the zones
            '''
            current_zone = 1
            for cell_coordinate in cells_coordinates:
                
                padding = 100
                # extract cells (- padding)
                cell = self.sudoku_without_thin_lines[cell_coordinate[1] + padding:cell_coordinate[1] + cell_height - padding,
                                                      cell_coordinate[0] + padding:cell_coordinate[0] +  cell_width - padding].copy()
                average_color = cv.mean(cell)[:3]

                e1 = abs(self.COLORS['0'][0] - average_color[0])
                e2 = abs(self.COLORS['0'][1] - average_color[1])
                e3 = abs(self.COLORS['0'][2] - average_color[2])
                
                if e1 <= 5 and e2 <= 5 and e3 <=5:
                    for contour in contours:
                        cell_is_inside = cv.pointPolygonTest(contour=contour,
                                                            pt=(cell_coordinate[0] + padding, cell_coordinate[1] + padding),
                                                            measureDist=False)
                        if cell_is_inside > 0:
                            # color the zone 
                            cv.drawContours(image=self.sudoku_without_thin_lines,
                                            contours=[contour],
                                            contourIdx=-1,
                                            color=self.COLORS[str(current_zone)],
                                            thickness=cv.FILLED)
                            current_zone += 1
                            break
            
            # Debug
            # self.show_an_image("Colored (without mistakes)", self.sudoku_without_thin_lines)

    def extract_zones_for_cells(self, cells, cell_width, cell_height):
        '''
        Save the colored zone for each cell in self.zones_for_cells
        '''
        for i in range(len(cells)):
                cell = cells[i]
                padding = 100 # remove possible contours

                # extract the cell
                cell = self.sudoku_without_thin_lines[cell[1] + padding:cell[1] + cell_height - padding,
                                                      cell[0] + padding:cell[0] + cell_width - padding].copy()
                
                # calculate the average color for the cell 
                # for the recognition of the zone
                average_color = cv.mean(cell)[:3]

                for zone_number in self.COLORS.keys():
                    color = self.COLORS[zone_number]
                    # calculate the errors
                    e1 = abs(color[0] - average_color[0])
                    e2 = abs(color[1] - average_color[1])
                    e3 = abs(color[2] - average_color[2])

                    # small errors < 5
                    if e1 <= 5 and e2 <= 5 and e3 <= 5:
                        self.zones_for_cells.append(zone_number)
                        break

    def extract(self):
        '''
        calculate the corners of the sudoku matrix
        cut the image with the sudoku
        transforms a possible image rotated by translation
        process the cells
        extract the contours of the zones
        colorate the zones
        extract the cells which contain numbers
        extract the zones for cells
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

            sudoku_matrix = np.array([
                                     [0, 0],
                                     [sudoku_width - 1, 0],
                                     [sudoku_width - 1, sudoku_height - 1],
                                     [0, sudoku_height - 1]
                                     ], 
                                     dtype='float32')
            
            # transforms the possibly rotated image by translating the corners
            perspective_transform = cv.getPerspectiveTransform(src=corners,
                                                               dst=sudoku_matrix)
            # self.image is RGB => extracted_sudoku is RGB
            self.extracted_sudoku = cv.warpPerspective(src=self.image,
                                                  M=perspective_transform,
                                                  dsize=(sudoku_width, sudoku_height))

            # Debug
            # self.show_an_image("The cut", extracted_sudoku)

            self.process_extracted_sudoku()

            self.remove_thin_lines()

            self.draw_extra_border()

            contours = self.get_contours_f2()
        
            self.iterate_through_contours_f2(contours)

            # calculate the dimensions for each cell (9x9 matrix)
            cell_width = self.extracted_sudoku.shape[1] // 9
            cell_height = self.extracted_sudoku.shape[0] // 9

            cells = self.get_cells_coordinates(cell_width, cell_height)

            self.remove_mistakes(cells, contours, cell_width, cell_height) 

            self.extract_zones_for_cells(cells, cell_width, cell_height)

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
                    l.append(self.zones_for_cells[cell_number] + 'x')
                else:
                    l.append(self.zones_for_cells[cell_number] + 'o')
                cell_number +=1
            self.sudoku_predicted.append(l)

def find_differences(path1, path2, name_pattern):
    errors = 0
    error_files = []
    for i in range(1, 41):
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
    for i in range (1, 41):
        if i < 10:
            output_path = output_folder_path + "0" + str(i) + name_pattern
            image_path =  input_folder_path + "0" + str(i) + ".jpg"
        else:
            output_path = output_folder_path + str(i) +  name_pattern
            image_path =  input_folder_path + str(i) + ".jpg"

        # Debug
        # print("Image ", i, ":")

        new_sudoku_jigsaw = SudokuJigsaw(image_path)
        new_sudoku_jigsaw.process_raw_image()
        new_sudoku_jigsaw.get_contours_f1()
        new_sudoku_jigsaw.iterate_through_contours_f1()
        new_sudoku_jigsaw.extract()
        new_sudoku_jigsaw.save_to_txt(output_path)

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

   
    