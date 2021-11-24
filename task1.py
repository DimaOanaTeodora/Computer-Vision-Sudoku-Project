import cv2 as cv
import numpy as np
import imutils 
import os 

INPUT_FOLDER_PATH =  ".\\antrenare\\images\\"
OUTPUT_FOLDER_PATH = ".\\results\\"
OUTPUT_IMAGE_PATTERN = "" #TODO !

class SudokuClassic:
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.tresholded_image = None
        self.contours = None
        self.sudoku_contours = None
        self.sudoku_predicted = []

    def show_a_image(window_name, image):
        cv.imshow(window_name, image)
        cv.waitKey(0)

    def save_to_txt(self, output_path):
        f =  open(output_path, "w+")
       
        for line in self.sudoku_predicted:
            for cell in line:
                f.write(cell)
            f.write('\n')
        f.close()

    def process_image(self):
        grayed_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(grayed_image, (15, 15), 6)
        thresholded_image = cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 4)
        self.thresholded_image = cv.bitwise_not(thresholded_image)

    def get_contours(self):
        contours = cv.findContours(self.thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        self.contours = sorted(contours, key=cv.contourArea, reverse=True)

    def iterate_through_contours(self):
        
        #sudoku_contour = None

        # Iterate through contours
        for c in self.contours:

            # Convex Hull
            epsilon = 0.02 * cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon, True)

            # Find the bounding rectangle of contour to check its size
            x, y, w, h = cv.boundingRect(c)

            # Draw contour if square and if size of box is higher than threshold (so that text cannot be picked up)
            if len(approx) == 4 and w * h > 500000:

                # TODO: show a image

                # Save the contour as the sudoku_contour
                self.sudoku_contour = approx
                break
    
    def wrap_the_image(self):

        if self.sudoku_predicted is None:
            raise Exception("A aparut o eroare la aceasta imagine!")
        else:

            # Order points from contour
            rect = np.zeros((4, 2), dtype='float32')
            sudoku_contour_reshaped = self.sudoku_contour.reshape(4, 2)

            # Calculate the sum and difference of x and y of each corner
            points_sum = sudoku_contour_reshaped.sum(axis=1)
            points_diff = np.diff(sudoku_contour_reshaped, axis=1)

            # First element will be top left and third will be bottom right (minimum sum and maximum sum)
            rect[0] = sudoku_contour_reshaped[np.argmin(points_sum)]
            rect[2] = sudoku_contour_reshaped[np.argmax(points_sum)]

            # Second element will be top right and last will be bottom left (minimum and maximum diff)
            rect[1] = sudoku_contour_reshaped[np.argmin(points_diff)]
            rect[3] = sudoku_contour_reshaped[np.argmax(points_diff)]

            # Calculate the width of the new reshaped image
            width_bottom = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            width_top = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width_max = max(int(width_top), int(width_bottom))

            # Calculate the height of the new reshaped image
            height_right = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            height_left = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            height_max = max(int(height_left), int(height_right))

            # TODO: afiseaza imagine

            # Construct the size of the new image and save it in a matrix
            sudoku_matrix_template = np.array([[0, 0], [width_max - 1, 0], [width_max - 1, height_max - 1], [0, height_max - 1]], dtype='float32')
            perspective_transform = cv.getPerspectiveTransform(rect, sudoku_matrix_template)
            sudoku_contour_warped = cv.warpPerspective(self.image, perspective_transform, (width_max, height_max))

            # Calculate step size for each cell
            width_step = sudoku_contour_warped.shape[1] // 9
            height_step = sudoku_contour_warped.shape[0] // 9

            # Array to hold each cell upper left corner coord
            coords = []

            # Calculate the upper left coord of each cell
            for c in range(0, 81):
                coord = ((c % 9) * width_step, (c // 9 * height_step))
                coords.append(coord)

                # TODO: afiseaza imagine

            # Array to hold indices of cells that contain numbers
            cells_with_numbers = []

            for i, coord in enumerate(coords):

                # Add padding to remove borders
                padding = 40
                cell_mean_bias = 10

                cell = sudoku_contour_warped[coord[1] + padding:coord[1] + height_step - padding, coord[0] + padding:coord[0] + width_step - padding].copy()
                cell_grayed = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
                cell_threshold = cv.threshold(cell_grayed, 145, 255, cv.THRESH_BINARY_INV)[1]

                # If there is something inside the cell (if the mean of the cell is higher than the cell_mean_bias) append to the final array
                if cell_threshold.mean() > cell_mean_bias:
                    cells_with_numbers.append(i)

                # TODO: afiseaza imagine

            self.make_sudoku(cells_with_numbers)

            # TODO: afiseaza imagine

    
    def make_sudoku(self, cells_with_numbers):
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

def find_differences(path1, path2):
    errors = 0
    for i in range(1, 21):
        if i < 10:
            p1 = path1 + "0" + str(i) + "_gt.txt"
            p2 = path2 + "0" + str(i) + "_gt.txt"
        else:
            p1 = path1 + str(i) + "_gt.txt"
            p2 = path2 + str(i) + "_gt.txt"

        f1 = open(p1, "r")
        f2 = open(p2, "r")
        mistakes = 0
        for i in range (9):
            l1 = f1.readline()
            l2 = f2.readline()
            for j in range (9):
                if l1[j] != l2[j]:
                    mistakes +=1
        if mistakes != 0:
            errors +=1
    return errors 

def apply_for_all():
    for i in range (1, 21):
        if i < 10:
            output_path = OUTPUT_FOLDER_PATH + "0" + str(i) + "_gt.txt"
            image_path = INPUT_FOLDER_PATH + "0" + str(i) + ".jpg"
        else:
            output_path = OUTPUT_FOLDER_PATH + str(i) + "_gt.txt"
            image_path = INPUT_FOLDER_PATH + str(i) + ".jpg"

        new_sudoku_classic = SudokuClassic(image_path)
        new_sudoku_classic.process_image()
        new_sudoku_classic.get_contours()
        new_sudoku_classic.iterate_through_contours()
        new_sudoku_classic.wrap_the_image()
        new_sudoku_classic.save_to_txt(output_path)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    apply_for_all()
    
    errors = find_differences(".\\antrenare\\results\\", OUTPUT_FOLDER_PATH)
    print("Au fost " + str(errors) + " imagini prezise gresit")