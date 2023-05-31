import cv2


# class to handle cropping options


class Cut:
    def __init__(self):
        pass

    # function to crop image
    # Parameters: image to crop, contour, and the image number
    def crop(self, image, contours, num):        
        # cycles through all the contours to crop all
        for c in contours:
            # creates an approximate rectangle around contour
            x, y, w, h = cv2.boundingRect(c)
            # Only crop decently large rectangles
            
                # pulls crop out of the image based on dimensions
            new_img = image[y:y + h, x:x + w]
                # writes the new file in the Crops folder
                # cv2.imwrite('Crops/' + 'crop_' + str(num) +
                #             '.png', new_img)
            # returns a number incremented up for the next file name
            return new_img
