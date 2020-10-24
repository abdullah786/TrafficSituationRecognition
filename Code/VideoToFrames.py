import cv2,imutils
# Function to extract frames
def FrameCapture():
    # Path to video file
    vidObj = cv2.VideoCapture('C://Users//Abdullah//Downloads//WhatsApp Video 2020-04-20 at 21.52.09 (1).mp4')
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        # vidObj object calls read
        # function extract frames


        success, image = vidObj.read()
        rotated = imutils.rotate(image, 270)
        # Saves the frames with frame-count
        cv2.imwrite("framesss%d.jpg" % count, rotated)
        count += 1
# Driver Code
if __name__ == '__main__':
    # Calling the function
    FrameCapture()