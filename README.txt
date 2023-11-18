Benjamin Plate
3035904988
bplate@berkeley.edu
https://inst.eecs.berkeley.edu/~cs180/fa23/upload/files/proj5/bplate

main.py is the main python file that generates the images toward completing the task here. As is, the file will train the full NeRF neural network on CPU, but I've also included main-gpu.py which I ran on Google Colab after pip installing viser. You can observe the results from part 1 by uncommenting the large section in the beginning, and you can view the viser results by uncommenting either of the functions on lines 296 or 297. The EPOCHS constant determines iterations are run for training, the IMAGE_WIDTH and IMAGE_HEIGHT values should be changed to fix the image width and height, and SAMPLES determines how many validation runs on the first network are saved to disk as images (I ran this part on CPU where this was a significant factor on performance).

If there is an issue with something to do with jpg/png, I converted all my outputs from png to jpg for submission. It may fix the issue to switch this back.

Create a Python virtual environment and use pip install -r requirements.txt to get the necessary packages.
