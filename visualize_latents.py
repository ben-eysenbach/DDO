import matplotlib.pyplot as plt
import numpy as np
import cv2
import data
import time
import sys
from logger import Logger
import os
from tqdm import tqdm
import json

exp_name = sys.argv[1]
with open(os.path.join(exp_name, 'args.json')) as f:
    game = json.load(f)['game']
traj_index = int(sys.argv[2])
iteration = int(sys.argv[3])

logger = Logger(exp_name, frozen=True)
z_vec = logger.load_z(iteration, traj_index)
image_vec = data.load_images(game, traj_index)

folder = os.path.join(exp_name, 'videos')
filename = os.path.join(folder, '%d_%d.avi' % (iteration, traj_index))
if not os.path.exists(folder):
    os.makedirs(folder)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(filename, fourcc, 30.0, (500, 500))


for (z, filename) in tqdm(zip(z_vec, image_vec)):
    img = cv2.imread(filename)
    img = cv2.resize(img, (500, 500), cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(z), (400, 50), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    video.write(img)
    # cv2.imshow('frame', img)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

video.release()

