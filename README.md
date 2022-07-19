# Creating TFRecords From Waymo Dataset

## Objective
The purpose of this is to convert the data from the Waymo Open Dataset into the tf record format used by the Tensorflow Object Detection API.

## Details
Each tf record files contains the data for an entire trip made by the car, meaning that 
it contains images from the different cameras as well as LIDAR data. Because we want 
to keep our dataset small, we are implementing the `create_tf_example` function to 
create cleaned tf records files.

I am using the Waymo Open Dataset github repository to parse the raw tf record files.

This task is part of the Self-Driving Car Engineer Nanodegree