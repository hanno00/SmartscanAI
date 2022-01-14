import open3d as o3d

# import custom classes
from Augmentation import Augmentation
from Preprocessing import Preprocessing
from Agent import Agent

# settings
# Augmentation
augment_new_pointclouds = False
original_pointclouds = "original_pointclouds"
augmented_pointclouds = "sphere"
save_augmented_as_csv = False

# Preprocessing
preprocess_pointclouds_to_size = False
preprocessing_size = 1000
downsample_voxelsize = 18
preprocessed_pointclouds = "preprocessed_pointclouds"
distortion = 20

# Training
train = False
continue_training = True
training_iterations = 100
steps_per_iteration = 20
trained_model_filepath = "trained_models/PPO/testing"
verbose_training = True

# Predicting
predict = True
prediction_input_filepath = "preprocessed_pointclouds/hydrant.ply"
predicted_pointclouds = "result/pointcloud_result"
verbose_prediction = False

def Main():

    if augment_new_pointclouds:
        Augmentation.augment_folder(original_pointclouds,augmented_pointclouds,distortion,save_augmented_as_csv)
    
    if preprocess_pointclouds_to_size:
        Preprocessing.convert_folder(augmented_pointclouds,preprocessed_pointclouds,preprocessing_size,distortion,downsample_voxelsize)

    if train:
        Agent.training(preprocessed_pointclouds,trained_model_filepath,training_iterations,continue_training,steps_max=steps_per_iteration,size_model=preprocessing_size, prints=verbose_training)
    
    if predict:
        pcd = o3d.io.read_point_cloud(prediction_input_filepath)
        Agent.predict(trained_model_filepath,pcd,preprocessed_pointclouds,predicted_pointclouds,prints=verbose_prediction,max_steps=steps_per_iteration,model_size=preprocessing_size)
        
Main()