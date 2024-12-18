class Constants():
    ct_only_filtered2 = 'D:/thesis/ct_only_filtered_2'
    ct_only_cleaned = 'D:/thesis/ct_only_cleaned'
    ct_only_cleaned_resized = 'D:/thesis/ct_only_cleaned_resized'
    organized_dataset_2 = "D:/thesis/organized_dataset_2"
    model_outputs = "D:/thesis/model_runs"
    n_gpus = 0   # 0 = CPU, -1 = all available GPUs
    default_felz_params = {'scale':150, 'sigma':0.7, 'min_size':50}
    default_perms = 'D:/thesis/ct_only_cleaned/permutations.npy'
    pretrained_jigsaw = 'D:/thesis/model_runs/pretrained_jigsaw_resnet34/logs/default/version_0/checkpoints/epoch=9-step=48689.ckpt'