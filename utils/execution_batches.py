
def execute_training_batch(experiments):
    sr_experiment = experiments.get_super_resolution()
    sr_experiment.run()

    lr_seg_experiment = experiments.get_lr_segmentation()
    lr_seg_experiment.run()

    hr_seg_experiment = experiments.get_hr_segmentation()
    hr_seg_experiment.run()

    frozen_sr_trainable_seg_experiment = experiments.get_frozen_sr_trainable_seg()
    frozen_sr_trainable_seg_experiment.run()

    trainable_sr_frozen_seg_experiment = experiments.get_trainable_sr_frozen_seg()
    trainable_sr_frozen_seg_experiment.run()

    joint_sr_seg_e2e_experiment = experiments.get_joint_sr_seg_e2e()
    joint_sr_seg_e2e_experiment.run()

    joint_sr_seg_combined_experiment = experiments.get_joint_sr_seg_combined()
    joint_sr_seg_combined_experiment.run()

def execute_testing_batch(experiments):
    sr_experiment = experiments.get_super_resolution()
    sr_experiment.test()

    lr_seg_experiment = experiments.get_lr_segmentation()
    lr_seg_experiment.test()

    hr_seg_experiment = experiments.get_hr_segmentation()
    hr_seg_experiment.test()

    frozen_sr_frozen_seg_experiment = experiments.get_frozen_sr_frozen_seg()
    frozen_sr_frozen_seg_experiment.test()

    frozen_sr_trainable_seg_experiment = experiments.get_frozen_sr_trainable_seg()
    frozen_sr_trainable_seg_experiment.test()

    trainable_sr_frozen_seg_experiment = experiments.get_trainable_sr_frozen_seg()
    trainable_sr_frozen_seg_experiment.test()

    joint_sr_seg_e2e_experiment = experiments.get_joint_sr_seg_e2e()
    joint_sr_seg_e2e_experiment.test()

    joint_sr_seg_combined_experiment = experiments.get_joint_sr_seg_combined()
    joint_sr_seg_combined_experiment.test()