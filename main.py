from experiments.mslesseg_experiments import MSLesSegExperiments

def main():
    experiments = MSLesSegExperiments()
    experiment = experiments.get_super_resolution()
    # experiment = experiments.get_lr_segmentation()
    # experiment = experiments.get_hr_segmentation()
    # experiment = experiments.get_frozen_sr_frozen_seg()
    # experiment = experiments.get_frozen_sr_trainable_seg()
    # experiment = experiments.get_trainable_sr_frozen_seg()
    # experiment = experiments.get_joint_sr_seg_e2e()
    # experiment = experiments.get_joint_sr_seg_combined()

    experiment.run()
    # experiment.test()

if __name__ == "__main__":
    main()