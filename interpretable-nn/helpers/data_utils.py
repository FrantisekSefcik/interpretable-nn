

class LggHggGenerator:

    def __init__(self, image_generator, seg_generator, image_dir, seg_dir,
                 seed=42):
        self.image_generator = image_generator
        self.seg_generator = seg_generator
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.seed = seed
        self.params = {
            'shuffle': False,
            'classes': ['LGG', 'HGG'],
            'color_mode': "rgb",
            'target_size': (240, 240),
            'class_mode': 'sparse',
            'seed': self.seed
        }

    def get_image_generator(self, batch_size=16):
        return self.image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=self.image_dir,
                                                        classes=['LGG', 'HGG'],
                                                        color_mode="rgb",
                                                        target_size=(240, 240),
                                                        class_mode='sparse',
                                                        seed=self.seed)

    def get_seg_generator(self, batch_size=16):
        return self.seg_generator.flow_from_directory(batch_size=batch_size,
                                                      directory=self.seg_dir,
                                                      classes=['LGG', 'HGG'],
                                                      color_mode="grayscale",
                                                      target_size=(240, 240),
                                                      class_mode='sparse',
                                                      seed=self.seed)

    def get_image_seg_generator(self, batch_size=16):
        return zip(self.get_image_generator(batch_size),
                   self.get_seg_generator(batch_size))