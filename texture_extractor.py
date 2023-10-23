import os, sys
from PIL import Image
import pandas as pd
import numpy as np
from radiomics import firstorder, glcm, glszm, glrlm, ngtdm, gldm
import SimpleITK as sitk
import matplotlib.pyplot as plt
import six

class TextureExtractor:
    def __init__(self, clip_size=64, num_sample=100) -> None:
        self.clip_size = clip_size
        self.num_sample = num_sample
        self.feature_class_func = {
            "FO" : firstorder.RadiomicsFirstOrder,
            "GLCM" : glcm.RadiomicsGLCM,
            "GLSZM" : glszm.RadiomicsGLSZM,
            "GLRLM" : glrlm.RadiomicsGLRLM,
            "NGTDM" : ngtdm.RadiomicsNGTDM,
            "GLDM" : gldm.RadiomicsGLDM,
        }
        
    def parse_input_data(self, image_array, mask_array):
        image = sitk.GetImageFromArray( np.expand_dims(image_array, axis=0) )
        mask = sitk.GetImageFromArray( np.expand_dims(mask_array, axis=0))
        return image, mask

    def get_clip_image(self, image, clip_size):
        size = image.shape[0]
        center_x = int( np.random.uniform(clip_size//2, size-clip_size//2) )
        center_y = int( np.random.uniform(clip_size//2, size-clip_size//2) )
        clip_image = image[center_x-clip_size//2:center_x+clip_size//2, center_y-clip_size//2:center_y+clip_size//2]
        return clip_image

    def get_texture_feature(self, image_array, mask_array, settings={}):
        image, mask = self.parse_input_data(image_array, mask_array)
        result = {}
        for class_key in self.feature_class_func.keys():
            feature_extractor = self.feature_class_func[class_key](image, mask, **settings)
            feature_extractor.enableAllFeatures()
            for (feature_key, value) in six.iteritems(feature_extractor.execute()):
                result[f'{class_key}-{feature_key}'] = np.float64(value)
        
        return result
    
    def get_median_texture_feature_from_patch(self, image):
        if( min(image.shape) > self.clip_size ):
            print('ERROR')
            return 0
        
        mask = np.ones((self.clip_size, self.clip_size))
        feature_df = pd.DataFrame()
        for i in range(self.num_sample):
            clip_image = self.get_clip_image(image, self.clip_size)
            result = self.get_texture_feature(clip_image, mask)
            feature_df = pd.concat([feature_df, pd.DataFrame(result, index=[0])], ignore_index=True)
        return feature_df.median().to_dict()

if __name__ == "__main__":
    print('say, hello')
