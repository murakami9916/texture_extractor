import os, sys
from PIL import Image
import pandas as pd
import numpy as np
from radiomics import firstorder, glcm, glszm, glrlm, ngtdm, gldm
import SimpleITK as sitk
import matplotlib.pyplot as plt
import six

class TextureExtractor:
    def __init__(self) -> None:
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
        
    def get_texture_feature(self, image_array, mask_array, settings={}):
        
        image, mask = self.parse_input_data(image_array, mask_array)
        
        result = {}
        for class_key in self.feature_class_func.keys():
            feature_extractor = self.feature_class_func[class_key](image, mask, **settings)
            feature_extractor.enableAllFeatures()
            for (feature_key, value) in six.iteritems(feature_extractor.execute()):
                result[f'{class_key}-{feature_key}'] = np.float64(value)
        
        return result

if __name__ == "__main__":
    print('say, hello')
