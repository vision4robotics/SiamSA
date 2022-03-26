from .uav10fps import UAV10Dataset
from .uamt100 import UAMT100
from .uamt20l import UAMT20L

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'UAV123_10fps' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAMT100' in name:
            dataset = UAMT100(**kwargs)
        elif 'UAMT20L' in name:
            dataset = UAMT20L(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

