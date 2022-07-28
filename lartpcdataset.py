import os,sys
import numpy as np
import torch
import torchvision
import torchvision.datasets

class lartpcDataset( torchvision.datasets.DatasetFolder ):

    VIEWS = ["x","y","z"]
    CLASSNAMES = ["electron","gamma","muon","proton","pion"]
    NORM_AND_CLIP = True
    ADC_SCALE = 50.0
    def __init__(self, root='./data', extensions='.npz', load_views=["z"], norm_and_clip=True, adc_scale=50.0 ):
        super().__init__( root=root, loader=lartpcDataset.load_data, extensions=extensions )
        self.metadata = {}
        self.load_views = load_views
        lartpcDataset.NORM_AND_CLIP = norm_and_clip
        if lartpcDataset.NORM_AND_CLIP:
            lartpcDataset.ADC_SCALE = adc_scale

    def load_data(inp):
        print("lartpcDataset.load_data: path=",inp)
        with open(inp, 'rb') as f:
            npin = np.load(f)
            x = npin['arr_0']
            if lartpcDataset.NORM_AND_CLIP:
                x = np.clip( x/lartpcDataset.ADC_SCALE, 0, 10.0 )
        return x


if __name__ == "__main__":
    """
    a quick test program
    """
    import torch
    
    data = lartpcDataset(root="./data")
    test_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=4,
        shuffle=True)
    
    it = iter(test_loader)
    batch = next(it)
    print(batch[0].sum())
    print(batch[1])
    x = batch[0]
    print(x[ x>0 ].mean())
