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
    def __init__(self, root='./data/z-view/', extensions='.npz',
                 load_views=["z"],
                 norm_and_clip=True,
                 adc_scale=50.0,
                 load_meta_data=False,
                 verbose=False):
        #super().__init__( root=root, loader=lartpcDataset.load_data, extensions=extensions )
        super().__init__( root=root, loader=self.load_data, extensions=extensions ) # will this work?
        self.load_views = load_views
        lartpcDataset.NORM_AND_CLIP = norm_and_clip
        if lartpcDataset.NORM_AND_CLIP:
            lartpcDataset.ADC_SCALE = adc_scale
            
        self.metadata = {}
        self.load_meta_data = load_meta_data
        if self.load_meta_data:
            for part in lartpcDataset.CLASSNAMES:
                metafile = root+"/../metadata-"+part+".txt"
                partdata = {}
                with open(metafile) as f:
                    metalines = f.readlines()
                    for l in metalines:
                        idnum,pid,px,py,px,pmom = l.strip().split(",")
                        partdata[int(idnum)] = (float(pid),float(px),float(py),float(px),float(pmom))
                self.metadata[part] = partdata
                if verbose: print("Particle Meta-data loaded for ",part,": ",len(self.metadata[part]))

    def load_data(self, inp):
        #print("lartpcDataset.load_data: path=",inp)

        with open(inp, 'rb') as f:
            npin = np.load(f)
            x = npin['arr_0']
            if lartpcDataset.NORM_AND_CLIP:
                x = np.expand_dims( np.clip( x/lartpcDataset.ADC_SCALE, 0, 10.0 ), axis=0 )
        
        if self.load_meta_data:
            metadata = self.get_file_meta_data(inp)
            m = np.array( metadata )
            return x,m
        else:
            return x

    def get_file_meta_data(self, inp ):
        # parse filename: e.g. electron-000000.npz
        x = os.path.basename(inp).strip()[:-4].split("-")
        idnum = int( x[-1] )
        classname = x[0]
        #print("idnum: ",idnum," classname=",classname)
        return self.metadata[classname][idnum]

if __name__ == "__main__":
    """
    a quick test program
    """
    import torch
    
    data = lartpcDataset(root="./data/z-view/",load_meta_data=False,verbose=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=4,
        shuffle=True)
    
    it = iter(test_loader)
    batch = next(it)
    print(len(batch))
    data = batch[0]
    labels = batch[1]
    print(data)    
    if len(data)==2:
        # metadata tensor is returned
        meta = data[1]
        imgs = data[0]
    else:
        imgs = data
        meta = None
    print("images: ",imgs)
    print("labels: ",labels)
    if meta is not None:
        print("meta: ",meta)
    #img0, meta0 = batch[0]
    #print(img0)
    #print(meta0)
    #print(img0[ img0>0 ].mean())
