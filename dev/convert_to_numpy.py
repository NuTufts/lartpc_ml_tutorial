import os,sys
import numpy as np
from larcv import larcv
larcv.load_pyutil()

"""
The single-particle classification rootfiles are expected to carry the following trees.

  KEY: TTree	image2d_data_tree;1	data tree
  KEY: TTree	particle_mctruth_tree;1	mctruth tree

"""

VIEWS = ["x","y","z"]
CLSNAMES = ["electron","gamma","muon","proton","pion"]
    
def convert_rootfile( rootfile, out_folder="./" ):
    io = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
    io.add_in_file( rootfile )
    io.initialize()
    nentries = io.get_n_entries()
    print("Number of entries in file: ",nentries)

    
    # make output folders
    for view in VIEWS:
        for i,clsname in enumerate(CLSNAMES):
            os.system("mkdir -p %s/%s-view/%02d-%s"%(out_folder,view,i,clsname))

    labelcount = {}
    metafiles = {}
    classid = {}
    for cid,clsname in enumerate(CLSNAMES):
        labelcount[clsname] = 0
        metafiles[clsname] = open("%s/metadata-%s.txt"%(out_folder,clsname),'w')
        classid[clsname] = cid
        
    for ientry in range( nentries ):
        io.read_entry(ientry)
        ev_img2d = io.get_data( larcv.kProductImage2D, "data" )
        ev_truth = io.get_data( larcv.kProductParticle, "mctruth" )
        if ientry%1000==0:
            verbose = True
        else:
            verbose = False

        if verbose:
            print("Entry ",ientry)
            print("  num of images: ",ev_img2d.as_vector().size())
            print("  truth entries: ",ev_truth.as_vector().size())

        # get truth
        mcpart = ev_truth.as_vector().at(0)
        pdg = mcpart.pdg_code()
        mom = mcpart.p()
        px = mcpart.px()
        py = mcpart.py()
        pz = mcpart.pz()
        edep = mcpart.energy_deposit()

        if pdg in [-11,11]:
            labelname="electron"
        elif pdg in [22]:
            labelname="gamma"
        elif pdg in [-13,13]:
            labelname="muon"
        elif pdg in [2212]:
            labelname="proton"
        elif pdg in [211,-211]:
            labelname="pion"
        else:
            raise ValueError("unrecognized PDG code: ",pdg)
        if verbose: print("  labelname=",labelname)
        
        for p,view in enumerate(VIEWS):
            img2d = ev_img2d.as_vector().at(p)
            imgnp = larcv.as_ndarray( img2d )
            if verbose: print("  [%d] view-%s: "%(p,view),imgnp.shape)
            
            with open(out_folder+"/%s-view"%(view)+"/%02d-"%(classid[labelname])+labelname+"/"+"%s-%06d.npz"%(labelname,labelcount[labelname]), 'wb') as f:
                np.savez_compressed(f, imgnp)

        print("%d,%d,%.2f,%.2f,%.2f,%.2f"%(labelcount[labelname],pdg,px,py,pz,mom),file=metafiles[labelname])
        labelcount[labelname] += 1
              
        if False and ientry>=1000:
            break
        
    return


if __name__ == "__main__":

    convert_rootfile("train_50k.root")
    
