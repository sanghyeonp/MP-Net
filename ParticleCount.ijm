//MP-VAT (Microplastics Visual Analysis Tool) v1.0
//Created by J.C.Prata, V. Reis, J. Matos, J.P.da Costa, A.Duarte, T.Rocha-Santos 2019

macro "MPVAT Action Tool - C369D32D72D92Da2Db2Dc2D33D43D63D73D93Dd3D34D54D74D94Dd4D35D55D75D95Da5Db5Dc5D36D76D96D37D77D97D3aD5aD7aD8aD9aDbaDcaDdaD3bD5bD7bD9bDcbD3cD5cD7cD8cD9cDccD3dD5dD7dD9dDcdD4eD7eD9eDce"{

//Color inversion, 8-bit conversion and automatic threshold
//run("Invert");
//run("8-bit");
//setAutoThreshold("MaxEntropy");
//setOption("BlackBackground", false);
//run("Convert to Mask");
//run("Set Measurements...", "area shape feret's display redirect=None decimal=3");

//title
title = getTitle();
setBatchMode(true);
mpshape = newArray ("Fibers", "Fragments", "Particles");
for (i = 0; i < mpshape.length; i++) {
    selectWindow(title);
    run("Duplicate...", " ");
    rename(mpshape[i]);
}
run("Tile");

//Analyze Fibers
selectWindow(mpshape[0]);
run("Analyze Particles...", "size=3-1000000 pixel circularity=0.0-0.3 display");

//Analyze Fragments
selectWindow(mpshape[1]);
run("Analyze Particles...", "size=3-1000000 pixel circularity=0.3-0.6 display");

//Analyze Particles
selectWindow(mpshape[2]);
run("Watershed");
run("Analyze Particles...", "size=3-1000000 pixel circularity=0.6-1.0 display");

//Get results and save to excel
for (i = 0; i < mpshape.length; i++) {
    close(mpshape[i]);
}
run("Original Scale");

dir = File.directory; 
name = File.nameWithoutExtension; 
saveAs("results",  dir + name + "_results.xls"); 

}
