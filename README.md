## `RESEARCH-CONTEST-3-2024`

```
This repository keeps track of my preparation for the upcoming Research Contest held by my university
```

`Start date: January 22 2024`

`PROJECT TRACKER:`

- Jan 22: 
  - Get stressed learning about LaTeX
  - Set up environment in vscode `Failed`
  - Currently using **Texmaker** and [Overleaf](https://www.overleaf.com/project) instead
  - Learning LaTeX commands
    - [LaTeX – Full Tutorial for Beginners](https://www.youtube.com/watch?v=ydOTMQC7np0&ab_channel=freeCodeCamp.org) (extremely low-paced)
    - [Learn LaTeX in 30 minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes) (self-paced)
  - Daily LaTeX practice is preferable to learning all commands at once because there are just too many things
- Jan 23:
  - Grasp the idea of skeletonization as it is a new concept to me
    - [Skeletonization 101](<paper/papers to research/skeletonization_101.md>)
    - [Skeletonization/Medial Axis Transform](https://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm#:~:text=Brief%20Description,of%20the%20original%20foreground%20pixels.)
  - Read [Hi-LASSIE Paper](<paper/papers to research/Yao_Hi-LASSIE_High-Fidelity_Articulated_Shape_and_Skeleton_Discovery_From_Sparse_Image_CVPR_2023_paper.pdf>)
  - [Hi-LASSIE Project Page](https://chhankyao.github.io/hi-lassie/)
  - Learn LaTeX commands
  - Documents for later use:
    - [Distance Transform](https://homepages.inf.ed.ac.uk/rbf/HIPR2/distance.htm)
    - [Thinning](https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm)
    - [Binary Images](https://homepages.inf.ed.ac.uk/rbf/HIPR2/binimage.htm)
    - [Thresholding](https://homepages.inf.ed.ac.uk/rbf/HIPR2/threshld.htm)
- Jan 24:
  - Continue reading [Hi-LASSIE Paper](<paper/papers to research/Yao_Hi-LASSIE_High-Fidelity_Articulated_Shape_and_Skeleton_Discovery_From_Sparse_Image_CVPR_2023_paper.pdf>) `Unfinished`
  - Analyze and Run/Test Hi-LASSIE Code `Failed`
  - Successfully using **Zhang-Suen thinning** to extract a `skeleton image` from [Crack detection's Panicle image](test_binary_img.png)
- Jan 25:
  - Read [Zhang-Suen Thinning paper](<paper/papers to research/A Fast Parallel Algorithm for Thinning Digital Patterns.pdf>) `Done`
  - Testing code for **Zhang-Suen** and making [Skeleton dataset](skeleton_dataset) `Done`
  - Manually add junctions and their classes to [dataset (only on local machine)](<paper/target paper/T2-PLT9-1C8-1 (Dataset)>) `Unfinished`
  - Learn LaTeX commands
- Jan 26:
  - Reread [2013 Jouannic Paper](<paper/target paper/1471-2229-13-122 (2013_paper_Stefan Jouannic).pdf>) to answer the question: `How to distinguish Generating point and Primary point` `Unfinished`
  - Successfully label (add junctions) to dataset
  - Junction-related scripts:
    - [RGB_2_GRAY](rgb_2_gray.py): Turning RGB image to Gray
    - [Junction coordinate](vertices_coordinates.py): Extracting coordinate from [paper's xml](Jouannic_xml) file
    - [label_dataset](label_dataset.py): Adding junctions to grayscale
  - Creating new dataset folders:
    - [grayscale_dataset](grayscale_dataset)
    - [labelled_dataset](labelled_dataset)
- Jan 27:
  - In general, first week's work is done
  - Reorganize code and files
- Jan 30:
  - Read [Denmune](<paper/papers to research/2021_DenMune Density peak based clustering using mutual nearest neighbors.pdf>)
  - Learn LaTeX commands

`Expected end date: March 18 2024`