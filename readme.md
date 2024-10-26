This is the repository of **Foundation Models for Remote Sensing and Earth Observation: A Survey**, a comprehensive survey of recent progress in multimodal foundation models for remote sensing and earth observation. For details, please refer to:

 **Foundation Models for Remote Sensing and Earth Observation: A Survey**  
 [[Paper](https://arxiv.org/abs/2410.16602)] 
 
 [![arXiv](https://img.shields.io/badge/arXiv-2410.16602-b31b1b.svg)](https://arxiv.org/abs/2410.16602)
 [![Survey](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
<!-- [![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

## Abstract
Remote Sensing (RS) is a crucial technology for observing, monitoring, and interpreting our planet, with broad applications across geoscience, economics, humanitarian fields, etc. While artificial intelligence (AI), particularly deep learning, has achieved significant advances in RS, unique challenges persist in developing more intelligent RS systems, including the complexity of Earth's environments, diverse sensor modalities, distinctive feature patterns, varying spatial and spectral resolutions, and temporal dynamics. Meanwhile, recent breakthroughs in large Foundation Models (FMs) have expanded AI’s potential across many domains due to their exceptional generalizability and zero-shot transfer capabilities. However, their success has largely been confined to natural data like images and video, with degraded performance and even failures for RS data of various non-optical modalities. This has inspired growing interest in developing Remote Sensing Foundation Models (RSFMs) to address the complex demands of Earth Observation (EO) tasks, spanning the surface, atmosphere, and oceans. This survey systematically reviews the emerging field of RSFMs. It begins with an outline of their motivation and background, followed by an introduction of their foundational concepts. It then categorizes and reviews existing RSFM studies including their datasets and technical contributions across Visual Foundation Models (VFMs), Visual-Language Models (VLMs), Large Language Models (LLMs), and beyond. In addition, we benchmark these models against publicly available datasets, discuss existing challenges, and propose future research directions in this rapidly evolving field.

# Citation
If you find our work useful in your research, please consider citing:
```
@article{xiao2024foundation,
  title={Foundation Models for Remote Sensing and Earth Observation: A Survey},
  author={Xiao, Aoran and Xuan, Weihao and Wang, Junjue and Huang, Jiaxing and Tao, Dacheng and Lu, Shijian and Yokoya, Naoto},
  journal={arXiv preprint arXiv:2410.16602},
  year={2024}
}
```

## Menu
- [Visual Foundation models (VFMs) for RS](#visual-foundation-models-for-rs)
  - [VFM Datasets](#vfm-datasets)
  - [VFM Models](#vfm-models)
    - [Pre-training studies](#pre-training-studies)
    - [SAM-based studies](#sam-based-studies)
- [Vision-Language Models for RS](#vision-language-models-for-rs)
  - [VLM Datasets](#vlm-datasets)
  - [VML Models](#vlm-models)
- [Large Language Models for RS](#large-language-models-for-rs)
- [Generative Foundation Models for RS](#generative-foundation-models-for-rs)
- [Other RSFMs](#other-rsfms)


## Visual Foundation models for RS
### VFM Datasets
| Datasets         | Date |  #Samples   |      Modal      |      Annotations       |             Data Sources              |    GSD     |   paper   |                                      Link                                      |
|:-----------------|:----:|:-----------:|:---------------:|:----------------------:|:-------------------------------------:|:----------:|:---------:|:------------------------------------------------------------------------------:|
| FMoW-RGB         | 2018 |   363.6k    |       RGB       |       62 classes       | QuickBird-2, GeoEye-1, WorldView-2/3  |  varying   | [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Christie_Functional_Map_of_CVPR_2018_paper.html) |                  [download](https://github.com/fMoW/dataset)                   |
| BigEarthNet      | 2019 | 1.2 million |     MSI,SAR     |    19 LULC classes     |             Sentinel-1/2              | 10,20,60m  | [paper](https://ieeexplore.ieee.org/abstract/document/8900532) |                        [download](https://bigearth.net)                        |
| SeCo             | 2021 |  1 million  |       MSI       |          None          |           Sentinel-2; NAIP            | 10,20,60m  | [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html) | [download](https://github.com/ServiceNow/seasonal-contrast?tab=readme-ov-file) |
| FMoW-Sentinel    | 2022 |   882,779   |       MSI       |          None          |              Sentinel-2               |    10m     | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html) |             [download](https://sustainlab-group.github.io/SatMAE/)             |
| MillionAID       | 2022 | 1 million   |      RGB        |    51 LULC classes     | SPOT, IKONOS,WorldView, Landsat, etc. | 0.5m-153m  | [paper](https://ieeexplore.ieee.org/abstract/document/9782149) |                [download](https://captain-whu.github.io/DiRS/)                 |
| GeoPile          | 2023 |    600K     |       RGB       |         None           |        Sentinel-2, NAIP, etc.         |  0.1m-30m  | [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html) |                  [download](https://github.com/mmendiet/GFM)                   |
| SSL4EO-S12       | 2023 |  3 million  |    MSI, SAR     |          None          |            Sentinel-1/2               |   10m      | [paper](https://arxiv.org/pdf/2211.07044) |               [download](https://github.com/zhu-xlab/SSL4EO-S12)               |
| SatlasPretrain   | 2023 | 856K tiles  |   RGB,MSI,SAR   | 137 classes of 7 types | Sentinel-1/2, NAIP, NOAA Lidar Scans  | 0.5–2m,10m | [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html) |   [download](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)    |
| MMEarth          | 2024 | 1.2 million | RGB,MSI,SAR,DSM |          None          |     Sentinel-1/2, Aster DEM, etc.     | 10,20,60m  | [paper](https://arxiv.org/pdf/2405.02771) |                   [download](https://github.com/vishalned/MMEarth-data)        | 

### VFM Models
#### Pre-training studies
1. An empirical study of remote sensing pretraining. TGRS2022. | [paper](https://ieeexplore.ieee.org/abstract/document/9782149) | [code](https://github.com/ViTAE-Transformer/RSP) |
2. Satlaspretrain: A large-scale dataset for remote sensing image understanding. ICCV2023.  | [paper](http://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html) | [code](https://github.com/allenai/satlas) |
3. Seasonal contrast: Unsupervised pre-training from uncurated remote sensing data. ICCV2021.  | [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.pdf) | [code](https://github.com/ServiceNow/seasonal-contrast) |
4. Geography-aware self-supervised learning. ICCV2021. | [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ayush_Geography-Aware_Self-Supervised_Learning_ICCV_2021_paper.pdf) | [code](https://github.com/sustainlab-group/geography-aware-ssl) |
5. Self-supervised material and texture representation learning for remote sensing tasks. CVPR2022. | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Akiva_Self-Supervised_Material_and_Texture_Representation_Learning_for_Remote_Sensing_Tasks_CVPR_2022_paper.pdf) | [code](https://github.com/periakiva/MATTER) |
6. Change-aware sampling and contrastive learning for satellite images. CVPR2023. | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf) | [code](https://github.com/utkarshmall13/CACo) |
7. Csp: Self-supervised contrastive spatial pre-training for geospatial-visual representations. ICML2023. | [paper](https://proceedings.mlr.press/v202/mai23a/mai23a.pdf) | [code](https://github.com/gengchenmai/csp) |
8. Skysense: A multi-modal remote sensing foundation model towards universal interpretation for earth observation imagery. CVPR2024. | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf) | [code]() |
9. Contrastive ground-level image and remote sensing pre-training improves representation learning for natural world imagery. ECCV2024. | [paper](https://arxiv.org/abs/2409.19439) | [code]() |
10. Satmae: Pre-training transformers for temporal and multi-spectral satellite imagery. NeurIPS2022. | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/01c561df365429f33fcd7a7faa44c985-Paper-Conference.pdf) | [code](https://github.com/sustainlab-group/SatMAE) |
11. Towards geospatial foundation models via continual pretraining. ICCV2023. | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf) | [code](https://github.com/mmendiet/GFM) |
12. Scale-mae: A scale-aware masked autoencoder for multiscale geospatial representation learning. ICCV2023. | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf) | [code](https://github.com/bair-climate-initiative/scale-mae) |
13. Bridging remote sensors with multisensor geospatial foundation models. CVPR2024. | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_Bridging_Remote_Sensors_with_Multisensor_Geospatial_Foundation_Models_CVPR_2024_paper.pdf) | [code](https://github.com/boranhan/Geospatial_Foundation_Models) |
14. Rethinking transformers pre-training for multi-spectral satellite imagery. CVPR2024. | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.pdf) | [code](https://github.com/techmn/satmae_pp) |
15. Masked angle-aware autoencoder for remote sensing images. ECCV2024. | [paper](https://arxiv.org/pdf/2408.01946) | [code](https://github.com/benesakitam/MA3E) |
16. Mmearth: Exploring multi-modal pretext tasks for geospatial representation learning. ECCV2024. | [paper](https://arxiv.org/pdf/2405.02771) | [code](https://vishalned.github.io/mmearth/) |
17. Croma: Remote sensing representations with contrastive radar-optical masked autoencoders. NeurIPS2023. | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) | [code](https://github.com/antofuller/CROMA) |
18. Cross-scale mae: A tale of multiscale exploitation in remote sensing. NeurIPS2023. | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/3fadcbd0437f4717723ff3f6f7216800-Paper-Conference.pdf) | [code](https://github.com/aicip/Cross-Scale-MAE) |

#### SAM-based studies
1. SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model. NeurIPS2023 (DB). | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1be3843e534ee06d3a70c7f62b983b31-Paper-Datasets_and_Benchmarks.pdf) | [code](https://github.com/ViTAE-Transformer/SAMRS) |
2. Sam-assisted remote sensing imagery semantic segmentation with object and boundary constraints. TGRS2024. | [paper](https://arxiv.org/pdf/2312.02464) | [code](https://github.com/sstary/SSRS) |
3. Uv-sam: Adapting segment anything model for urban village identification. AAAI2024. | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/30260/32248) | [code](https://github.com/tsinghua-fib-lab/UV-SAM) |
4. Cs-wscdnet: Class activation mapping and segment anything model-based framework for weakly supervised change detection. TGRS2023. | [paper](https://ieeexplore.ieee.org/abstract/document/10310006) | [code](https://github.com/WangLukang/CS-WSCDNet) |
5. Adapting segment anything model for change detection in vhr remote sensing images. TGRS2024. | [paper](https://arxiv.org/pdf/2309.01429) | [code](https://github.com/ggsDing/SAM-CD) |
6. Segment any change. NeurIPS2024. | [paper](https://arxiv.org/pdf/2402.01188) |
7. Rsprompter: Learning to prompt for remote sensing instance segmentation based on visual foundation model. TGRS2024. | [paper](https://ieeexplore.ieee.org/iel7/36/10354519/10409216.pdf) | [code](https://github.com/KyanChen/RSPrompter) |
8. Ringmo-sam: A foundation model for segment anything in multimodal remote-sensing images. TGRS2023. | [paper](https://ieeexplore.ieee.org/document/10315957)
9. The segment anything model (sam) for remote sensing applications: From zero to one shot. JSTAR2023. | [paper](https://www.sciencedirect.com/science/article/pii/S1569843223003643) | [code](https://github.com/opengeos/segment-geospatial) |
10. Cat-sam: Conditional tuning for few-shot adaptation of segmentation anything model. ECCV2024 (oral). | [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05662.pdf) | [code](https://xiaoaoran.github.io/projects/CAT-SAM) |
11. Segment anything with multiple modalities. arXiv2024. | [paper](https://arxiv.org/pdf/2408.09085) | [code](https://xiaoaoran.github.io/projects/MM-SAM) |


## Vision-Language Models for RS
### VLM Datasets
|           Task           |    Dataset    |  Image Size   |  GSD (m)  |     #Text     |    #Images    |                                                 Content                                               |                                         Link                                         |
|:------------------------:|:-------------:|:-------------:|:---------:|:-------------:|:-------------:|:-----------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
|           VQA            |   RSVQA-LR    |      256      |    10     |      77K      |      772      |         Questions for existing judging, area estimation, object comparison, scene recognition         |                    [download](https://zenodo.org/records/6344334)                    |        
|           VQA            |   RSVQA-HR    |      512      |   0.15    |     955K      |    10,659     |         Questions for existing judging, area estimation, object comparison, scene recognition         |                    [download](https://zenodo.org/records/6344367)                    |        
|           VQA            |   RSVQAxBen   |      120      |  10--60   |      15M      |    590,326    |                 Questions for existing judging, object comparison, scene recognition                  |                    [download](https://zenodo.org/records/5084904)                    |       
|           VQA            |    RSIVQA     |  512--4,000   |  0.3--8   |     111K      |    37,000     |         Questions for existing judging, area estimation, object comparison, scene recognition         |                 [download](https://github.com/spectralpublic/RSIVQA)                 |    
|           VQA            |     HRVQA     |     1,024     |   0.08    |    1,070K     |    53,512     |                 Questions for existing judging, object comparison, scene recognition                  |                             [download](https://hrvqa.nl)                             |                        
|           VQA            |     CDVQA     |      512      |  0.5--3   |     122K      |     2,968     |                                     Questions for object changes                                      |                   [download](https://github.com/YZHJessica/CDVQA)                    |              
|           VQA            |   FloodNet    | 3,000--4,000  |     -     |      11K      |     2,343     |               Questions for for building and road damage assessment in disaster scenes                |      [download](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)       |
|           VQA            | RescueNet-VQA | 3,000--4,000  |   0.15    |     103K      |     4,375     |                 Questions for building and road damage assessment in disaster scenes                  |               [download](https://www.codabench.org/competitions/1550)                |          
|           VQA            |   EarthVQA    |     1,024     |    0.3    |     208K      |     6,000     | Questions for relational judging, relational counting, situation analysis, and comprehensive analysis |                 [download](https://github.com/Junjue-Wang/EarthVQA)                  |              
| Image-Text Pre-tranining |  RemoteCLIP   |    varied     |  varied   | not specified | not specified |                     Developed based on retrieval, detection and segmentation data                     |               [download](https://github.com/ChenDelong1999/RemoteCLIP)               |
| Image-Text Pre-tranining |     RS5M      | not specified |  varied   |      5M       |      5M       |                           Filtered public datasets, captioned existing data                           |                    [download](https://github.com/om-ai-lab/RS5M)                     |
| Image-Text Pre-tranining |   SKyScript   | not specified | 0.1 - 30  |     2.6M      |     2.6M      |                        Earth Engine images linked with OpenStreetMap semantics                        |                [download](https://github.com/wangzhecheng/SkyScript)                 |
|         Caption          |     RSICD     |      224      |     -     |    24,333     |    10,921     |                                  Urban scenes for object description                                  |             [download](https://github.com/201528014227051/RSICD_optimal)             |
|         Caption          |  UCM-Caption  |      256      |    0.3    |     2,100     |    10,500     |                                  Urban scenes for object description                                  |             [download](https://pan.baidu.com/s/1mjPToHq#list/path=\%2F)              |
|         Caption          |    Sydney     |      500      |    0.5    |      613      |     3,065     |                                  Urban scenes for object description                                  |             [download](https://pan.baidu.com/s/1hujEmcG#list/path=\%2F)              |
|         Caption          | NWPU-Caption  |      256      |  0.2-30   |    157,500    |    31,500     |                                  Urban scenes for object description                                  |              [download](https://github.com/HaiyanHuang98/NWPU-Captions)              | 
|         Caption          |    RSITMD     |      224      |     -     |     4,743     |     4,743     |                                  Urban scenes for object description                                  |         [download](https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD)         |
|         Caption          |    RSICap     |      512      |  varied   |     3,100     |     2,585     |                                  Urban scenes for object description                                  |                   [download](https://github.com/Lavender105/RSGPT)                   |
|         Caption          | ChatEarthNet  |      256      |    10     |   173,488     |   163,488     |                             Urban and rural scenes for object description                             |                 [download](https://github.com/zhu-xlab/ChatEarthNet)                 |
|     Visual Grounding     |     GeoVG     |    1,024      | 0.24--4.8 |     7,933     |     4,239     |                       Visual grounding based on object properties and relations                       |  [download](https://drive.google.com/file/d/1kgnmVC6FVKdxCwaoG77sOfkaIHS_XiFt/view)  | 
|     Visual Grounding     |   DIOR-RSVG   |      800      |  0.5--30  |    38,320     |    17,402     |                       Visual grounding based on object properties and relations                       | [download](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_) |
|     Mixed Multi-task     |    MMRS-1M    |    varied     |  varied   |      1M       |    975,022    |         Collections of RSICD, UCM-Captions, FloodNet, RSIVQA, UC Merced, DOTA, DIOR-RSVG, etc         |                  [download](https://github.com/wivizhang/EarthGPT)                   |
|     Mixed Multi-task     |  Geochat-Set  |    varied     |  varied   |     318k      |    141,246    |               Developed based on DOTA, DIOR, FAIR1M, FloodNet, RSVQA and NWPU-RESISC45                |                  [download](https://github.com/mbzuai-oryx/GeoChat)                  |
|     Mixed Multi-task     |  LHRS-Align   |      256      |    1.0    |     1.15M     |     1.15M     |                            Constructed from Google Map and OSM properties                             |                   [download](https://github.com/NJU-LHRS/LHRS-Bot)                   |
|     Mixed Multi-task     |   VRSBench    |      512      |  varied   |    205,307    |    29,614     |                              Developed based on DOTA-v2 and DIOR dataset                              |                    [download](https://github.com/lx709/VRSBench)                    |
### VLM Models
1. Remoteclip: A vision language foundation model for remote sensing. TGRS2024. | [paper](https://arxiv.org/pdf/2306.11029) | [code](https://github.com/ChenDelong1999/RemoteCLIP) |
2. Rs5m: A large scale vision-language dataset for remote sensing vision-language foundation model. TGRS2024. | [paper](https://arxiv.org/abs/2306.11300) | [code](https://github.com/om-ai-lab/RS5M) |
3. Skyscript: A large and semantically diverse vision-language dataset for remote sensing. AAAI2024. | [paper](https://ojs.aaai.org/index.php/AAAI/article/download/28393/28768) | [code](https://github.com/wangzhecheng/SkyScript) |
4. Remote sensing vision-language foundation models without annotations via ground remote alignment. ICLR2024. | [paper](https://arxiv.org/pdf/2312.06960) |
5. Csp: Self-supervised contrastive spatial pre-training for geospatial-visual representations. ICML2023. | [paper](https://proceedings.mlr.press/v202/mai23a/mai23a.pdf) | [code](https://gengchenmai.github.io/csp-website/) |
6. Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization. NeurIPS2024. | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b57aaddf85ab01a2445a79c9edc1f4b-Paper-Conference.pdf) | [code](https://github.com/VicenteVivan/geo-clip) |
7. Satclip: Global, general-purpose location embeddings with satellite imagery. arXiv2023. | [paper](https://arxiv.org/pdf/2311.17179) | [code](https://github.com/microsoft/satclip) |
8. Learning representations of satellite images from metadata supervision. ECCV2024. | [paper](https://hal.science/hal-04709749/document) |

## Large Language Models for RS
1. Geollm: Extracting geospatial knowledge from large language models. ICLR2024. | [paper](https://arxiv.org/pdf/2310.06213) | [code](https://github.com/rohinmanvi/GeoLLM) |

## Generative Foundation Models for RS
1. Diffusionsat: A generative foundation model for satellite imagery. ICLR2024. | [paper](https://arxiv.org/pdf/2312.03606) | [code](https://github.com/samar-khanna/DiffusionSat) |
2. MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing Dataset and Benchmark for Text-to-Image Generation. NeurIPS2024.

## Other RSFMs
### weather forecasting
1. Accurate medium-range global weather forecasting with 3d neural networks. Nature, 2023. | [paper](https://www.nature.com/articles/s41586-023-06185-3.pdf) | [code](https://github.com/198808xc/Pangu-Weather) |