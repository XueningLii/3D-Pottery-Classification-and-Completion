# 3D-Pottery-Classification-and-Completion
3D Pottery Classification and Completion Project for CSC8499 Individual Project

## Project Overview

The aim of this project is to explore the feasibility of applying 3D graphic classification and complementation algorithms based on deep learning to the automatic classification and restoration of pottery—a specific type of cultural artifact. The goal is to achieve more optimal results and valuable evaluation metrics, while also accumulating comprehensive engineering implementation experience in using deep learning for cultural artifact restoration. This experience will serve as an important reference for the application of artificial intelligence in the field of cultural heritage preservation.

### Key Contributions of This Project:

1. **Comprehensive Research:** Conducted an extensive survey to understand and compile existing works related to the application of AI in artifact classification and restoration, providing a solid theoretical foundation for the project's implementation.
  
2. **3D Dataset Compilation:** Created a comprehensive 3D dataset of classical ancient pottery, named 3D Pottery 8, which includes simulated broken samples handcrafted for each original item.
  
3. **Classification Model Application:** Applied the 3D point cloud classification model, PointNet, to the 3D Pottery 8 dataset, achieving the best possible performance under current conditions.
  
4. **Complementation Model Application:** Applied 3D point cloud completion models, PCN and PF-Net, to the 3D Pottery 8 dataset, obtaining the best performance achievable under the current setup.
  
5. **Full Project Implementation:** Built a fully functional deep learning project pipeline, from data preprocessing to evaluation metric visualization, enabling seamless model integration and flexible switching between different models.
  
6. **Efficiency Enhancements:** Added new features such as data preprocessing scripts, automatic dataset splitting, and data preloading to the existing model training workflows, thereby improving the efficiency of large-scale data training under limited resource conditions.
  
7. **Evaluation Metrics Analysis:** Conducted multi-dimensional comparisons, correlations, and analyses of evaluation metrics, identifying current limitations and providing suggestions for future improvements.

### Usage Instructions

1. Create a folder named `data` in the project directory, and then download the 3D Pottery 8 dataset into this folder.
2. Run the `train_PCN.ipynb`, `train_PF_Net.ipynb`, or `train_PointNet.ipynb` scripts on platforms like Google Colab to step through the experimental results. Alternatively, you can convert the `.ipynb` files into regular Python scripts and run them.
