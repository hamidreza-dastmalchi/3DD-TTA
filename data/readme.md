## Dataset Integration and Preparation

### Downloading the Datasets

To integrate these datasets with our code, download them using the links below:

- [ModelNet40](https://storage.googleapis.com/kaggle-data-sets/943894/1599485/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241206%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241206T172111Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3d018fdcad9da71cbea38bf88a546bc5bcf45dfed80eb6c87533ae6ed729e414224e3ecf232032f71be102f25a6051fd521a7a9d88c10b1d4fcfc4f8a7378fbc05db6e23ca0297b53d27876ae05505c8705a587351137c8f01de159f7f62c17af209775ba1ef874068ca3bf4f7cb7c0b4e385d49261cb07830d7aa678826a9ab66fb7e264e847cc89e63ed18b1e2f5cdc2ebda2d6e6440afcabd906008fa8158fdac361366f1a754664575a842833a62affeb8c0cb41a66b6a0b62f403912f4010658baa2d1baf7f45abf59e093ed85610910e65bfcad4ef70277b984e727b8d580782d132466474e1a90343ac322e085e6f4f4f458db9eec8ebcdcc9ca9947a)  
- [ShapeNetCore](https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/)  
- [ScanObjectNN](https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip)  
  *(Ensure you agree to the terms of use by filling out [this form](https://forms.gle/g29a6qSgjatjb1vZ6) before downloading.)*

After downloading, extract the contents of each dataset into the same parent directory for easy access.

---

### Adding Corruptions to the Dataset

For the **ModelNet40-C dataset**, you can directly download the corrupted version from [this Zenodo link](https://zenodo.org/records/6017834#.YgNeKu7MK3J). 
To generate the same corruptions for ShapeNetCore or ScanObjectNN (if required), run the following command:

```
python ./datasets/create_corrupted_dataset.py --main_path <path/to/dataset/parent/directory> --dataset <dataset_name>
```

Replace `<dataset_name>` with either `scanobjectnn` or `shapenet`, as appropriate.

**Note:** The corruptions "occlusion" and "lidar" require model meshes and are computed using the [Open3D](https://www.open3d.org/docs/release/getting_started.html) library.

Once the corrupted versions of the datasets and their labels are prepared, organize them into the following folder structure under the data root directory:
```
data/
├── modelnet40_c/
│   ├── data_background_5.npy
│   ├── data_cutout_5.npy
│   ├── ...
│   ├── label.npy
├── shapenet_c/
│   ├── data_background_5.npy
│   ├── data_cutout_5.npy
│   ├── ...
│   ├── label.npy
├── scanobjectnn_c/
│   ├── data_background_5.npy
│   ├── data_cutout_5.npy
│   ├── ...
│   ├── label.npy
```

Ensure each dataset's corrupted version is placed in its corresponding folder (`modelnet40_c`, `shapenet_c`, `scanobjectnn_c`) within the `data` root directory.
