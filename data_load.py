import shutil
import os
from config import settings
from picsellia import Client


class DataRetriver:
    def __init__(self, api_token, dataset_version_id, annotations_folder, assets_folder):
        self.dataset_ver = Client(api_token=api_token).get_dataset_version_by_id(id=dataset_version_id)

        annotation_zip_path = self.dataset_ver.export_annotation_file("YOLO", "./")
        shutil.unpack_archive(annotation_zip_path, annotations_folder)
        
        self.dataset_ver.download(assets_folder)

    

if __name__ == "__main__":
    data_retriver = DataRetriver(
                                 api_token=settings.api_token,
                                 dataset_version_id=settings.dataset_version_id,
                                 annotations_folder=settings.annotation_folder,
                                 assets_folder=settings.asset_folder
                                 )
