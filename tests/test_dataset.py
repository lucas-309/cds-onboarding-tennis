import os
import unittest

DATASET_PATH = 'atp_matches_2024.csv'


class TestDatasetDownload(unittest.TestCase):

    def test_dataset_exists(self):
        """
        Test if the dataset has been downloaded.
        Checks if the dataset file or directory exists.
        """
        self.assertTrue(
            os.path.exists(DATASET_PATH),
            f"Dataset not found at {DATASET_PATH}. Please download the dataset and place it in the correct location."
        )

    def test_dataset_not_empty(self):
        """
        Test if the dataset is not empty.
        Ensures the file or folder is not empty.
        """
        if os.path.isdir(DATASET_PATH):
            # Check if directory is not empty
            self.assertTrue(
                len(os.listdir(DATASET_PATH)) > 0,
                "Dataset folder is empty. Please check the download."
            )
        elif os.path.isfile(DATASET_PATH):
            # Check if file is not empty
            self.assertTrue(
                os.path.getsize(DATASET_PATH) > 0,
                "Dataset file is empty. Please check the download."
            )
        else:
            self.fail(f"{DATASET_PATH} is neither a file nor a directory.")


if __name__ == '__main__':
    unittest.main()