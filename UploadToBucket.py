import boto3
import configparser
class UploadToBucket:

    def __init__(self, bucket_name):
        config = configparser.ConfigParser()
        config.read('cloud_configuration.ini')
        session = boto3.Session(
            aws_access_key_id=config['aws_config']['aws_access_key_id'],
            aws_secret_access_key=config['aws_config']['aws_secret_access_key'],
        )
        self.client = session.client('s3')
        self.bucket_name = bucket_name

    def upload_file(self, key_name, file_to_upload):

        self.client.upload_file(
            Filename=file_to_upload,
            Bucket=self.bucket_name,
            Key=key_name
        )

        print("Uploaded ", file_to_upload, "as ", key_name)