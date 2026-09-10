import builtins
import sys
from UploadToBucket import UploadToBucket

# Lines 5 to 16 handles a bug in Sagemaker where Sagemaker automatically adds \n\r, leading to
# failure during cloud deployment.

from DataLoaderFactory import *
from GausiannBlurGenerator import *
from ModelArchitecture import *
from TrainModel import *

import sagemaker

import os
import configparser

from sagemaker.core.helper.session_helper import Session
from sagemaker.core import image_uris
from sagemaker.train.model_trainer import ModelTrainer, Mode
from sagemaker.train.configs import SourceCode, Compute, Networking, InputData

import boto3

def run_experiment(aws_configuration_file, training_script):
    """ Function that deploys and runs a training experiment onto a Sagemaker Instance.

    """
    print("Starting")
    # Code to get around the linespacing bug.
    if sys.platform == "win32":
        _orig_open = builtins.open

        def _lf_open(file, mode="r", *args, **kwargs):
            if "w" in mode and "newline" not in kwargs:
                kwargs["newline"] = "\n"
            return _orig_open(file, mode, *args, **kwargs)

        builtins.open = _lf_open

    print("Reading configuration file")

    aws_config = configparser.ConfigParser()
    aws_config.read(aws_configuration_file)

    AWS_DEFAULT_REGION = str(aws_config['aws_config']['aws_default_region'])
    AWS_ACCESS_KEY_ID = str(aws_config['aws_config']['aws_access_key_id'])
    AWS_SECRET_ACCESS_KEY = str(aws_config['aws_config']['aws_secret_access_key'])
    AWS_S3_BUCKET_ID = str(aws_config['s3_bucket_information']['s3_bucket_id'])
    AWS_ROLE_ID = str(aws_config['aws_config']['aws_role'])
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION,
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY, )

    print("Session Region", boto_session.region_name)

    save_to_bucket = AWS_S3_BUCKET_ID

    sagemaker_Session = Session(boto_session=boto_session, default_bucket=save_to_bucket)

    print("Sagemaker Region", sagemaker_Session.boto_session.region_name)

    print("Setting Up Training Image")

    training_image = image_uris.retrieve(
        framework="pytorch",
        region=AWS_DEFAULT_REGION,
        version="2.0.0",
        py_version="py310",
        instance_type="ml.g4dn.xlarge",
        image_scope="training"
    )

    print("Setting up Source Code")
    source_code = SourceCode(
        source_dir=SCRIPT_DIR,
        entry_script=training_script
    )

    print("Setting up Compute")

    compute = Compute(
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
    )

    print("Setting up networking")
    networking = Networking(
        security_group_ids=["sg-04979086a99a32f51"],
        subnets=["subnet-0bffbf7ebb538bea0"]
    )

    print("Setting up model trainer")

    model_trainer = ModelTrainer(
        training_image=training_image,
        role=AWS_ROLE_ID,
        sagemaker_session=sagemaker_Session,
        source_code=source_code,
        compute=compute,
        networking=networking
    )
    print("Checks Complete")
    print("Deploying experiment to Sagemaker Model Trainer")

    model_trainer.train()
