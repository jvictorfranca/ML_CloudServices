import boto3

print(boto3.__version__)
boto_session = boto3.Session()
credentials = boto_session.get_credentials()

bedrock_models = boto3.client('bedrock')
bedrock_models.list_foundation_models()
