import functions_framework
from google.cloud import storage
from google.cloud import vision
import pandas as pd
import json
import re
import os
from datetime import datetime

from llama_index import Document, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def parse_bucket(cloud_event):
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket = data["bucket"]
    name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket}")
    print(f"File: {name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")

    input_name = name.split("/")
    user_dir_name = input_name[0]
    resource_dir_name = input_name[1]
    file_name = input_name[2]
    print("User dir name: " + user_dir_name)
    print("Resource dir name: " + resource_dir_name)
 
    if resource_dir_name != "input": 
        return

    gs_input_file_URI = f"gs://{bucket}/{name}"
    print("Input file URI: " + gs_input_file_URI)
    gs_ocr_results_URI = f"gs://{bucket}/{user_dir_name}/ocr_results/"
    print("OCR dest URI: " + gs_ocr_results_URI)
    gs_index_results_URI = f"gs://{bucket}/{user_dir_name}/index_results/"
    print("Index results URI: " + gs_index_results_URI)
    gs_index_source_URI = f"gs://{bucket}/{user_dir_name}/index_source/"
    print("Index source URI: " + gs_index_source_URI)

    storage_client = storage.Client()
    bucket_obj = storage_client.bucket(bucket) # email-attachment-test is defined in deploy file
    blob = bucket_obj.get_blob(name)
    bt = blob.download_as_string()

    # determine file type
    file_ext = str(name)[-3:]
    print(f"Extension: {file_ext}")
    
    match file_ext:
        case "txt":
            # print(bt)
            new_name = f"{user_dir_name}/index_source/{file_name}"
            copy_blob(bucket, name, bucket, new_name,)
            delete_blob(bucket, name)            
        case "csv":
            s = str(bt, "utf-8")
            s = StringIO(s)
            df = pd.read_csv(s)
            print(df)
        case "pdf":
            # async_detect_document(gs_input_file_URI, gs_ocr_results_URI)
            async_detect_document(gs_input_file_URI, gs_index_source_URI)
            # write_to_text(gs_ocr_results_URI)
            delete_blob(bucket, name)
        case _:
            print(f"File type not handled")

    # build index
    construct_index(gs_index_source_URI, gs_index_results_URI)


def async_detect_document(gcs_source_uri, gcs_destination_uri):
    """OCR with PDF/TIFF as source files on GCS"""

    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'

    # How many pages should be grouped into each json output file.
    batch_size = 2 

    client = vision.ImageAnnotatorClient()

    feature = vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(
        gcs_source=gcs_source, mime_type=mime_type)

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size)

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config,
        output_config=output_config)

    operation = client.async_batch_annotate_files(
        requests=[async_request])

    print('Waiting for the operation to finish.')
    operation.result(timeout=420)

def write_to_text(gcs_destination_uri):
    # Once the request has completed and the output has been
    # written to GCS, we can list all the output files.
    storage_client = storage.Client()

    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)

    # List objects with the given prefix, filtering out folders.
    blob_list = [blob for blob in list(bucket.list_blobs(
        prefix=prefix)) if not blob.name.endswith('/')]
    print('Output files:')
    for blob in blob_list:
        print(blob.name)

    # Process the first output file from GCS.
    # Since we specified batch_size=2, the first response contains
    # the first two pages of the input file.
    output = blob_list[0]

    json_string = output.download_as_string()
    response = json.loads(json_string)

    # The actual response for the first page of the input file.
    first_page_response = response['responses'][0]
    annotation = first_page_response['fullTextAnnotation']

    # Here we print the full text from the first page.
    # The response contains more information:
    # annotation/pages/blocks/paragraphs/words/symbols
    # including confidence scores and bounding boxes
    print('Full text:\n')
    print(annotation['text'])


def construct_index(gcs_uri_input, gcs_uri_output):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    model_name = "text-embedding-ada-002" # "text-davinci-003"
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name, max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    storage_client = storage.Client()

    # find input bucket info
    match = re.match(r'gs://([^/]+)/(.+)', gcs_uri_input)
    bucket_name = match.group(1)
    prefix = match.group(2)

    print("bucket_name: " + bucket_name)
    print("prefix: " + prefix)

    bucket = storage_client.get_bucket(bucket_name)


    # read through input files and create a list of text/str objects
    blob_list = [blob for blob in list(bucket.list_blobs(
        prefix=prefix)) if not blob.name.endswith('/')]
    print("Full List of Input: ")
    text_list = []
    for blob in blob_list:
        print(blob.name)
        # documents = documents + blob.download_as_string().decode()
        text_list.append(blob.download_as_string().decode())
        delete_blob(bucket_name, blob.name)

    # combine text list into document
    documents = [Document(t) for t in text_list]

    # find output bucket info
    match = re.match(r'gs://([^/]+)/(.+)', gcs_uri_output)
    output_bucket_name = match.group(1)
    output_prefix = match.group(2)
    output_bucket = storage_client.get_bucket(output_bucket_name)

    # create datetime stamped output file
    # output_blob = output_bucket.blob(f"{output_prefix}index_{datetime.now():%Y-%m-%d_%H:%M:%S}.json")
    # index = GPTSimpleVectorIndex(
      #  documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # new code trying to add to an index
    output_blob = output_bucket.blob(f"{output_prefix}index.json")
    if output_blob.exists():
        index = GPTSimpleVectorIndex.load_from_string(output_blob.download_as_string())
        for doc in documents:
            index.insert(doc)
    else:
        index = GPTSimpleVectorIndex([])
        for doc in documents:
            index.insert(doc)

    index_str = index.save_to_string()
    output_blob.upload_from_string(index_str)


def copy_blob(
    bucket_name, blob_name, destination_bucket_name, destination_blob_name,
):
    """Copies a blob from one bucket to another with a new name."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # destination_bucket_name = "destination-bucket-name"
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)


    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to copy is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    # There is also an `if_source_generation_match` parameter, which is not used in this example.
    destination_generation_match_precondition = 0

    if destination_bucket.blob(destination_blob_name).exists():
        return()

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name, if_generation_match=destination_generation_match_precondition,
    )

    print(
        "Blob {} in bucket {} copied to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    generation_match_precondition = None

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to delete is aborted if the object's
    # generation number does not match your precondition.
    blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
    generation_match_precondition = blob.generation

    blob.delete(if_generation_match=generation_match_precondition)

    print(f"Blob {blob_name} deleted.")
