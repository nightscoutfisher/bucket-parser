import functions_framework
from google.cloud import storage
from google.cloud import vision
import pandas as pd
import json
import re

from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI


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
    gs_index_destination_URI = f"gs://{bucket}/{user_dir_name}/index/"
    print("Index destination URI: " + gs_index_destination_URI)

    storage_client = storage.Client()
    bucket_obj = storage_client.bucket(bucket) # email-attachment-test is defined in deploy file
    blob = bucket_obj.get_blob(name)
    bt = blob.download_as_string()

    # determine file type
    file_ext = str(name)[-3:]
    print(f"Extension: {file_ext}")
    
    match file_ext:
        case "txt":
            print(bt)
        case "csv":
            s = str(bt, "utf-8")
            s = StringIO(s)
            df = pd.read_csv(s)
            print(df)
        case "pdf":
            # async_detect_document("gs://email-attachment-test/mikefisher/Michael_Fisher_bio.pdf", "gs://email-attachment-test/mikefisher/ocr_results/")
            async_detect_document(gs_input_file_URI, gs_ocr_results_URI)
            # write_to_text("gs://email-attachment-test/mikefisher/ocr_results/")
            write_to_text(gs_ocr_results_URI)
        case _:
            print(f"File type not handled")
 

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

def construct_index(gcs_uri):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 


    storage_client = storage.Client()

    match = re.match(r'gs://([^/]+)/(.+)', gcs_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)

    blob_list = [blob for blob in list(bucket.list_blobs(
        prefix=prefix)) if not blob.name.endswith('/')]
    print('Output files:')
    for blob in blob_list:
        print(blob.name)
        documents = documents + blob.download_as_string()


    # documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # index.save_to_disk('index.json')
    # return index

    blob = bucket.blob("index.json")
    with blob.open("w") as f:
        f.write(index)
