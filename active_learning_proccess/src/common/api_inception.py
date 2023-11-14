import json
from pycaprio import Pycaprio
from pycaprio.mappings import InceptionFormat, DocumentState
import os


class InceptionClient:
    def __init__(self, url="", username="", password=""):
        self.client = Pycaprio(url, authentication=(username, password))

    def inset_document(self, project_id, doc_path, doc_format=InceptionFormat.TEXT_SENTENCE_PER_LINE):
        with open(doc_path, 'rb') as document_file:
            doc_name = doc_path.split('/')[-1]
            new_document = self.client.api.create_document(project_id, doc_name, document_file,
                                                      document_format=doc_format,
                                                      document_state=DocumentState.NEW)
        print(new_document)

    def download_annotations(self, project_id, document_state, folder_output):
        documents = self.client.api.documents(project_id)
        for document in documents:
            if document.document_state == document_state:
                curated_content = self.client.api.curation(1, document, curation_format='jsoncas')
                dd = json.loads(curated_content)
                json.dump(dd, open(os.getcwd()+ folder_output + document.document_name + '.json', 'w'), indent=4)

    def get_project(self, project_id):
        return self.client.api.project(project_id)

    def get_projects(self):
        return self.client.api.projects()

# file_path = os.getcwd() + '/models/' + 'VIL_1/unlabelled_to_annotator_{}.txt'.format(0)
client_inception = InceptionClient()
print(client_inception.get_projects())
# client_inception.download_annotations(1, DocumentState.CURATION_COMPLETE, folder_output="/data/inception_annotations/")