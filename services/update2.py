from services.load2 import load_database2
from services.deleteVectors import removeDocumentsUsingQuery



async def update_database2(uid, projectId, questionId, participantId, filename, videoType, metadata):
    
    # Prepare the query to find existing documents for deletion
    query_dict = {
        "uid": uid,
        "projectId": projectId,
        "questionId": questionId,
        "participantId": participantId,
        "videoType": videoType
    }

    if questionId is None:
        query_dict.pop("questionId", None)
    if participantId is None:
        query_dict.pop("participantId", None)

    if metadata:
        query_dict.update(metadata)
    
    # Remove existing documents that match the query
    await removeDocumentsUsingQuery(query_dict)
    
    # Add the new documents to the database
    response = await load_database(uid, projectId, questionId, participantId, filename, videoType, metadata)
    
    return response





