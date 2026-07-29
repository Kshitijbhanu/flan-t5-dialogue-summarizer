import weave

@weave.op()
def log_inference(query : str , response : str, model : str):
    return{
        "query" : query,
        "response" : response,
        "model" : model
    }