import chromadb


chroma_client = chromadb.Client()



def query_chunks(
        query,
        chunks,
        n_results=4,
        metadatas=None,
        collection_name='my_collection',
        chroma_client=chroma_client, # comment this outside
):
    collection = setup_collection(
            chunks=chunks,
            metadatas=metadatas,
            collection_name=collection_name,
            chroma_client=chroma_client,
    )
    answer_obj = query_collection(
            query=query,
            collection=collection,
            n_results=n_results,
    )
    
    return answer_obj


example_answer_obj = {'ids': [['chunk_31', 'chunk_2', 'chunk_0', 'chunk_3']],
 'embeddings': None,
 'documents': [['17 In summary, Burtner et al. raise an important question about the relevance of acetic acid effects in the yeast chronological aging model to aging in higher eukaryotes. Results presented here and in related studies that were not cited by Burtner et al. indicate that in yeast, accumulation of acetic acid in stationary phase cultures stimulates highly conserved growth signaling pathways and increases oxidative stress and replication stress, all of which have been implicated in aging and/or age-related diseases in more complex organisms.',
   '8 Issue 14 In a recent issue of Cell Cycle, Burtner et al. presented evidence that the accumulation of acetic acid in stationary phase budding yeast cultures is “the primary mechanism of chronological aging in yeast”. 1 Burtner et al. suggest that “how acetic acid-induced cell death could contribute to aging in higher organisms is not readily apparent.” We also recently investigated the effects of pH on chronological lifespan (CLS) of budding yeast.',
   'Acetic acid eﬀects on aging in budding yeast: Are they relevant to aging in higher eukaryotes? William C. Burhans & Martin Weinberger To cite this article: William C. Burhans & Martin Weinberger (2009) Acetic acid eﬀects on aging in budding yeast: Are they relevant to aging in higher eukaryotes?, Cell Cycle, 8:14, 2300-2302, DOI: 10.4161/cc.8.14.8852 To link to this article: https://doi.org/10.4161/cc.8.14.8852 Published online: 15 Jul 2009.',
   'We also recently investigated the effects of pH on chronological lifespan (CLS) of budding yeast. The results of our experiments and those of previous studies point to a mechanism of acetic acid toxicity in yeast related to the induction of growth signaling pathways and oxidative stress. These mechanisms are relevant to aging in all eukaryotes. CLS is defined as the length of time budding yeast cells survive after undergoing a nutrient depletion-induced arrest of the cell cycle in stationary phase. To simplify discussion, we refer to this arrest as “growth']],
 'uris': None,
 'included': ['metadatas', 'documents', 'distances'],
 'data': None,
 'metadatas': [[None, None, None, None]],
 'distances': [[0.15356242656707764,
   0.24700310826301575,
   0.33138254284858704,
   0.37009066343307495]]}


def setup_collection(
        chunks=None,
        metadatas=None,
        collection_name='my_collection',
        chroma_client=chroma_client,
):
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)
    
    collection = chroma_client.create_collection(name=collection_name)
    
    if chunks is not None:
        collection.add(
            ids = [f'chunk_{k}' for k in range(len(chunks))],
            documents = chunks,
            metadatas=metadatas,
        )
        
    return collection


def query_collection(
        query,
        collection,
        n_results=4,
):
    query_answer = collection.query(
        query_texts=query, 
        n_results=n_results,
        )
    
    return query_answer


def get_distances(query_answer): return query_answer['distances']


def get_chunks(query_answer): return query_answer['documents']

