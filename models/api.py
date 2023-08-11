from models.models import (
    Document,
    DocumentMetadataFilter,
    Query,
    QueryResult,
)
from pydantic import BaseModel
from typing import List, Optional


class DefaultModel(BaseModel):
    index_name: str


class UpsertRequest(DefaultModel):
    documents: List[Document]


class UpsertResponse(BaseModel):
    ids: List[str]


class QueryRequest(DefaultModel):
    embeddings: Optional[bool] = False
    queries: List[Query]


class QueryResponse(BaseModel):
    results: List[QueryResult]


class DeleteRequest(DefaultModel):
    ids: Optional[List[str]] = None
    filter: Optional[DocumentMetadataFilter] = None
    delete_all: Optional[bool] = False


class DeleteResponse(BaseModel):
    success: bool
