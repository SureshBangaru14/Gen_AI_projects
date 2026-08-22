import os
import shutil
from pathlib import Path

import chromadb


class ChromaDBManager:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="resume_collection"
    ):

        self.persist_directory = Path(
            persist_directory
        ).expanduser().resolve()

        self.collection_name = (
            collection_name
        )

        # ----------------------------------------------------
        # INITIALIZE DATABASE
        # ----------------------------------------------------

        self._initialize_database()


    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def _initialize_database(self):

        try:

            # ------------------------------------------------
            # CREATE DIRECTORY
            # ------------------------------------------------

            self.persist_directory.mkdir(

                parents=True,

                exist_ok=True

            )


            # ------------------------------------------------
            # WRITE PERMISSION CHECK
            # ------------------------------------------------

            if not os.access(

                self.persist_directory,

                os.W_OK

            ):

                raise PermissionError(

                    f"ChromaDB directory is not writable: "
                    f"{self.persist_directory}"

                )


            # ------------------------------------------------
            # PERSISTENT CLIENT
            # ------------------------------------------------

            self.client = (
                chromadb.PersistentClient(

                    path=str(
                        self.persist_directory
                    )

                )
            )


            # ------------------------------------------------
            # GET OR CREATE COLLECTION
            # ------------------------------------------------

            self.collection = (
                self.client.get_or_create_collection(

                    name=self.collection_name,

                    metadata={

                        "description":
                            "Resume matching collection"

                    }

                )
            )


        except Exception as error:

            raise RuntimeError(

                "Failed to initialize ChromaDB.\n\n"

                f"Path: {self.persist_directory}\n"

                f"Collection: {self.collection_name}\n\n"

                f"Original error: {error}"

            ) from error


    # ========================================================
    # GET COLLECTION
    # ========================================================

    def get_collection(self):

        return self.collection


    # ========================================================
    # COUNT
    # ========================================================

    def count(self):

        try:

            return self.collection.count()

        except Exception as error:

            raise RuntimeError(

                f"Failed to get ChromaDB count: {error}"

            ) from error


    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents,
        embeddings,
        metadatas,
        ids
    ):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not documents:

            return


        if not embeddings:

            raise ValueError(
                "Embeddings cannot be empty."
            )


        if not metadatas:

            raise ValueError(
                "Metadata cannot be empty."
            )


        if not ids:

            raise ValueError(
                "IDs cannot be empty."
            )


        # ----------------------------------------------------
        # LENGTH CHECK
        # ----------------------------------------------------

        lengths = {

            len(documents),

            len(embeddings),

            len(metadatas),

            len(ids)

        }


        if len(lengths) != 1:

            raise ValueError(

                "Documents, embeddings, metadata "
                "and IDs must have the same length."

            )


        try:

            # ------------------------------------------------
            # ADD TO CHROMADB
            # ------------------------------------------------

            self.collection.add(

                ids=ids,

                documents=documents,

                embeddings=embeddings,

                metadatas=metadatas

            )


        except Exception as error:

            raise RuntimeError(

                f"Failed to add documents to ChromaDB: "
                f"{error}"

            ) from error


    # ========================================================
    # UPSERT DOCUMENTS
    # ========================================================

    def upsert_documents(
        self,
        documents,
        embeddings,
        metadatas,
        ids
    ):

        if not documents:

            return


        lengths = {

            len(documents),

            len(embeddings),

            len(metadatas),

            len(ids)

        }


        if len(lengths) != 1:

            raise ValueError(

                "Documents, embeddings, metadata "
                "and IDs must have the same length."

            )


        try:

            self.collection.upsert(

                ids=ids,

                documents=documents,

                embeddings=embeddings,

                metadatas=metadatas

            )


        except Exception as error:

            raise RuntimeError(

                f"Failed to upsert documents: {error}"

            ) from error


    # ========================================================
    # QUERY BY EMBEDDING
    # ========================================================

    def query(
        self,
        query_embeddings,
        n_results=5,
        where=None,
        include=None
    ):

        if not query_embeddings:

            return {

                "ids": [[]],

                "documents": [[]],

                "metadatas": [[]],

                "distances": [[]]

            }


        if include is None:

            include = [

                "documents",

                "metadatas",

                "distances"

            ]


        try:

            result = self.collection.query(

                query_embeddings=[
                    query_embeddings
                ],

                n_results=n_results,

                where=where,

                include=include

            )


            return result


        except Exception as error:

            raise RuntimeError(

                f"ChromaDB query failed: {error}"

            ) from error


    # ========================================================
    # GET BY IDS
    # ========================================================

    def get(
        self,
        ids=None,
        where=None,
        limit=None,
        offset=None,
        include=None
    ):

        try:

            kwargs = {}


            if ids is not None:

                kwargs["ids"] = ids


            if where is not None:

                kwargs["where"] = where


            if limit is not None:

                kwargs["limit"] = limit


            if offset is not None:

                kwargs["offset"] = offset


            if include is not None:

                kwargs["include"] = include


            return self.collection.get(
                **kwargs
            )


        except Exception as error:

            raise RuntimeError(

                f"Failed to retrieve ChromaDB "
                f"documents: {error}"

            ) from error


    # ========================================================
    # DELETE BY IDS
    # ========================================================

    def delete(
        self,
        ids=None,
        where=None
    ):

        try:

            kwargs = {}


            if ids is not None:

                kwargs["ids"] = ids


            if where is not None:

                kwargs["where"] = where


            if not kwargs:

                raise ValueError(

                    "Provide ids or where condition "
                    "for deletion."

                )


            self.collection.delete(
                **kwargs
            )


        except Exception as error:

            raise RuntimeError(

                f"Failed to delete ChromaDB "
                f"documents: {error}"

            ) from error


    # ========================================================
    # CLEAR COLLECTION
    # ========================================================

    def clear(self):

        try:

            # ------------------------------------------------
            # DELETE COLLECTION
            # ------------------------------------------------

            self.client.delete_collection(

                name=self.collection_name

            )


        except Exception:

            # Collection may not exist.
            pass


        # ----------------------------------------------------
        # RECREATE COLLECTION
        # ----------------------------------------------------

        try:

            self.collection = (
                self.client.get_or_create_collection(

                    name=self.collection_name,

                    metadata={

                        "description":
                            "Resume matching collection"

                    }

                )
            )


        except Exception as error:

            raise RuntimeError(

                f"Failed to recreate ChromaDB collection: "
                f"{error}"

            ) from error


    # ========================================================
    # DELETE COLLECTION
    # ========================================================

    def delete_collection(self):

        try:

            self.client.delete_collection(

                name=self.collection_name

            )


        except Exception as error:

            raise RuntimeError(

                f"Failed to delete collection: "
                f"{error}"

            ) from error


    # ========================================================
    # RECREATE COLLECTION
    # ========================================================

    def recreate_collection(self):

        try:

            self.client.delete_collection(

                name=self.collection_name

            )

        except Exception:

            pass


        self.collection = (
            self.client.get_or_create_collection(

                name=self.collection_name,

                metadata={

                    "description":
                        "Resume matching collection"

                }

            )
        )


    # ========================================================
    # RESET DATABASE
    #
    # WARNING:
    # This deletes the entire ChromaDB directory.
    # ========================================================

    def reset_database(self):

        try:

            # ------------------------------------------------
            # CLOSE REFERENCES
            # ------------------------------------------------

            self.collection = None

            self.client = None


            # ------------------------------------------------
            # DELETE DIRECTORY
            # ------------------------------------------------

            if self.persist_directory.exists():

                shutil.rmtree(

                    self.persist_directory

                )


            # ------------------------------------------------
            # RECREATE
            # ------------------------------------------------

            self.persist_directory.mkdir(

                parents=True,

                exist_ok=True

            )


            # ------------------------------------------------
            # INITIALIZE
            # ------------------------------------------------

            self._initialize_database()


        except Exception as error:

            raise RuntimeError(

                f"Failed to reset ChromaDB: {error}"

            ) from error


    # ========================================================
    # COLLECTION INFORMATION
    # ========================================================

    def collection_info(self):

        try:

            return {

                "name":
                    self.collection.name,

                "count":
                    self.collection.count(),

                "persist_directory":
                    str(
                        self.persist_directory
                    )

            }


        except Exception as error:

            raise RuntimeError(

                f"Failed to retrieve collection "
                f"information: {error}"

            ) from error