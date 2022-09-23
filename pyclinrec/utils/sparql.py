import json

import redis
from SPARQLWrapper import JSON


class SparQLOffsetFetcher:

    def __init__(self, sparql_wrapper, page_size, where_body, select_columns, from_statement="", prefixes="", timeout=0,
                 redis=None):
        self.sparql_wrapper = sparql_wrapper
        self.sparql_wrapper.setTimeout(timeout)
        self.page_size = page_size
        self.current_offset = 0
        self.where_body = where_body
        self.from_statement = from_statement
        self.prefixes = prefixes
        self.select_columns = select_columns
        sparql_wrapper.setReturnFormat(JSON)
        self.redis = redis
        self.count = -1
        self.__get_count__()

    def __get_count__(self):
        if self.count == -1:
            if len(self.from_statement) == 0:
                query = f"""define sql:big-data-const 0
                 {self.prefixes}
                 SELECT count(distinct *) as ?count WHERE {{
                    {self.where_body}
                }}
                """
            else:
                query = f"""define sql:big-data-const 0\n {self.prefixes}\n SELECT count(distinct *) as ?count \nFROM <{self.from_statement}> WHERE {{
                        {self.where_body}
                    }}
                """
                print(query)
            result = self._fetch_from_cache_or_query(query)
            count = int(result['results']['bindings'][0]['count']["value"])
            self.count = count
            return count
        return self.count

    def next_page(self):
        if self.current_offset < self.count:
            if len(self.from_statement) == 0:
                query = f""" define sql:big-data-const 0 \n{self.prefixes} \nSELECT {self.select_columns} WHERE {{
                        {self.where_body}
                    }} LIMIT {self.page_size} OFFSET {self.current_offset}
                    """
            else:
                query = f""" define sql:big-data-const 0\n {self.prefixes}\n SELECT {self.select_columns} FROM <{self.from_statement}> WHERE {{
                                        {self.where_body}
                                    }} LIMIT {self.page_size} OFFSET {self.current_offset}
                                    """
            result = self._fetch_from_cache_or_query(query)
            self.current_offset += self.page_size
            return result['results']['bindings']
        return None

    def fetch_all(self):
        result = list()
        page = list()
        while page is not None:
            page = self.next_page()
            if page is not None:
                result.extend(page)
        return result

    def _fetch_from_cache_or_query(self, query):
        result = str()
        found = False
        cache_key = query
        # If redis was successfully initialized
        if self.redis is not None:
            # Get cache value and check whether it exists
            val = self.redis.get(cache_key)
            if val is not None:
                result = val
                found = True
        # If it doesn't exist, query annotator and cache the result
        if not found:
            self.sparql_wrapper.setQuery(query)
            result = self.sparql_wrapper.query().response.read()
            if len(result) == 0:
                result = ""
            if self.redis is not None:
                self.redis.set(cache_key, result)
        strres = str(result, 'utf-8')
        return json.loads(strres)
