#!/usr/bin/env python3

from argparse import ArgumentParser
from string import punctuation
from json import load
from typing import Any, TypedDict, Literal
from nltk.stem import PorterStemmer
from operator import itemgetter
from collections import Counter
import pickle
import os
import math

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_SEARCH_LIMIT = 5
CACHE_DIR = 'cache'
DATA_DIR = 'data'
MAX_LEN = 150

def safe_print(s: str):
  return f"{s[:MAX_LEN - 3]}..." if len(s) >= MAX_LEN else s

class Movie(TypedDict):
  id: int
  title: str
  description: str

class IndexStats(TypedDict):
  total: int
  avg_doc_length: float

class IndexCachedData(TypedDict):
  index: dict[str, list[int]]
  docmap: dict[int, Movie]
  term_frequencies: dict[int, Counter]
  doc_lengths: dict[int, int]

CacheType = Literal[
  'index',
  'docmap',
  'term_frequencies',
  'doc_lengths',
]

IdfType = Literal['normal', 'bm25']
SearchType = Literal['normal', 'tf_idf', 'bm25']

class PreProcessor:
  def __init__(self, stop_words_path: str):
    self.__stemmer = PorterStemmer()
    self.__stop_words = None
    with open(stop_words_path) as f:
      self.__stop_words = f.read().splitlines()
  def __sanitize(self, input: str) -> str:
    return input.replace('\n', ' ').lower().translate(str.maketrans('', '', punctuation))
  def __tokenize(self, input: str) -> list[str]:
    return input.split(" ")
  def __remove_stop_words(self, tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in self.__stop_words]
  def __stem(self, tokens: list[str]) -> list[str]:
    return [self.__stemmer.stem(token) for token in tokens]
  def process(self, input: str) -> list[str]:
    return self.__stem(self.__remove_stop_words(self.__tokenize(self.__sanitize(input))))

class InvertedIndex:
  def __init__(self, db_path: str, pre_processor: PreProcessor):
    self.__pre_processor = pre_processor
    self.__data: IndexCachedData = {
      'index': {},
      'docmap': {},
      'term_frequencies': {},
      'doc_lengths': {},
    }
    self.__stats: IndexStats = {
      'total': 0,
      'avg_doc_length': 1.0,
    }
    self.__db_path = db_path

  def __process_doc(self, id: int, text: str) -> int:
    tokens = self.__pre_processor.process(text)
    freq = Counter[str, int](tokens)
    self.__data['term_frequencies'][id] = freq
    self.__data['doc_lengths'][id] = len(tokens)
    print(f"[{id}] -> {safe_print(str(freq))}")
    for token in tokens:
      if token not in self.__data['index']:
        self.__data['index'][token] = [id]
      elif id not in self.__data['index'][token]:
        self.__data['index'][token].append(id)
        self.__data['index'][token].sort()

  def __eval_stats(self):
    self.__stats['total'] = len(self.__data['docmap'])
    total_length = sum(length for length in self.__data['doc_lengths'].values())
    self.__stats['avg_doc_length'] = total_length / self.__stats['total']

  def __save_cache_file(self, type: CacheType):
    with open(f"{CACHE_DIR}/{type}.pkl", "wb") as f:
      pickle.dump(self.__data[type], f)

  def __load_cache_file(self, type: CacheType):
    if not os.path.exists(f"{CACHE_DIR}/{type}.pkl"):
      raise FileNotFoundError(f"{CACHE_DIR}/{type}.pkl does not exist")
    with open(f"{CACHE_DIR}/{type}.pkl", "rb") as f:
      self.__data[type] = pickle.load(f)

  def get_data(self, type: CacheType):
    return self.__data[type]
  
  def build(self):
    print(f"Building inverted index with {len(self.__data['docmap'])} documents")
    with open(self.__db_path) as f:
      movies = load(f)['movies']
      for movie in movies:
        id, title, description = itemgetter('id', 'title', 'description')(movie)
        self.__data['docmap'][id] = movie
        self.__process_doc(id, f"{title} {description}")
    self.__eval_stats()

  def save(self):
    print(f"Saving inverted index to cache")
    os.makedirs("cache", exist_ok=True)
    self.__save_cache_file('index')
    self.__save_cache_file('docmap')
    self.__save_cache_file('term_frequencies')
    self.__save_cache_file('doc_lengths')
  
  def load(self):
    self.__load_cache_file('index')
    self.__load_cache_file('docmap')
    self.__load_cache_file('term_frequencies')
    self.__load_cache_file('doc_lengths')
    self.__eval_stats()
  
  def get_tf(self, doc_id: int, term: str, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B, type: IdfType = 'normal') -> float:
    # print(f"tf = {self.__data['term_frequencies'][doc_id]}")
    tf = self.__data['term_frequencies'][doc_id]
    if not tf:
      print(f"No term frequencies for document {doc_id}")
      return 0.0
    elif term not in tf:
      print(f"No term frequency for term {term} in document {doc_id}")
      return 0.0
    else:
      naive_tf = tf[term]
      if type == 'normal':
        tf = float(naive_tf)
      elif type == 'bm25':
        doc_length = self.__data['doc_lengths'][doc_id]
        avg_doc_length = self.__stats['avg_doc_length']
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = (naive_tf * (k1 + 1)) / (naive_tf + k1 * length_norm)
      else:
        raise ValueError(f"Invalid IDF type: {type}")
      # print(f"Term frequency for term {term} in document {doc_id}: {tf}")
      return tf

  def get_idf(self, term: str, type: IdfType = 'normal') -> float:
    if term not in self.__data['index']:
      print(f"No index for term {term}")
      return 0
    else:
      n = self.__stats['total']
      df = len(self.__data['index'][term])
      # print(f"Index for term {term}: {self.__data['index'][term]}")
      # print(f"total = {n}")
      if type == 'normal':
        idf = math.log((n + 1) / (df + 1))
      elif type == 'bm25':
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
      else:
        raise ValueError(f"Invalid IDF type: {type}")
      # print(f"Inverse document frequency for term {term}: {idf}")
      return idf

class QuerySearch:
  def __init__(self, query: str, pre_processor: PreProcessor, inverted_index: InvertedIndex):
    self.__pre_processor = pre_processor
    self.__inverted_index = inverted_index
    self.processed_query = self.__pre_processor.process(query)
  
  def naive_search(self, movie: Movie) -> bool:
    return any(token in self.processed_query for token in self.__pre_processor.process(movie['title']))
  
  def search(self, type: SearchType = 'normal', limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[Movie, float]]:
    self.__inverted_index.load()
    if type == 'normal':
      results: list[Movie] = []
      index = self.__inverted_index.get_data('index')
      docmap = self.__inverted_index.get_data('docmap')
      for token in self.processed_query:
        if token in index:
          for doc_id in index[token]:
            if doc_id not in [movie['id'] for movie in results]:
              results.append(docmap[doc_id])
              if len(results) >= limit:
                return [(movie, 1.0) for movie in results]
      return [(movie, 1.0) for movie in results]
    else:
      scores: dict[int, float] = {}
      index = self.__inverted_index.get_data('index')
      docmap = self.__inverted_index.get_data('docmap')
      for token in self.processed_query:
        if token in index:
          for doc_id in index[token]:
            tf = self.__inverted_index.get_tf(doc_id, token, type=type)
            idf = self.__inverted_index.get_idf(token, type=type)
            scores[doc_id] = tf * idf if not scores.get(doc_id) else scores[doc_id] + tf * idf
      return [(docmap[doc_id], scores[doc_id]) for doc_id in sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]]

def main() -> None:
  parser = ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  search = subparsers.add_parser("search", help="Search movies using BM25")
  search.add_argument("query", type=str, help="Search query")
  subparsers.add_parser("build", help="Build the inverted index")
  tf = subparsers.add_parser("tf", help="Get the term frequency of a term in a document")
  tf.add_argument("doc_id", type=int, help="Document ID")
  tf.add_argument("term", type=str, help="Term")
  idf = subparsers.add_parser("idf", help="Get the inverse document frequency of a term")
  idf.add_argument("term", type=str, help="Term")
  tf_idf = subparsers.add_parser("tfidf", help="Get the TF-IDF of a term in a document")
  tf_idf.add_argument("doc_id", type=int, help="Document ID")
  tf_idf.add_argument("term", type=str, help="Term")
  bm25_tf = subparsers.add_parser("bm25tf", help="Get the term frequency of a term in a document")
  bm25_tf.add_argument("doc_id", type=int, help="Document ID")
  bm25_tf.add_argument("term", type=str, help="Term")
  bm25_tf.add_argument("k1", type=float, nargs='?', default=DEFAULT_BM25_K1, help="Tunable BM25 K1 parameter")
  bm25_tf.add_argument("b", type=float, nargs='?', default=DEFAULT_BM25_B, help="Tunable BM25 b parameter")
  bm25_idf = subparsers.add_parser("bm25idf", help="Get the TF-IDF of a term in a document")
  bm25_idf.add_argument("term", type=str, help="Term")
  bm25search = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
  bm25search.add_argument("query", type=str, help="Search query")

  args = parser.parse_args()

  pre_processor = PreProcessor(stop_words_path=f"{DATA_DIR}/stop_words.txt")
  inverted_index = InvertedIndex(db_path=f"{DATA_DIR}/movies.json", pre_processor=pre_processor)

  match args.command:
    case "build":
      inverted_index.build()
      inverted_index.save()
    case "search":
      query_search = QuerySearch(args.query, pre_processor, inverted_index)
      print(f"processed query = {query_search.processed_query}")
      movies = query_search.search()
      [print(f"{movie['id']} {movie['title']}") for movie in movies]
    case "tf":
      inverted_index.load()
      terms = pre_processor.process(args.term)
      for term in terms:
        print(f"tf = {inverted_index.get_tf(args.doc_id, term)}")
    case "idf":
      inverted_index.load()
      terms = pre_processor.process(args.term)
      for term in terms:
        print(f"Inverse document frequency of '{term}': {inverted_index.get_idf(term):.2f}")
    case "tfidf":
      inverted_index.load()
      terms = pre_processor.process(args.term)
      for term in terms:
        print(f"TF-IDF score of '{term}' in document '{args.doc_id}': {inverted_index.get_tf(args.doc_id, term) * inverted_index.get_idf(term):.2f}")
    case "bm25tf":
      inverted_index.load()
      terms = pre_processor.process(args.term)
      for term in terms:
        print(f"TF-IDF score of '{term}' in document '{args.doc_id}': {inverted_index.get_tf(args.doc_id, term, args.k1, args.b, type='bm25'):.2f}")
    case "bm25idf":
      inverted_index.load()
      terms = pre_processor.process(args.term)
      for term in terms:
        print(f"TF-IDF score of '{term}': {inverted_index.get_idf(term, type='bm25'):.2f}")
    case "search":
      query_search = QuerySearch(args.query, pre_processor, inverted_index)
      print(f"processed query = {query_search.processed_query}")
      movies = query_search.search()
      [print(f"{movie['id']} {movie['title']}") for movie in movies]
    case "bm25search":
      query_search = QuerySearch(args.query, pre_processor, inverted_index)
      print(f"processed query = {query_search.processed_query}")
      results = query_search.search(type='bm25')
      [print(f"({movie['id']}) {movie['title']} - {score:.2f}") for movie, score in results]
    case _:
      parser.print_help()

if __name__ == "__main__":
  main()
