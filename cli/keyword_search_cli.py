#!/usr/bin/env python3

from argparse import ArgumentParser
from string import punctuation
from json import load
from typing import TypedDict
from nltk.stem import PorterStemmer

class Movie(TypedDict):
  id: int
  title: str
  description: str

class QuerySearch:
  def __init__(self, query: str, stop_words_path: str):
    self.stemmer = PorterStemmer()
    self.stop_words = None
    with open(stop_words_path) as f:
      self.stop_words = f.read().splitlines()
    self.processed_query = self.process(query)

  def sanitize(self, input: str) -> str:
    return input.lower().translate(str.maketrans('', '', punctuation))

  def tokenize(self, input: str) -> list[str]:
    return input.split(" ")

  def remove_stop_words(self, tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in self.stop_words]

  def stem(self, tokens: list[str]) -> list[str]:
    return [self.stemmer.stem(token) for token in tokens]

  def process(self, input: str) -> list[str]:
    return self.stem(self.remove_stop_words(self.tokenize(self.sanitize(input))))

  def is_hit(self, movie: Movie) -> bool:
    # for title_token in self.process(movie['title']):
    #   for query_token in self.processed_query:
    #     if query_token in title_token:
    #       return True
    # return False
    return any(token in self.processed_query for token in self.process(movie['title']))

def main() -> None:
  parser = ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  args = parser.parse_args()

  match args.command:
    case "search":
      query_search = QuerySearch(query=args.query, stop_words_path="./data/stop_words.txt")
      print(f"processed query = {query_search.processed_query}")
      with open("./data/movies.json") as f:
        movies = load(f)
        for movie in movies['movies']:
          if query_search.is_hit(movie):
            print(movie['title'])
    case _:
      parser.print_help()

if __name__ == "__main__":
  main()
