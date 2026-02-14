import ctypes
import os
import tempfile
import subprocess
from ctypes import c_void_p, c_char_p, c_size_t, c_float, c_int, POINTER, Structure

class VDBError:
  OK = 0
  NULL_POINTER = -1
  INVALID_DIMENSIONS = -2
  OUT_OF_MEMORY = -3
  NOT_FOUND = -4
  INVALID_INDEX = -5
  THREAD_FAILURE = -6

class VDBMetric:
  COSINE = 0
  EUCLIDEAN = 1
  DOT_PRODUCT = 2

class VDBResult(Structure):
  _fields_ = [
    ("index", c_size_t),
    ("distance", c_float),
    ("id", c_char_p),
    ("metadata", c_void_p)
  ]

class VDBResultSet(Structure):
  _fields_ = [
    ("results", POINTER(VDBResult)),
    ("count", c_size_t)
  ]

class VectorDatabase:
  _lib = None
  _lib_path = None
  
  @classmethod
  def _compile_library(cls, multithreaded=True):
    if cls._lib is not None:
      return
    
    vdb_header = os.path.join(os.path.dirname(__file__), 'vdb.h')
    if not os.path.exists(vdb_header):
      vdb_header = 'vdb.h'
    
    temp_dir = tempfile.gettempdir()
    c_file = os.path.join(temp_dir, 'vdb_wrapper.c')
    
    wrapper_code = '''
#define VDB_MULTITHREADED
#include "vdb.h"

vdb_database* wrap_vdb_create(size_t dims, int metric) {
  return vdb_create(dims, (vdb_metric)metric);
}

int wrap_vdb_add_vector(vdb_database* db, float* data, const char* id) {
  return vdb_add_vector(db, data, id, NULL);
}

vdb_result_set* wrap_vdb_search(vdb_database* db, float* query, size_t k) {
  return vdb_search(db, query, k);
}

void wrap_vdb_free_result_set(vdb_result_set* rs) {
  vdb_free_result_set(rs);
}

void wrap_vdb_destroy(vdb_database* db) {
  vdb_destroy(db);
}

size_t wrap_vdb_count(vdb_database* db) {
  return vdb_count(db);
}

size_t wrap_vdb_dimensions(vdb_database* db) {
  return vdb_dimensions(db);
}

int wrap_vdb_save(vdb_database* db, const char* filename) {
  return vdb_save(db, filename);
}

vdb_database* wrap_vdb_load(const char* filename) {
  return vdb_load(filename);
}

int wrap_vdb_remove_vector(vdb_database* db, size_t index) {
  return vdb_remove_vector(db, index);
}
'''
    
    with open(c_file, 'w') as f:
      f.write(wrapper_code)
    
    lib_name = 'libvdb.so'
    lib_path = os.path.join(temp_dir, lib_name)
    
    compile_cmd = [
      'gcc', '-shared', '-fPIC', '-O3',
      '-I' + os.path.dirname(vdb_header),
      c_file, '-o', lib_path,
      '-lm', '-lpthread'
    ]
    
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(f"Compilation failed: {result.stderr}")
    
    cls._lib_path = lib_path
    cls._lib = ctypes.CDLL(lib_path)
    
    cls._lib.wrap_vdb_create.argtypes = [c_size_t, c_int]
    cls._lib.wrap_vdb_create.restype = c_void_p
    
    cls._lib.wrap_vdb_add_vector.argtypes = [c_void_p, POINTER(c_float), c_char_p]
    cls._lib.wrap_vdb_add_vector.restype = c_int
    
    cls._lib.wrap_vdb_search.argtypes = [c_void_p, POINTER(c_float), c_size_t]
    cls._lib.wrap_vdb_search.restype = POINTER(VDBResultSet)
    
    cls._lib.wrap_vdb_free_result_set.argtypes = [POINTER(VDBResultSet)]
    cls._lib.wrap_vdb_free_result_set.restype = None
    
    cls._lib.wrap_vdb_destroy.argtypes = [c_void_p]
    cls._lib.wrap_vdb_destroy.restype = None
    
    cls._lib.wrap_vdb_count.argtypes = [c_void_p]
    cls._lib.wrap_vdb_count.restype = c_size_t
    
    cls._lib.wrap_vdb_dimensions.argtypes = [c_void_p]
    cls._lib.wrap_vdb_dimensions.restype = c_size_t
    
    cls._lib.wrap_vdb_save.argtypes = [c_void_p, c_char_p]
    cls._lib.wrap_vdb_save.restype = c_int
    
    cls._lib.wrap_vdb_load.argtypes = [c_char_p]
    cls._lib.wrap_vdb_load.restype = c_void_p
    
    cls._lib.wrap_vdb_remove_vector.argtypes = [c_void_p, c_size_t]
    cls._lib.wrap_vdb_remove_vector.restype = c_int
  
  def __init__(self, dimensions, metric=VDBMetric.COSINE, multithreaded=True):
    VectorDatabase._compile_library(multithreaded)
    self.db = self._lib.wrap_vdb_create(dimensions, metric)
    if not self.db:
      raise RuntimeError("Failed to create database")
    self.dimensions = dimensions
    self.metric = metric
  
  def add_vector(self, vector, vector_id=None):
    if len(vector) != self.dimensions:
      raise ValueError(f"Vector dimension mismatch: expected {self.dimensions}, got {len(vector)}")
    
    arr = (c_float * len(vector))(*vector)
    id_bytes = vector_id.encode('utf-8') if vector_id else None
    
    result = self._lib.wrap_vdb_add_vector(self.db, arr, id_bytes)
    if result != VDBError.OK:
      raise RuntimeError(f"Failed to add vector: error {result}")
  
  def search(self, query, k=5):
    if len(query) != self.dimensions:
      raise ValueError(f"Query dimension mismatch: expected {self.dimensions}, got {len(query)}")
    
    arr = (c_float * len(query))(*query)
    result_set_ptr = self._lib.wrap_vdb_search(self.db, arr, k)
    
    if not result_set_ptr:
      return []
    
    result_set = result_set_ptr.contents
    results = []
    
    for i in range(result_set.count):
      res = result_set.results[i]
      results.append({
        'index': res.index,
        'distance': res.distance,
        'id': res.id.decode('utf-8') if res.id else None
      })
    
    self._lib.wrap_vdb_free_result_set(result_set_ptr)
    return results
  
  def remove_vector(self, index):
    result = self._lib.wrap_vdb_remove_vector(self.db, index)
    if result != VDBError.OK:
      raise RuntimeError(f"Failed to remove vector: error {result}")
  
  def count(self):
    return self._lib.wrap_vdb_count(self.db)
  
  def save(self, filename):
    result = self._lib.wrap_vdb_save(self.db, filename.encode('utf-8'))
    if result != VDBError.OK:
      raise RuntimeError(f"Failed to save database: error {result}")
  
  @classmethod
  def load(cls, filename):
    cls._compile_library()
    db_ptr = cls._lib.wrap_vdb_load(filename.encode('utf-8'))
    if not db_ptr:
      raise RuntimeError("Failed to load database")
    
    instance = cls.__new__(cls)
    instance.db = db_ptr
    instance.dimensions = cls._lib.wrap_vdb_dimensions(db_ptr)
    instance.metric = VDBMetric.COSINE
    return instance
  
  def __del__(self):
    if hasattr(self, 'db') and self.db:
      self._lib.wrap_vdb_destroy(self.db)
  
  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.db:
      self._lib.wrap_vdb_destroy(self.db)
      self.db = None
