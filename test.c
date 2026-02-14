#define VDB_MULTITHREADED
#include "vdb.h"
#include <stdio.h>

int main(void) {
  vdb_database* db = vdb_create(128, VDB_METRIC_COSINE);

  if (!db) {
    printf("vdb_create\n");
    return 1;
  }

  float vec1[128], vec2[128], vec3[128];

  for (int i = 0; i < 128; i++) {
    vec1[i] = (float)i / 128.0f;
    vec2[i] = (float)(i + 10) / 128.0f;
    vec3[i] = (float)(i + 50) / 128.0f;
  }

  vdb_add_vector(db, vec1, "vec1", NULL);
  vdb_add_vector(db, vec2, "vec2", NULL);
  vdb_add_vector(db, vec3, "vec3", NULL);

  printf("database contains %zu vectors\n", vdb_count(db));

  float query[128];

  for (int i = 0; i < 128; i++) {
    query[i] = (float)(i + 5) / 128.0f;
  }

  vdb_result_set* results = vdb_search(db, query, 2);

  if (results) {
    printf("top %zu results:\n", results->count);

    for (size_t i = 0; i < results->count; i++) {
      printf("  %zu. id=%s, distance=%.4f\n", i + 1,
             results->results[i].id ? results->results[i].id : "NULL",
             results->results[i].distance);
    }

    vdb_free_result_set(results);
  }

  vdb_save(db, "test.vdb");
  vdb_destroy(db);

  vdb_database* loaded = vdb_load("test.vdb");

  if (loaded) {
    printf("loaded database with %zu vectors\n", vdb_count(loaded));
    vdb_destroy(loaded);
  }

  return 0;
}
