// Archive #129/D.4: distribute live query tiles, preserving every task exactly once.
// This is fixed host-known integer geometry, not metadata readback or extra KV storage.
#pragma once
#include <cstdint>
#ifndef OSCAR_SCHEDULE_FN
#define OSCAR_SCHEDULE_FN inline
#define OSCAR_SCHEDULE_LOCAL_MACRO
#endif
namespace oscar_ascend_schedule {
struct CvTaskSchedule {
  int64_t tokens, queryTile, perToken;
  OSCAR_SCHEDULE_FN int64_t QueryTiles() const {return (tokens+queryTile-1)/queryTile;}
  OSCAR_SCHEDULE_FN int64_t WorkItems() const {return QueryTiles()*perToken;}
  OSCAR_SCHEDULE_FN int64_t TokenBegin(int64_t workId) const {return (workId%QueryTiles())*queryTile;}
  OSCAR_SCHEDULE_FN int64_t Segment(int64_t workId) const {return workId/QueryTiles();}
  OSCAR_SCHEDULE_FN int64_t TaskId(int64_t workId,int64_t token) const {return token*perToken+Segment(workId);}
};
}
#ifdef OSCAR_SCHEDULE_LOCAL_MACRO
#undef OSCAR_SCHEDULE_FN
#undef OSCAR_SCHEDULE_LOCAL_MACRO
#endif
