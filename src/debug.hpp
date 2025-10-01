#pragma once
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <string_view>
#include <vector>
#include <optional>
#include <source_location>
#include "mpi.h"
#include <limits>

#ifndef INT_MAX
#define INT_MAX std::numeric_limits<int>::max();
#endif
// ---------- formatting (C++20 std::format if available) ----------
#if defined(__cpp_lib_format) && __cpp_lib_format >= 201907L
  #include <format>
  template <class... Args>
  static inline std::string _fmt(std::string_view f, Args&&... args) {
    return std::vformat(f, std::make_format_args(std::forward<Args>(args)...));
  }
#else
  // Fallback: very small %s-only shim to avoid pulling extra deps.
  // Prefer keeping messages simple (pre-format to a std::string if needed).
  template <class... Args>
  static inline std::string _fmt(std::string_view f, Args&&...) {
    return std::string(f);
  }
#endif

// ---------- rank helper (keep it here so callers don't duplicate) ----------
static inline int rank_of(MPI_Comm c) { int r=0; MPI_Comm_rank(c,&r); return r; }

// ---------- levels & config ----------
enum class LogLevel { off=0, error=1, warn=2, info=3, trace=4 };

struct DebugCfg {
  LogLevel level = LogLevel::off;        // default off
  bool rank0_only = false;               // only rank 0 prints
  std::optional<std::vector<int>> ranks; // whitelist ranks if set
  int every = 1;                         // print every Nth iteration
  int from_it = 0;                       // start iteration (inclusive)
  int to_it = INT_MAX;                   // stop iteration (inclusive)
  std::optional<std::vector<std::string>> cats; // category whitelist
};

static inline DebugCfg& dbgcfg() {
  static DebugCfg cfg;
  static bool inited=false;
  if (!inited) {
    inited=true;
    auto getenv_s = [](const char* k)->const char*{const char* v=std::getenv(k);return v?v:"";};
    auto level_s = std::string(getenv_s("BRATU_LOG")); // off|error|warn|info|trace
    auto tolower_str=[&](std::string s){for(auto&c:s)c=std::tolower(c);return s;};
    level_s = tolower_str(level_s);
    if (level_s=="error") cfg.level=LogLevel::error;
    else if (level_s=="warn") cfg.level=LogLevel::warn;
    else if (level_s=="info") cfg.level=LogLevel::info;
    else if (level_s=="trace") cfg.level=LogLevel::trace;
    else cfg.level=LogLevel::off;

    if (std::string(getenv_s("BRATU_LOG_RANK0"))=="1") cfg.rank0_only=true;

    if (const char* ev=getenv("BRATU_LOG_RANKS"); ev && *ev) {
      cfg.ranks.emplace();
      int x=0; const char* p=ev;
      while (*p) { if (std::isdigit(*p)) { x=std::strtol(p,(char**)&p,10); cfg.ranks->push_back(x);}
                   else ++p; }
    }
    if (const char* ev=getenv("BRATU_LOG_EVERY"); ev && *ev) cfg.every=std::max(1,std::atoi(ev));
    if (const char* ev=getenv("BRATU_LOG_FROM");  ev && *ev) cfg.from_it=std::atoi(ev);
    if (const char* ev=getenv("BRATU_LOG_TO");    ev && *ev) cfg.to_it  =std::atoi(ev);
    if (const char* ev=getenv("BRATU_LOG_CATS");  ev && *ev) {
      cfg.cats.emplace();
      std::string s(ev); std::string cur;
      for (char c: s) { if (c==','){ if(!cur.empty()) cfg.cats->push_back(cur), cur.clear(); } else cur.push_back(std::tolower(c)); }
      if (!cur.empty()) cfg.cats->push_back(cur);
    }
  }
  return cfg;
}

static inline bool _cat_allowed(std::string_view cat) {
  auto& cfg = dbgcfg();
  if (!cfg.cats) return true;
  std::string lc; lc.reserve(cat.size());
  for (char c: cat) lc.push_back(std::tolower(c));
  for (auto& c: *cfg.cats) if (c == lc) return true;
  return false;
}

static inline bool _rank_allowed(MPI_Comm comm) {
  auto& cfg = dbgcfg();
  int r = rank_of(comm);
  if (cfg.rank0_only && r!=0) return false;
  if (cfg.ranks) {
    for (int k: *cfg.ranks) if (k==r) return true;
    return false;
  }
  return true;
}

static inline bool _iter_allowed(int it) {
  auto& cfg = dbgcfg();
  if (it < cfg.from_it || it > cfg.to_it) return false;
  if (cfg.every <= 1) return true;
  return (it - cfg.from_it) % cfg.every == 0;
}

static inline bool _should_log(LogLevel lvl, std::string_view cat, MPI_Comm comm, int it) {
  if (lvl == LogLevel::off) return false;
  if ((int)lvl > (int)dbgcfg().level) return false;
  if (!_cat_allowed(cat)) return false;
  if (!_rank_allowed(comm)) return false;
  return _iter_allowed(it);
}

// ---------- core log ----------
template <class... Args>
static inline void _log(LogLevel lvl, std::string_view cat, MPI_Comm comm,
                        int it, std::string_view fmt,
                        const std::source_location loc = std::source_location::current(),
                        Args&&... args) {
  if (!_should_log(lvl, cat, comm, it)) return;
  int r = rank_of(comm);
  auto msg = _fmt(fmt, std::forward<Args>(args)...);
  std::fprintf(stderr, "[%d] %.*s it=%d %s:%u %s\n",
               r, (int)cat.size(), cat.data(), it,
               loc.file_name(), loc.line(), msg.c_str());
  std::fflush(stderr);
}

// ---------- user-facing macros ----------
#define LOGE(cat, it, fmt, ...) _log(LogLevel::error, (cat), MPI_COMM_WORLD, (it), (fmt), std::source_location::current(), ##__VA_ARGS__)
#define LOGW(cat, it, fmt, ...) _log(LogLevel::warn,  (cat), MPI_COMM_WORLD, (it), (fmt), std::source_location::current(), ##__VA_ARGS__)
#define LOGI(cat, it, fmt, ...) _log(LogLevel::info,  (cat), MPI_COMM_WORLD, (it), (fmt), std::source_location::current(), ##__VA_ARGS__)
#define LOGT(cat, it, fmt, ...) _log(LogLevel::trace, (cat), MPI_COMM_WORLD, (it), (fmt), std::source_location::current(), ##__VA_ARGS__)

// ---------- RAII phase (enter/exit paired automatically) ----------
struct ScopedPhase {
  MPI_Comm comm;
  std::string cat, name;
  int it;
  ScopedPhase(MPI_Comm c, std::string_view ccat, std::string_view n, int i)
    : comm(c), cat(ccat), name(n), it(i) {
    LOGT(cat.c_str(), it, "ENTER {}", name);
  }
  ~ScopedPhase() {
    LOGT(cat.c_str(), it, "EXIT  {}", name);
  }
};

// ---------- MPI checked call with trace ----------
#define MPI_CALL(comm, it, name, call)                                         \
  do {                                                                          \
    LOGT("mpi", (it), "ENTER {}", (name));                                      \
    int _rc = (call);                                                           \
    if (_rc != MPI_SUCCESS) {                                                   \
      char s[MPI_MAX_ERROR_STRING]; int l=0;                                    \
      MPI_Error_string(_rc, s, &l);                                             \
      std::fprintf(stderr, "[%d] mpi ERROR in %s: %.*s\n",                      \
                   rank_of(comm), (name), l, s);                                \
      std::fflush(stderr);                                                      \
      MPI_Abort((comm), _rc);                                                   \
    }                                                                           \
    LOGT("mpi", (it), "EXIT  {}", (name));                                      \
  } while(0)

// ---------- asserts / checks ----------
#define ASSERTF(comm, cond, fmt, ...)                                          \
  do { if (!(cond)) {                                                          \
    std::fprintf(stderr, "[%d] ASSERT FAIL: %s:%d: ",                          \
                 rank_of(comm), __FILE__, __LINE__);                           \
    auto _m = _fmt((fmt), ##__VA_ARGS__);                                      \
    std::fprintf(stderr, "%s\n", _m.c_str());                                  \
    std::fflush(stderr);                                                       \
    MPI_Abort((comm), 1);                                                      \
  } } while(0)

template <class Vec>
static inline void check_finite(MPI_Comm comm, std::string_view cat, int it,
                                std::string_view name, const Vec& v) {
  for (size_t i=0;i<v.size();++i)
    if (!std::isfinite(v[i])) {
      LOGE(cat, it, "nonfinite {} at [{}] = {}", name, (int)i, v[i]);
      MPI_Abort(comm, 2);
    }
}
