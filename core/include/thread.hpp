#pragma once
#include <cmath>
#include <concepts>
#include <pthread.h>
#include <tuple>
#include <vector>
struct Thread {
private:
  pthread_t thread;
  void *(*func)(void *);

public:
  Thread(void *(*func)(void *));
  Thread();
  Thread &create(void *(*func)(void *));
  void start(void *arg);
  void join(void *return_value = NULL);
  void detach();
  void cancel();
};

struct Mutex {
  pthread_mutex_t mutex;
  Mutex();
  ~Mutex();
  void lock();
  void unlock();
};

struct Condition {
private:
  pthread_cond_t cond;

public:
  Condition();
  ~Condition();
  void wait(Mutex &mutex);
  void signal();
  void broadcast();
};

template <typename T, typename F>
  requires std::invocable<F, int, T &>
void parallel_for(std::vector<T> &data, int n_threads, F &&fnc) {
  struct thread_ctx {
    int thread_id;
    std::vector<T> *data;
    F *fnc;
    int n_threads;
    thread_ctx(int id, std::vector<T> *d, F *f, int n)
        : thread_id(id), data(d), fnc(f), n_threads(n) {}
    thread_ctx() = default;
  };
  int chunk_size = data.size() / n_threads;
  Thread threads[n_threads];
  thread_ctx ctxs[n_threads];
  for (int i = 0; i < n_threads; ++i) {
    threads[i].create([](void *ctx) -> void * {
      thread_ctx *context = static_cast<thread_ctx *>(ctx);
      int chunk_size = ceil((double)context->data->size() / context->n_threads);
      for (int j = context->thread_id * chunk_size;
           j < (context->thread_id + 1) * chunk_size &&
           j < context->data->size();
           ++j)
        (*(context->fnc))(j, (*context->data)[context->thread_id]);
      return nullptr;
    });
    ctxs[i] = thread_ctx{i, &data, &fnc, n_threads};
    threads[i].start(&ctxs[i]);
  }

  for (int i = 0; i < n_threads; ++i) {
    threads[i].join();
  }
}

template <typename F>
  requires std::invocable<F, int>
void parallel_for(int n, int end, F &&fnc) {
  struct thread_ctx {
    int thread_id;
    int n_threads;
    int end;
    F *fnc;
    thread_ctx(int id, int end, int n_threads, F *f) : thread_id(id), n_threads(n_threads), end(end), fnc(f) {}
    thread_ctx() = default;
  };

  Thread threads[n];
  thread_ctx ctxs[n];
  for (int i = 0; i < n; ++i) {
    threads[i].create([](void *ctx) -> void * {
      thread_ctx *context = static_cast<thread_ctx *>(ctx);
      int chunk_size = ceil((double)context->end / context->n_threads);
      for (int j = context->thread_id * chunk_size;
           j < (context->thread_id + 1) * chunk_size &&
           j < context->end;
           ++j)
        (*(context->fnc))(j);
      return nullptr;
    });
    ctxs[i] = thread_ctx{i, end,n, &fnc};
    threads[i].start(&ctxs[i]);
  }

  for (int i = 0; i < n; ++i) {
    threads[i].join();
  }
}