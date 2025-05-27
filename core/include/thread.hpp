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


struct thread_block{
  int thread_id;
  int n_threads;

  thread_block(int id,int n) : thread_id(id), n_threads(n) {}
  thread_block() = default;
};

template <typename F>
  requires std::invocable<F, thread_block>
void parallel_for(int n, F &&fnc) {
  struct thread_ctx {
    thread_block blk;
    F *fnc;
    thread_ctx(int id, int n_threads, F *f) :  fnc(f), blk(id,n_threads) {}
    thread_ctx() = default;
  };

  Thread threads[n];
  thread_ctx ctxs[n];
  for (int i = 0; i < n; ++i) {
    threads[i].create([](void *ctx) -> void * {
      thread_ctx *context = static_cast<thread_ctx *>(ctx);
        (*(context->fnc))(context->blk);
      return nullptr;
    });
    ctxs[i] = thread_ctx{i, n, &fnc};
    threads[i].start(&ctxs[i]);
  }

  for (int i = 0; i < n; ++i) {
    threads[i].join();
  }
}