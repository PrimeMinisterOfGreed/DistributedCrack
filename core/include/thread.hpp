#pragma once
#include <pthread.h>

struct Thread{
    private:
    pthread_t thread;
    void* (*func)(void*);
    public:
    Thread(void* (*func)(void*));
    void start(void* arg);
    void join(void* return_value = NULL);
    void detach();
    void cancel();
};


struct Mutex{
    pthread_mutex_t mutex;
    Mutex();
    ~Mutex();
    void lock();
    void unlock();
};


struct Condition{
    private:
    pthread_cond_t cond;
    public:
    Condition();
    ~Condition();
    void
    wait(Mutex& mutex);
    void signal();
    void broadcast();
};

