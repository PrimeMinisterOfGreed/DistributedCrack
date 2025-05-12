#pragma once
#include <pthread.h>

struct Thread{
    private:
    pthread_t thread;
    void* (*func)(void*);
    public:
    Thread(void* (*func)(void*));
    Thread(void* (*func)(void*), void*arg);

    void start(void* arg);
    void join();
    void detach();
    void cancel();

};
