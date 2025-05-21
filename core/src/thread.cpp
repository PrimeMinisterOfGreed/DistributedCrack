#include "thread.hpp"

Thread::Thread(void *(*func)(void *)) : func(func) {}

void Thread::start(void *arg) { pthread_create(&thread, nullptr, func, arg); }

void Thread::join(void *return_value) { pthread_join(thread, &return_value); }

void Thread::detach() { pthread_detach(thread); }

Mutex::Mutex() : mutex(PTHREAD_MUTEX_INITIALIZER) {}

Mutex::~Mutex() { pthread_mutex_destroy(&mutex); }

void Mutex::lock() { pthread_mutex_lock(&mutex); }

void Mutex::unlock() { pthread_mutex_unlock(&mutex); }

Condition::Condition() : cond(PTHREAD_COND_INITIALIZER) {}

Condition::~Condition() { pthread_cond_destroy(&cond); }

void Condition::wait(Mutex &mutex) {
    pthread_cond_wait(&cond, &mutex.mutex);
}
void Condition::signal() { pthread_cond_signal(&cond); }

void Condition::broadcast() {
    pthread_cond_broadcast(&cond);
}
