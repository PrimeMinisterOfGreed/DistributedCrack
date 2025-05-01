#pragma once

#include <cstdlib>
#include <cstdio>
template<typename Ok, typename Error>
struct Result{
    bool _is_ok;
    union {
        Ok ok;
        Error error;
    };
    Result(Ok ok) : _is_ok(true), ok(ok) {}
    Result(Error error) : _is_ok(false), error(error) {}
    ~Result() {
        if (!_is_ok) {
            error.~Error();
        } else {
            ok.~Ok();
        }
    }

    bool is_ok() const {
        return _is_ok;
    }
    bool is_error() const {
        return !_is_ok;
    }

    Ok& unwrap() {
        if (!_is_ok) {
            printf("Error: Attempt to access ok value when Result is in error state.\n");
            exit(1);
        }
        return ok;
    }

    Error& unwrap_error() {
        if (_is_ok) {
            printf("Error: Attempt to access error value when Result is in ok state.\n");
            exit(1);
        }
        return error;
    }
    
    Result(const Result&) = delete;

};