//
// Created by drfaust on 03/02/23.
//

#pragma once

#include "EventHandler.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

template <typename T = double> class Measure
{

  private:
    int _moments;
    T * _sum;
    T _max;
    T _min;
    size_t _count;

    T get(int moment) const
    {
        if (moment >= _moments)
            throw std::invalid_argument("moment selected is over Moments stored");
        return _sum[moment];
    }

    void constexpr ForMoment(std::function<void(T &, int)> operation)
    {
        for (int i = 0; i < _moments; i++)
            operation(_sum[i], i + 1);
    }

    std::string _name;
    std::string _unit;

  public:
    EventHandler<T> OnCatch;

    Measure(int moments = 2) : _count(0), _moments(moments)
    {
        _sum = new T[_moments];
    }

    void Accumulate(T value)
    {
        ForMoment([value](T &val, int moment) { val += pow(value, moment); });
        _count++;
        OnCatch.Invoke(value);
        if (_max < value)
            _max = value;
        if (_min > value)
            _min = value;
    }

    inline void operator()(T value)
    {
        Accumulate(value);
    }

    void Reset()
    {
        ForMoment([](T &val, int moment) { val = 0; });
    }

    inline int count() const
    {
        return _count;
    }

    inline T mean(int moment = 0) const
    {
        return get(moment) / _count;
    }

    inline T variance() const
    {
        return mean(1) - pow(mean(0), 2); // this can lead to catastrophic cancellation
    }

    inline T sum() const
    {
        return _sum[0];
    }

    inline T max() const
    {
        return _max;
    }
    inline int moments() const{return _moments;}

};

template <typename T = double> class CovariatedMeasure
{
  private:
    
    Measure<T> &_m1;
    Measure<T> &_m2;
    size_t _count = 0;

  public:
    CovariatedMeasure(Measure<T> &m1, Measure<T> &m2) : _m1{m1}, _m2(m2)
    {
    }

    CovariatedMeasure() : _m1{*new Measure<T>{}}, _m2{*new Measure<T>{}}
    {
    }

    void Accumulate(T v1, T v2)
    {
        _m1(v1);
        _m2(v2);
    }

    void operator()(T v1, T v2)
    {
        Accumulate(v1, v2);
    }

    T covariation() const
    {
    }
};
