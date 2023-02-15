//
// Created by drfaust on 03/02/23.
//

#pragma once

#include <string>
#include "EventHandler.hpp"
#include <stdexcept>
#include <cmath>

template<typename T = double, int Moments = 2>
    class Measure
    {

    private:
        T _sum[Moments];

        size_t _count;


        T get(int moment) const
        {
            if (moment >= Moments) throw std::invalid_argument("moment selected is over Moments stored");
            return _sum[moment];
        }

        void constexpr ForMoment(std::function<void(T &, int)> operation)
        {
            for (int i = 0; i < Moments; i++)
                operation(_sum[i], i + 1);
        }

        std::string _name;
        std::string _unit;
    public:

        EventHandler<T> OnCatch;

        Measure() : _count(0)
        {

        }

        void Accumulate(T value)
        {
            ForMoment([value](T &val, int moment)
                      { val += pow(value, moment); });
            _count++;
            OnCatch.Invoke(value);
        }

        inline void operator()(T value)
        { Accumulate(value); }

        void Reset()
        {
            ForMoment([](T &val, int moment)
                      { val = 0; });
        }

        inline int count() const
        { return _count; }

        inline T mean(int moment = 0) const
        {
            return get(moment) / _count;
        }

        inline T variance() const
        {
            return mean(1) - pow(mean(0), 2); // this can lead to catastrophic cancellation
        }

        inline T sum() const
        { return _sum[0]; }
    };

template<typename T = double, int Moments = 2>
    class CovariatedMeasure
    {
    private:
        T _acc[Moments];
        Measure<T, Moments> &_m1;
        Measure<T, Moments> &_m2;
        size_t _count = 0;
    public:
        CovariatedMeasure(Measure<T, Moments> &m1, Measure<T, Moments> &m2) : _m1{m1}, _m2(m2)
        {}

        CovariatedMeasure() : _m1{*new Measure<T, Moments>{}}, _m2{*new Measure<T, Moments>{}}
        {}

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
