//
// Created by drfaust on 28/01/23.
//
#pragma once

#include <functional>
#include <cstdint>

template<typename ...Args>
    class IHandler
    {
    public:
        virtual void operator()(Args...) = 0;
        virtual bool compare(const IHandler<Args...>& oth) = 0;
    };

template<typename ...Args>
    class FunctionHandler : public IHandler<Args...>
    {
    private:
        std::function<void(Args...)> _handler;
        static size_t _generated;
        size_t _ordType = -1;
    public:
        FunctionHandler(std::function<void(Args...)> handler): _handler(handler){
            _ordType = _generated;
            _generated++;
            _generated %= INT64_MAX; // i don't think you'll need this, but i agree that people that do this jobs are mad mans so just in case....
        }

         void operator()(Args... args) override
        {
            _handler(args...);
        }

        bool compare(const IHandler<Args...>& oth) override
        {
            const auto* othH= dynamic_cast<const FunctionHandler<Args...>*>(&oth);
            return othH != nullptr && othH->_ordType == _ordType;
        }


    };

template<typename ...Args>
    size_t FunctionHandler<Args...>::_generated = 0;

/**
 * EventHandler in C# style, use FunctionHandler to implement lambda type handlers
 * @tparam Args
 * @example with lambda EventHandler handler += new FunctionHandler<>([](){});
 */
template<typename ...Args>
    class EventHandler
    {
    private:
        std::vector<IHandler<Args...>*> _handlers;

    public:


        void operator+=(IHandler<Args...>* handler)
        {
            _handlers.push_back(handler);
        }


        void operator-=(IHandler<Args...>* handler)
        {

            std::remove_if(_handlers.begin(),_handlers.end(),
                           [handler](IHandler<Args...>* h){ return h->compare(*handler);});
        }


        void Invoke(Args... args)
        {
            for (auto handler: _handlers)
            {
                (*handler)(args...);
            }
        }

        virtual ~EventHandler() = default;
    };