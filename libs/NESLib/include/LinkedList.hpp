#pragma once

#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>

template<class T>
    class DoubleLinkedList;

template<typename T>
    class Node
    {
        friend class DoubleLinkedList<T>;

    private:
        T *_val;

    protected:
        Node *_previous;
        Node *_next;

    public:
        Node();

        Node(T *value);

        Node(T &value);

        Node(Node &&);

        Node(Node &);

        ~Node();

        void operator>>(Node &next);

        void operator<<(Node &previous);

        T *Value() const;

        T &operator*() const
        {
            return *Value();
        }

        Node<T> *Next() const
        {
            return _next;
        }

        Node<T> *Previous() const
        {
            return _previous;
        }
    };

template<typename T>
    inline Node<T>::Node()
    {
        _val = nullptr;
    }

template<typename T>
    inline Node<T>::Node(T *value) : _val{value}, _next{nullptr}, _previous{nullptr}
    {
    }

template<typename T>
    inline Node<T>::Node(T &value)
    {
        _val = &value;
    }

template<typename T>
    inline Node<T>::Node(Node &&n) : _val{n._val}
    {
        n.~Node();
    }

template<typename T>
    inline Node<T>::Node(Node &n) : _val{n._val}
    {
    }

template<typename T>
    inline Node<T>::~Node()
    {
        if (_val != nullptr)
        {
            delete _val;
        }
        _next = nullptr;
        _previous = nullptr;
    }



template<typename T>
    inline void Node<T>::operator>>(Node &next)
    {
        _next = &next;
        next._previous = this;
    }

template<typename T>
    inline void Node<T>::operator<<(Node &previous)
    {
        _previous = &previous;
        previous._next = this;
    }

template<typename T>
    inline T *Node<T>::Value() const
    {
        return _val;
    }

template<typename T>
    class NodeIterator
    {
    private:
        Node<T> *_currentNode;

    public:

        NodeIterator &operator++(int)
        {
            _currentNode = _currentNode->Next();
            return *this;
        }

        NodeIterator &operator--(int)
        {
            _currentNode = _currentNode->Previous();
            return *this;
        }

        NodeIterator &operator+(int offset)
        {
            for (int i = 0; i < offset; i++)
                _currentNode = _currentNode->Next();
            return *this;
        }

        NodeIterator &operator-(int offset)
        {
            for (int i = 0; i < offset; i++)
                _currentNode = _currentNode->Previous();
            return *this;
        }

        T &operator*()
        {
            return *_currentNode->Value();
        }

        Node<T> &operator()()
        {
            return *_currentNode;
        }

        NodeIterator(Node<T> *reference) : _currentNode{reference}
        {
        }

        inline bool operator==(NodeIterator &itr)
        {
            return _currentNode == &itr();
        }

        inline bool operator!=(NodeIterator &itr)
        {
            return _currentNode != &itr();
        }
    };

template<class T>
    class DoubleLinkedList
    {
    private:
        Node<T> *_begin;
        Node<T> *_end;
        size_t _count = 0;

    public:
        NodeIterator<T> begin() const
        {
            return NodeIterator(_begin);
        }

        NodeIterator<T> end() const
        {
            return NodeIterator(_end);
        }

        inline size_t Count() const
        {
            return _count;
        }

        void Push(T *val);

        void Push(T val)
        {
            Push(&val);
        }

        T &Pull();

        void Enqueue(T *val);

        void Enqueue(T val)
        {
            Enqueue(&val);
        }

        T &Dequeue();

        void Insert(T *val, int index);

        void Insert(T val, int index)
        {
            Insert(&val, index);
        }

        void Insert(T *val, std::function<bool(const T &, const T &)> comparer);

        void Insert(T val, std::function<bool(const T &, const T &)> comparer)
        {
            Insert(&val, comparer);
        }

        void Clear();

        int IndexOf(T *val);

        bool Contains(T *val);

        std::string ToString();
    };

template<class T>
    bool DoubleLinkedList<T>::Contains(T *val)
    {
        return IndexOf(val) != -1;
    }

template<class T>
    int DoubleLinkedList<T>::IndexOf(T *val)
    {
        auto itr = begin();
        int p = 0;
        while (itr != end())
        {
            if (itr() == val)
                return p;
            itr++;
        }
        return -1;
    }

template<class T>
    void DoubleLinkedList<T>::Push(T *val)
    {
        Node<T> *new_node = new Node<T>(val);
        if (_begin != nullptr)
            *_begin << *new_node;
        if (_end == nullptr)
            _end = new_node;
        _begin = new_node;
        _count++;
    }

template<class T>
    T &DoubleLinkedList<T>::Pull()
    {
        if (_begin == nullptr)
            throw new std::invalid_argument("Trying to pull from the list but is empty");
        Node<T> *res = _begin;
        if (_begin != _end)
        {
            _begin = _begin->Next();
            if (_begin != nullptr)
                _begin->_previous = nullptr;
        }
        else
        {
            _begin = nullptr;
            _end = nullptr;
        }
        auto val = res->Value();
        _count--;
        res->_next = nullptr;
        res->_previous = nullptr;
        res->_val = nullptr;
        return *val;
    }

template<class T>
    inline void DoubleLinkedList<T>::Enqueue(T *val)
    {
        Node<T> *newNode = new Node<T>(val);
        if (_begin == nullptr)
        {
            _begin = newNode;
            _end = newNode;
        }
        else
        {
            auto ptr = _end;
            *ptr >> *newNode;
            *newNode << *ptr;
            _end = newNode;
        }
        _count++;
    }

template<class T>
    inline T &DoubleLinkedList<T>::Dequeue()
    {
        return Pull();
    }

template<class T>
    void DoubleLinkedList<T>::Insert(T *val, int index)
    {
        if (index > _count)
            throw new std::invalid_argument("Index out of bounds");
        if (index == 0)
            Push(val);
        else if (index == _count)
            Enqueue(val);
        else
        {
            Node<T> &newNode = *new Node<T>(val);
            auto itr = (begin() + index);
            auto ptr = itr();
            Node<T> &prev = (itr--)();
            prev >> newNode;
            ptr << newNode;
            _count++;
        }
    }

template<class T>
    void DoubleLinkedList<T>::Insert(T *val, std::function<bool(const T &, const T &)> comparer)
    {
        Node<T> &newNode = *new Node<T>(val);
        auto itr = begin();
        if (_begin == nullptr || comparer(*itr, *newNode))
            Push(val);
        else
        {
            while (itr != end() && !comparer(*itr, *newNode))
                itr++;

            if (itr == end() && !comparer(*itr, *newNode))
                Enqueue(val);
            else
            {
                Node<T> &prev = *itr().Previous();
                prev >> newNode;
                itr() << newNode;
                _count++;
            }
        }
    }

template<class T>
    inline void DoubleLinkedList<T>::Clear()
    {
        if(_count == 0) return;
        auto itr = begin();
        while (itr != end())
        {
            auto prev = &itr();
            itr++;
            if (prev->Value() != nullptr)
            {
                delete prev;
            }
        }
        _begin = nullptr;
        _end = nullptr;
        _count = 0;
    }
