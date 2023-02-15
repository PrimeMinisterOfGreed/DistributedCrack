#include "Generators.hpp"
#include "LinkedList.hpp"
#include "rngs.hpp"
#include <cassert>
#include <functional>
#include <gtest/gtest.h>
#include <queue>

TEST(test_linked_list, test_enqueue)
{
    DoubleLinkedList<int> linkedList{};
    linkedList.Enqueue(new int(1));
    linkedList.Enqueue(new int(2));
    linkedList.Enqueue(new int(3));

    auto itr = linkedList.begin();
    itr++;
    assert(*itr == 2);
    assert(*linkedList.begin() == 1);
    assert(*linkedList.end() == 3);

    for (int i = 1; i < 4; i++)
        assert(linkedList.Dequeue() == i);
}

TEST(test_linked_list, test_push)
{
    DoubleLinkedList<int> linkedList{};
    linkedList.Push(new int(1));
    linkedList.Push(new int(2));
    linkedList.Push(new int(3));

    auto itr = linkedList.begin();
    itr++;
    assert(*itr == 2);
    assert(*linkedList.begin() == 3);
    assert(*linkedList.end() == 1);

    for (int i = 0; i < 3; i++)
        assert(linkedList.Pull() == 3 - i);
}

TEST(test_linked_list, test_insertion)
{
    DoubleLinkedList<int> linkedList{};
    std::function<bool(int, int)> comparator = [](int a, int b) { return a > b; };
    linkedList.Insert(new int(2), comparator);
    linkedList.Insert(new int(3), comparator);
    linkedList.Insert(new int(1), comparator);
    linkedList.Insert(new int(5), comparator);
    linkedList.Insert(new int(4), comparator);

    auto itr = linkedList.begin();
    itr++;
    assert(*itr == 2);
    itr++;
    assert(*itr == 3);
    assert(*linkedList.begin() == 1);
    assert(*linkedList.end() == 5);
}

TEST(test_linked_list, test_random_comparator)
{
    DoubleLinkedList<double> linkedList{};
    std::function<bool(double, double)> comparator = [](double a, double b) { return a > b; };
    for (int i = 0; i < 1000; i++)
    {
        linkedList.Insert(NegExp(1 / 100), comparator);
    }
    double lastVal = linkedList.Dequeue();
    while (linkedList.Count() > 0)
    {
        double dequed = linkedList.Dequeue();
        assert(lastVal <= dequed);
        lastVal = dequed;
    }
}

TEST(test_linked_list, test_random_enqueuement)
{
    DoubleLinkedList<double> linkedList{};
    std::queue<double> queue{};
    for (int i = 0; i < 1000; i++)
    {
        double val = NegExp(1 / 20);
        queue.emplace(val);
        linkedList.Enqueue(val);
    }

    while (linkedList.Count() > 0)
    {
        auto v = linkedList.Dequeue();
        assert(v == queue.front());
        queue.pop();
    }

    for (int i = 0; i < 1000; i++)
    {
        double val = NegExp(1 / 20);
        queue.emplace(val);
        linkedList.Enqueue(val);
    }

    while (linkedList.Count() > 0)
    {
        auto v = linkedList.Dequeue();
        assert(v == queue.front());
        queue.pop();
    }
}

TEST(test_linked_list, TestClearance)
{
    DoubleLinkedList<double> linkedList{};
    std::queue<double> queue{};
    for (int i = 0; i < 1000; i++)
    {
        double val = NegExp(1 / 20);
        queue.emplace(val);
        linkedList.Enqueue(new double(val));
    }
    linkedList.Clear();
    assert(linkedList.Count() == 0);
   
}