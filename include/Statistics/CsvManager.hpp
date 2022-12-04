#pragma once
#include <fstream>
#include <ios>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

template <typename T> class RowIterator
{
  private:
    const std::vector<std::vector<T>> &_values;
    int _currentRow = 0;

  public:
    RowIterator(const std::vector<std::vector<T>> &values) : _values(values)
    {
    }

    std::vector<T> operator*()
    {
        std::vector<T> values{};
        for (std::vector<T> v : _values)
            values.push_back(v.at(_currentRow));
        return values;
    }

    RowIterator<T> &operator++(int)
    {
        _currentRow++;
        return *this;
    }
    RowIterator<T> &operator--(int)
    {
        _currentRow--;
        return *this;
    }

    bool operator==(RowIterator<T> &itr)
    {
        return _values == itr._values && _currentRow == itr._currentRow;
    }

    RowIterator<T> end()
    {
        RowIterator<T> itr{_values};
        itr._currentRow = _values.size();
        return itr;
    }

    RowIterator<T> begin()
    {
        RowIterator<T> itr{_values};
        itr._currentRow = 0;
        return itr;
    }
};

class CsvManager
{
  private:
    std::fstream *_file;
    const char *_filePath;

  public:
    CsvManager(const char *path);
    void Save(const std::vector<std::string> &headers, RowIterator<double> rowItr);
    std::pair<std::vector<std::string>, RowIterator<double>> Load();
};
