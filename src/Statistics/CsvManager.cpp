#include "Statistics/CsvManager.hpp"
#include <fstream>
#include <ios>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

std::vector<std::string> &split(std::string const &str, const char delim)
{
    // create a stream from the string
    std::stringstream s(str);
    std::vector<std::string> &out = *new std::vector<std::string>();
    std::string buffer;
    while (std::getline(s, buffer, delim))
    {
        out.push_back(buffer); // store the string in s2
    }
    return out;
}

std::pair<std::vector<std::string>, RowIterator<double>> CsvManager::Load()
{
    _file->open(_filePath, std::ios_base::in);
    char *buffer = new char[1024];
    std::vector<std::string> &headers = *new std::vector<std::string>();
    std::vector<std::vector<double>> &values = *new std::vector<std::vector<double>>();
    while (!_file->eof())
    {
        _file->getline(buffer, 1024);
        auto &splitted = split(std::string(buffer), ';');
        if (headers.size() == 0)
        {
            for (auto val : splitted)
                headers.push_back(val);
        }
        else
        {
            std::vector<double> &row = *new std::vector<double>();
            for (auto &val : splitted)
                row.push_back(std::stod(val));
            values.push_back(row);
        }
    }
    _file->close();
    return std::pair<std::vector<std::string>, RowIterator<double>>(headers, *new RowIterator<double>(values));
}

CsvManager::CsvManager(const char *path)
{
    _file = new std::fstream();
    _filePath = path;
}

void CsvManager::Save(const std::vector<std::string> &headers, RowIterator<double> rows)
{
    _file->open(_filePath, std::ios_base::out | std::ios_base::trunc);
    auto headerItr = headers.begin();
    while (headerItr != headers.end())
    {
        std::string toWrite = *headerItr + (headerItr == headers.end() - 1 ? std::string("\n") : std::string(";"));
        _file->write(toWrite.c_str(), toWrite.size());
        headerItr++;
    }
    auto valueItr = rows;
    auto end = valueItr.end();
    while (valueItr != end)
    {
        std::string toWrite = "";
        auto row = *valueItr;
        auto rowItr = row.begin();
        while (rowItr != row.end())
        {
            toWrite += std::to_string(*rowItr) + (rowItr == row.end() - 1 ? "\n" : ";");
            rowItr++;
        }
        valueItr++;
        _file->write(toWrite.c_str(), toWrite.size());
    }
    _file->flush();
    _file->close();
}
