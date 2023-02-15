#include "LogEngine.hpp"
#include <cstdio>
#include <fstream>
#include <ios>
#include <istream>
#include <ostream>
#include <streambuf>
#include <string>



ConsoleLogEngine *ConsoleLogEngine::_instance = nullptr;

std::string &LogTypeToString(LogType logType)
{
    switch (logType)
    {
    case LogType::EXCEPTION:
        return *new std::string("[Exception]");
    case LogType::RESULT:
        return *new std::string("[Result]");
    case LogType::INFORMATION:
        return *new std::string("[Information]");
    case LogType::TRANSFER:
        return *new std::string("[Transfer]");
    case LogType::DEBUG:
        return *new std::string("[Debug]");
    }
}

void writeBuffer(std::istream * buffer, std::ostream * save)
{
    using namespace std;
    std::string line;
    while (std::getline(*buffer, line))
    {
        char statBuf[line.size() + 1];
        sprintf(statBuf, "%s\n",line.c_str());
        save->write(statBuf,line.size()+1);
    }
}

/**
 * @brief  Save the log file
 * @note   used RTTI on fstream  to save the log file so at the end of the function the stream will be closed
 * @retval None
 */
void ConsoleLogEngine::Finalize()
{
    using namespace std;
    fstream logFile(_logFile, ios_base::app);
    writeBuffer(&_buffer, &logFile);
    
}



void ConsoleLogEngine::Trace(LogType type, std::string &message)
{
    using namespace std;
    if (_verbosity >= (int)type)
    {
        std::string log = LogTypeToString(type) + message;
        char buffer[log.size() + 1];
        sprintf(buffer, "%s\n",log.c_str());
        printf("%s\n",log.c_str());
        _buffer.write(buffer, log.size()+1);
    }
}
