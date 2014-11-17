/**
 * Logger
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-05
 * Last Modified: 2014-11-05
 */

#ifndef LOGGING_H
#define LOGGING_H

#include <sstream>
#include <string>

#include "common.h"

/** Different levels for logging from lowest to highest */
enum LogLevel {
    ERROR = 0, WARNING, INFO, DEBUG, DEBUG1, DEBUG2, DEBUG3
};

/**
 * A simple static class for logging which is not thread-safe.
 */
class Logging {
 private:
    std::ostringstream os;
    static LogLevel reportingLevel;
    LogLevel messageLevel;

 public:
    /** Output when destroyed. */
    ~Logging() {
        if (messageLevel >= Logging::ReportingLevel()) {
            os << std::endl;
            fprintf(stderr, "%s", os.str().c_str());
            fflush(stderr);
        }
    }

    /**
     * Gets an out-string-stream at message level `mesgLevel`.
     * The caller to this function can append strings to this stream.
     */
    std::ostringstream& Get(LogLevel mesgLevel = INFO) {
        os << toString(mesgLevel) << ": ";
        messageLevel = mesgLevel;
        return os;
    }

    /** Accessing the reporting level */
    static LogLevel& ReportingLevel() {
        return reportingLevel;
    }

 private:
    std::string toString(LogLevel level) {
        std::string strLevel;
        switch (level) {
        case ERROR:    strLevel = std::string("ERROR"); break;
        case WARNING:  strLevel = std::string("WARNING"); break;
        case INFO:     strLevel = std::string("INFO"); break;
        case DEBUG:    strLevel = std::string("DEBUG"); break;
        case DEBUG1:   strLevel = std::string("DEBUG1"); break;
        case DEBUG2:   strLevel = std::string("DEBUG2"); break;
        case DEBUG3:   strLevel = std::string("DEBUG3"); break;
        default:       strLevel = std::string("UNKNOWN");
        }
        return strLevel;
    }
};

/** Macro for easy use  */
LogLevel Logging::reportingLevel = ERROR;

/** Macro for easy use  */
#define LOG(level) Logging().Get(level)


#endif  // LOGGING_H


