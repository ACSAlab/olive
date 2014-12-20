/**
 * Unit test for the logging utility
 *
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-15
 * Last Modified: 2014-11-15
 */

#include "logging.h"
#include "unitest_common.h"

void all_levels(void) {
    Logging().Get(ERROR) << "error: " << 1;
    Logging().Get(WARNING) << "warning: " << 2;
    Logging().Get(INFO) << "info: " << 3;
    Logging().Get(DEBUG) << "debug: " << 4;
    Logging().Get(DEBUG1) << "debug1: " << 5;
    Logging().Get(DEBUG2) << "debug2: " << 6;
    Logging().Get(DEBUG3) << "debug3: " << 7;
}


int main(int argc, char **arg) {
    printf("testing logging (all)\n");
    all_levels();

    printf("testing logging (above warning)\n");
    Logging::ReportingLevel() = WARNING;
    all_levels();

    printf("testing logging (above info)\n");
    Logging::ReportingLevel() = INFO;
    all_levels();
    return 0;
}
