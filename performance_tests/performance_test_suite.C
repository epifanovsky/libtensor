#include <iomanip>
#include <sstream>
#include <typeinfo>
#include "performance_test_suite.h"

namespace libtensor {


performance_test_suite::performance_test_suite(const char* name) :
        libtest::test_suite(name), m_ntests(0) {

}


void performance_test_suite::add_tests(const char* name, const char* comment,
        performance_test_scenario_i& pts) {

    for ( size_t i=0; i<pts.size(); i++ ) {
        m_ntests++;
        std::ostringstream fullname;
        fullname << std::setfill('0') << std::setw(3) << m_ntests << ": ";
        fullname << name << " - " << pts.test_name(i) << std::endl;
        fullname << " ... " << pts.test_description(i) << std::endl;
        fullname << " ... " << comment;
        add_test(fullname.str().c_str(),pts.test(i));
    }
}


}

